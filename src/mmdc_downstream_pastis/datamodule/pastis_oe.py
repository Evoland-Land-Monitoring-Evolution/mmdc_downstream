# Taken from
# https://github.com/VSainteuf/utae-paps/blob/3d83dc50e67e7ae204559e819cfbb432b1b21e10/src/dataset.py#L85
# some adaptations were made
import logging
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from pytorch_lightning import LightningDataModule
from torch import nn

from mmdc_downstream_pastis.constant.dataset import S2_BAND  # TODO change
from mmdc_downstream_pastis.constant.pastis import LABEL_PASTIS
from mmdc_downstream_pastis.datamodule.datatypes import (
    METEO_BANDS,
    BatchInputUTAE,
    MMDCDataStruct,
    OneSatellitePatch,
    PastisBatch,
    PastisDataSets,
    PastisFolds,
    PASTISOptions,
    randomcropindex,
)
from mmdc_downstream_pastis.datamodule.pastis_oe_only_satellite import pad_tensor

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class PASTISDataset(tdata.Dataset):
    def __init__(
        self,
        options: PASTISOptions,
        reference_date: str = "2018-09-01",
        sats: list[str] = ["S2"],
        crop_size: int | None = 64,
        strict_s1: bool = False,
        # s2_band=None,
        crop_type: Literal["Center", "Random"] = "Random",
        transform: nn.Module | None = None,
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.

            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
        """

        super().__init__()

        if sats is None:
            sats = ["S2"]
        # if s2_band is None:
        #     self.s2_band = S2_BAND
        # else:
        #     self.s2_band = s2_band

        self.s2_band = S2_BAND

        self.crop_size = crop_size
        self.crop_type = crop_type

        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.options = options

        self.memory = {}
        self.memory_dates = {}

        self.sats = sats

        self.strict_s1 = strict_s1

        # Get metadata
        meta_patch = self.get_metadata()
        self.id_patches = meta_patch.index
        self.len = len(self.id_patches)

        # self.dict_classes = dict_classes if dict_classes is not None else LABEL_PASTIS
        self.dict_classes = LABEL_PASTIS

        self.patch_size = None

        self.labels = list(self.dict_classes.values())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Done.")

        print("Dataset ready.")

    def get_one_satellite_patches(self, satellite: str) -> np.array:  # TODO change
        """
        For each satellite, we check that all the modalities are available for a patch
        """
        path = self.options.dataset_path_oe
        sat_patches = [
            int(file[:-3].split("_")[-1])
            for file in os.listdir(os.path.join(path, satellite))
            if file.endswith(".pt")
        ]
        meteo_patches = {
            m: [
                int(file[:-3].split("_")[-1])
                for file in os.listdir(os.path.join(path, f"{satellite}_METEO"))
                if m in file and file.endswith(".pt")
            ]
            for m in METEO_BANDS
        }

        valid_meteo_patches = [
            patch
            for patch in meteo_patches[list(meteo_patches.keys())[0]]
            if all(patch in mp for mp in meteo_patches.values())
        ]

        dem_patches = [
            int(file[:-3].split("_")[-1])
            for file in os.listdir(os.path.join(path, "DEM"))
            if file.endswith(".pt")
        ]

        return np.intersect1d(
            np.intersect1d(sat_patches, valid_meteo_patches, assume_unique=False),
            dem_patches,
            assume_unique=True,
        )

    # TODO adapt for three sensors
    def patches_to_keep(self) -> np.array:
        """
        We make lists of all available patches for a satellite
        and corresponding meteo and dem data.
        And we choose only patches presents for all modalities.
        In case we use several satellites, we choose patches present for all of them.
        """

        valid_patches = []

        if len(self.sats) == 1:
            return self.get_one_satellite_patches(self.sats[0])

        elif len(self.sats) == 2:
            for satellite in self.sats:
                valid = self.get_one_satellite_patches(satellite)
                valid_patches.append(valid)

            if (
                (np.sort(self.sats) == ["S1_ASC", "S2"]).all()
                or (np.sort(self.sats) == ["S1_DESC", "S2"]).all()
                or self.strict_s1
            ):
                return np.intersect1d(
                    valid_patches[0], valid_patches[1], assume_unique=True
                )
            else:  # ["S1_ASC", "S1_DESC"]
                return np.union1d(valid_patches[0], valid_patches[1])

        else:
            for satellite in np.sort(self.sats):
                valid = self.get_one_satellite_patches(satellite)
                valid_patches.append(valid)
            if not self.strict_s1:
                return np.union1d(
                    np.intersect1d(
                        valid_patches[0], valid_patches[2], assume_unique=True
                    ),
                    np.intersect1d(
                        valid_patches[1], valid_patches[2], assume_unique=True
                    ),
                )
            else:
                return np.intersect1d(
                    np.intersect1d(
                        valid_patches[0], valid_patches[2], assume_unique=True
                    ),
                    valid_patches[1],
                    assume_unique=True,
                )

    def get_metadata(self) -> gpd.GeoDataFrame:
        """Return a GeoDatFrame with the patch metadata"""
        logger.info("Reading patch metadata . . .")
        meta_patch = gpd.read_file(
            os.path.join(self.options.dataset_path_pastis, "metadata.geojson")
        )
        meta_patch.index = meta_patch["ID_PATCH"].astype(int)
        meta_patch.sort_index(inplace=True)

        # Select Fold samples
        if self.options.folds is not None:
            meta_patch = pd.concat(
                [meta_patch[meta_patch["Fold"] == f] for f in self.options.folds]
            )

        # We only keep patches with all modalities available
        meta_patch = meta_patch[meta_patch.ID_PATCH.isin(self.patches_to_keep())]
        return meta_patch

    def __len__(self):
        return self.len

    def load_file(self, path: str):
        """Load file if exists"""
        # logger.info(path)
        return (
            torch.load(path, map_location=torch.device("cpu"))
            if Path(path).exists()
            else None
        )

    def read_data_from_disk(
        self,
        item: int,
        id_patch: int,
    ) -> tuple[dict[str, MMDCDataStruct], torch.Tensor, dict[str, torch.Tensor],]:
        """Get id_patch from disk and cache it as item in the dataset"""
        data_pastis = {
            satellite: self.load_file(
                os.path.join(
                    self.options.dataset_path_oe,
                    satellite,
                    f"{satellite}_{id_patch}.pt",
                )
            )
            for satellite in self.sats
        }  # pastis_eo.dataclass.PASTISItem
        data: dict[str, MMDCDataStruct] = {}
        for satellite in self.sats:
            data_all = data_pastis[satellite]
            if self.patch_size is None:
                self.patch_size = data_all.sits.shape[-1]
            if data_all is not None:
                dem = self.load_file(
                    os.path.join(
                        self.options.dataset_path_oe,
                        "DEM",
                        f"DEM_{id_patch}.pt",
                    )
                )
                data_one_sat = {
                    "img": data_all.sits,
                    "mask": data_all.mask,
                    "angles": data_all.angles.nan_to_num(),
                    "dem": dem.nan_to_num(),
                }
                data_one_sat.update(
                    {
                        f"meteo_{k}": self.load_file(
                            os.path.join(
                                self.options.dataset_path_oe,
                                f"{satellite}_METEO",
                                f"{satellite}_{k}_{id_patch}.pt",
                            )
                        )
                        for k in METEO_BANDS
                    }
                )

                data[satellite] = (
                    MMDCDataStruct.init_empty()
                    .fill_empty_from_dict(data_one_sat)
                    .concat_meteo()
                )
            else:
                data[satellite] = None

        # Retrieve date sequences
        true_doys = {
            s: (self.prepare_dates(a.dates_dt.values[:, 0]) if a is not None else None)
            for s, a in data_pastis.items()
        }
        if self.options.task == "semantic":
            target = np.load(
                os.path.join(
                    self.options.dataset_path_pastis,
                    "ANNOTATIONS",
                    f"TARGET_{id_patch}.npy",
                )
            )
            target = torch.from_numpy(target[0].astype(int))
        else:
            raise NotImplementedError
        if self.options.cache:  # Not sure it works
            if self.options.mem16:
                self.memory[item] = [
                    {k: v.casting(torch.float32) for k, v in data.items()},
                    target,
                    {k: d for k, d in true_doys.items()},
                ]
            else:
                self.memory[item] = [data, target, true_doys]

        return data, target, true_doys

    def __getitem__(
        self, item: int
    ) -> (dict[str, OneSatellitePatch], torch.Tensor, torch.Tensor, int):
        """Get one item"""
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.options.cache or item not in self.memory.keys():
            data, target, true_doys = self.read_data_from_disk(item, id_patch)

        else:
            data, target, true_doys = self.memory[item]
            if self.options.mem16:
                data = {k: v.casting(torch.float32) for k, v in data.items()}

        if self.crop_size is not None:
            t, c, h, w = data[self.sats[0]].data.img.shape
            y, x = self.get_crop_idx(
                rows=h, cols=w
            )  # so that same crop for all the bands of a sits
            output = {
                sat: self.return_one_satellite(data, true_doys, sat, [x, y])
                for sat in self.sats
            }
            target = target[y : y + self.crop_size, x : x + self.crop_size]
        else:
            output = {
                sat: self.return_one_satellite(data, true_doys, sat)
                for sat in self.sats
            }
        target[target == 19] = 0  # merge background and void class
        mask = (target != 0) & (target != 19)

        return output, target, mask, id_patch

    def prepare_dates(self, true_doys: np.array) -> torch.Tensor:
        """
        Transform date to day of interest relative to ref date
        """
        if not torch.is_tensor(true_doys):
            return torch.Tensor(
                (pd.to_datetime(true_doys) - pd.to_datetime(self.reference_date)).days
            )
        return true_doys

    def return_one_satellite(self, data, true_doys, sat, crop_xy: list[int] = None):
        if (sits := data[sat]) is not None:
            if crop_xy is not None:
                x, y = crop_xy
                sits = sits.crop(x, y, self.crop_size)
            true_doy = true_doys[sat]
            # if self.transform is not None:
            #     my_logger.debug("Apply transform per SITS")
            #     sits = rearrange(sits, "t c h w -> c t h w")
            #     sits = self.transform(sits)
            #     sits = rearrange(sits, "c t h w -> t c h w")

            return OneSatellitePatch(
                sits=sits,
                true_doy=true_doy,
            )
        return OneSatellitePatch(
            sits=MMDCDataStruct.init_empty_zeros_s1(
                1, self.patch_size, self.patch_size
            ),
            true_doy=torch.zeros(1, 1),
        )

    def get_crop_idx(self, rows, cols) -> tuple[int, int]:
        """return the coordinate, width and height of the window loaded
        by the SITS depending od the value of the attribute
        self.crop_type

        Args:
            rows: int
            cols: int
        """
        if self.crop_type == "Random":
            return randomcropindex(rows, cols, self.crop_size, self.crop_size)

        return int(rows // 2 - self.crop_size // 2), int(
            cols // 2 - self.crop_size // 2
        )

    def extract_class_count(self, workers):
        M = self.len
        # df = pd.DataFrame(columns=[i for i in range(len(self.dict_classes))])
        l_s = []
        for i in range(M):
            out = self.__getitem__(i)
            labels = out["labels"].cpu().numpy()
            values, counts = np.unique(labels, return_counts=True)
            s_counts = pd.Series(dict(zip(values, counts)))
            s_counts.name = self.id_patches[i]
            l_s += [s_counts]
        df = pd.concat(l_s)
        return df


def pad_and_stack(
    key: str, to_collate: list, pad_value: int = 0, max_size: int = -1
) -> torch.Tensor:
    return torch.stack(
        [pad_tensor(getattr(c, key), max_size, pad_value) for c in to_collate]
    )


def collate(item: Any, pad_value: int = 0) -> dict[str, Any]:
    dict_collate = {}
    sizes = [
        (getattr(e, key)).shape[0] for e in item for key in e.__dict__ if "doy" in key
    ]
    max_size = (
        max(sizes)
        if (len(sizes) > 1 and not sizes.count(sizes[0]) == len(sizes))
        else -1
    )
    for key in item[0].__dict__:
        if type(getattr(item[0], key)) is MMDCDataStruct:
            to_collate = [getattr(b, key) for b in item]
            sits_col = {}
            for k in to_collate[0].__dict__:
                if type(getattr(to_collate[0], k)) is torch.Tensor:
                    if k == "dem":
                        sits_col[k] = pad_and_stack(
                            k, to_collate, pad_value, max_size=-1
                        )
                    else:
                        sits_col[k] = pad_and_stack(k, to_collate, pad_value, max_size)
                else:
                    # type is MMDCData
                    to_collate2 = [getattr(c, k) for c in to_collate]
                    sits_col.update(
                        {
                            k2: pad_and_stack(k2, to_collate2, pad_value, max_size)
                            for k2 in to_collate2[0].__dict__
                        }
                    )
            dict_collate[
                key
            ] = MMDCDataStruct.init_empty_concat_meteo().fill_empty_from_dict(
                sits_col, flatten_meteo=True
            )
        elif type(getattr(item[0], key)) is torch.Tensor:
            dict_collate[key] = pad_and_stack(key, item, pad_value, max_size)
        elif getattr(item[0], key) is None:
            dict_collate[key] = None
    return dict_collate


def custom_collate_classif(
    batch: Iterable[dict[str, OneSatellitePatch], torch.Tensor, torch.Tensor, int],
    pad_value: int = 0,
) -> (dict[str, PastisBatch], torch.Tensor, torch.Tensor, list[int]):
    batch_dict = {}

    sat_dict, target, mask, id_patch = zip(*batch)
    target = torch.stack(target, 0)
    mask = torch.stack(mask, 0)

    for sat in sat_dict[0]:
        item = [v[sat] for v in sat_dict]
        dict_collate = collate(item, pad_value)
        batch_dict[sat] = PastisBatch.init_empty().fill_empty_from_dict(dict_collate)
    return BatchInputUTAE(sits=batch_dict, gt=target, gt_mask=mask, id_patch=id_patch)


class PastisOEDataModule(LightningDataModule):
    """
    A DataModule implements 4 key methods:
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(
        self,
        dataset_path_oe: str,
        dataset_path_pastis: str,
        folds: PastisFolds | None,
        sats: list[str] = ["S2"],
        reference_date: str = "2018-09-01",
        task: Literal["semantic"] = "semantic",
        batch_size: int = 2,
        strict_s1: bool | None = None,
        crop_size: int | None = None,
        crop_type: Literal["Center", "Random"] = "Random",
        num_workers: int = 1,
        pad_value: int = 0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # init different variables
        self.dataset_path_oe = dataset_path_oe
        self.dataset_path_pastis = dataset_path_pastis

        self.crop_size = crop_size
        self.crop_type = crop_type

        self.reference_date = reference_date

        self.folds = folds
        self.batch_size = batch_size
        self.task = task

        self.strict_s1 = strict_s1

        self.sats = sats

        self.num_workers = num_workers

        self.pad_value = pad_value

        self.data: PastisDataSets | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables:
        `self.data.train`, `self.data.val`, `self.data.test`.

        This method is called by lightning when
        doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice!
        The `stage` can be used to differentiate whether
        it's called before trainer.fit()` or `trainer.test()`.
        """
        if stage == "fit":
            self.data = PastisDataSets(
                self.instanciate_dataset(self.folds.train),
                self.instanciate_dataset(self.folds.val),
            )
        if stage == "test":
            assert self.data
            self.data = PastisDataSets(
                self.data.train,
                self.data.val,
                self.instanciate_dataset(self.folds.test),
            )

    def train_dataloader(self) -> tdata.DataLoader:
        """Train dataloader"""
        assert self.data is not None
        assert self.data.train is not None
        return self.instanciate_data_loader(self.data.train, shuffle=True)

    def val_dataloader(self) -> tdata.DataLoader:
        """Validation dataloader"""
        assert self.data is not None
        assert self.data.val is not None
        return self.instanciate_data_loader(self.data.val, shuffle=False)

    def test_dataloader(self) -> tdata.DataLoader:
        """Test dataloader"""
        assert self.data is not None
        assert self.data.test is not None
        return self.instanciate_data_loader(
            self.data.test, shuffle=False, drop_last=False
        )

    def predict_dataloader(self) -> tdata.DataLoader:
        """Predict dataloader"""
        return self.test_dataloader()

    def instanciate_dataset(self, fold: list[int] | None) -> PASTISDataset:
        return PASTISDataset(
            PASTISOptions(
                task=self.task,
                folds=fold,
                dataset_path_oe=self.dataset_path_oe,
                dataset_path_pastis=self.dataset_path_pastis,
            ),
            sats=self.sats,
            reference_date=self.reference_date,
            crop_size=self.crop_size,
            crop_type=self.crop_type,
            strict_s1=self.strict_s1,
        )

    def instanciate_data_loader(
        self, dataset: tdata.Dataset, shuffle: bool = False, drop_last: bool = True
    ) -> tdata.DataLoader:
        """Return a data loader with the PASTIS data set"""

        return tdata.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: custom_collate_classif(x, pad_value=self.pad_value),
        )


# def build_dm(sats) -> PastisDataModule:
#     """Builds datamodule"""
#     return PastisDataModule(
#         dataset_path_oe=dataset_path_oe,
#         dataset_path_pastis=dataset_path_pastis,
#         folds=PastisFolds([1, 2, 3], [4], [5]),
#         sats=sats,
#         task="semantic",
#         batch_size=1,
#         max_len=0
#     )
#
#
# dataset_path_oe = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis_OE"
# dataset_path_pastis = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"
#
#
# def pastisds_dataloader(sats) -> None:
#     """Use a dataloader with PASTIS dataset"""
#     dm = build_dm(sats)
#     dm.setup(stage="fit")
#     dm.setup(stage="test")
#     assert (
#         hasattr(dm, "train_dataloader")
#         and hasattr(dm, "val_dataloader")
#         and hasattr(dm, "test_dataloader")
#     )  # type: ignore[truthy-function]
#     for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
#         assert loader
#         for (batch_dict, target, mask, id_patch), _ in zip(loader, range(4)):
#             assert list(batch_dict.keys()) == sats
#             for sat in sats:
#                 b_s_x, t_s_x, nb_b, p_s_x, _ = batch_dict[sat].sits.data.img.shape
#                 b_s_ang, t_s_ang, nb_b_ang, p_s_ang, _ = batch_dict[
#                     sat
#                 ].sits.data.angles.shape
#                 b_s_imsk, t_s_imsk, nb_b_imsk, p_s_imsk, _ = batch_dict[
#                     sat
#                 ].sits.data.mask.shape
#                 b_s_dem, nb_b_dem, p_s_dem, _ = batch_dict[sat].sits.dem.shape
#                 b_s_meteo, t_s_meteo, nb_b_meteo, p_s_meteo, _ = batch_dict[
#                     sat
#                 ].sits.meteo.shape
#                 b_s_y, p_s_y, _ = target.shape
#                 b_s_m, p_s_m, _ = mask.shape
#                 b_s_id = len(id_patch)
#                 assert nb_b_dem == 4
#                 assert nb_b_meteo == 48
#                 assert (
#                     b_s_x
#                     == b_s_ang
#                     == b_s_imsk
#                     == b_s_dem
#                     == b_s_meteo
#                     == b_s_y
#                     == b_s_m
#                     == b_s_id
#                 )
#                 assert t_s_x == t_s_ang == t_s_imsk == t_s_meteo
#                 # assert nb_b == PASTIS_BANDS[sat]
#                 assert (
#                     p_s_x
#                     == p_s_ang
#                     == p_s_imsk
#                     == p_s_dem
#                     == p_s_meteo
#                     == p_s_y
#                     == p_s_m
#                 )
#
#
# pastisds_dataloader(["S1_ASC", "S1_DESC"])
