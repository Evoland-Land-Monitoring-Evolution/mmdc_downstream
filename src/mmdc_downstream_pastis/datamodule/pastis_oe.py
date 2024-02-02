# Taken from https://github.com/VSainteuf/utae-paps/blob/3d83dc50e67e7ae204559e819cfbb432b1b21e10/src/dataset.py#L85
# some adaptations were made
import logging
import os
from datetime import datetime
from typing import Literal

from dataclasses import dataclass

import torch.utils.data as tdata
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from datatypes import *

from mmdc_downstream_pastis.constant.dataset import S2_BAND  # TODO change
from mmdc_downstream_pastis.constant.pastis import LABEL_PASTIS
from pytorch_lightning import LightningDataModule
from mmdc_downstream_pastis.datamodule.utils import OutClassifItem, apply_padding, MMDCDataStruct, randomcropindex

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(level=NUMERIC_LEVEL,
                    format="%(asctime)-15s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class PASTISDataset(tdata.Dataset):

    def __init__(
        self,
        options: PASTISOptions,
        reference_date: str = "2018-10-01",
        sats=["S2"],
        crop_size=64,
        # s2_band=None,
        # dict_classes=None,
        crop_type: Literal["Center", "Random"] = "Random",
        transform: nn.Module | None = None,
        max_len: int = 60,
        allow_padd: bool = True,
        extract_true_doy=False,
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
        path_subdataset (None or str). if not None indicates a path to a csv which contains a column id_patch. All the patches from get_item will be from this subset
        """

        super().__init__()

        self.extract_true_doy = extract_true_doy
        self.allow_pad = allow_padd
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

        self.max_len = max_len
        self.sats = sats

        # Get metadata
        meta_patch = self.get_metadata()
        self.id_patches = meta_patch.index
        self.len = len(self.id_patches)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 5000))

        # self.dict_classes = dict_classes if dict_classes is not None else LABEL_PASTIS
        self.dict_classes = LABEL_PASTIS

        self.labels = list(self.dict_classes.values())

        logger.info("Done.")

        # TODO change norm
        # # Get normalisation values
        # if norm:
        #     self.norm = {}
        #     for s in self.sats:
        #         with open(
        #             os.path.join(dataset_path_pastis, f"NORM_{s}_patch.json"),
        #         ) as file:
        #             normvals = json.loads(file.read())
        #         selected_folds = folds if folds is not None else range(1, 6)
        #         means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
        #         stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
        #         self.norm[s] = np.stack(means).mean(axis=0), np.stack(
        #             stds
        #         ).mean(axis=0)
        #         self.norm[s] = (
        #             torch.from_numpy(self.norm[s][0]).float(),
        #             torch.from_numpy(self.norm[s][1]).float(),
        #         )
        # else:
        #     self.norm = None

        print("Dataset ready.")

    # TODO adapt for three sensors
    def patches_to_keep(self) -> np.array:
        """
        We make lists of all available patches for a satellite and corresponding meteo and dem data.
        And we choose only patches presents for all modalities.
        In case we use several satellites, we choose patches present for all of them.
        """
        path = self.options.dataset_path_oe
        valid_patches = []
        for satellite in self.sats:
            sat_patches = [
                int(file[:-3].split("_")[-1])
                for file in os.listdir(os.path.join(path, satellite))]

            meteo_patches = [int(file[:-3].split("_")[-1])
                             for file in os.listdir(
                                 os.path.join(path, f"{satellite}_METEO"))]

            dem_patches = [int(file[:-3].split("_")[-1])
                           for file in os.listdir(os.path.join(path, "DEM"))]

            valid = np.intersect1d(np.intersect1d(sat_patches, meteo_patches, assume_unique=True),
                                   dem_patches, assume_unique=True)
            if len(self.sats) == 1:
                return valid

            valid_patches.append(valid)

        return np.intersect1d(valid_patches[0], valid_patches[1], assume_unique=True)

    def get_metadata(self) -> gpd.GeoDataFrame:
        """Return a GeoDatFrame with the patch metadata"""
        logger.info("Reading patch metadata . . .")
        meta_patch = gpd.read_file(
            os.path.join(self.options.dataset_path_pastis, "metadata.geojson"))
        meta_patch.index = meta_patch["ID_PATCH"].astype(int)
        meta_patch.sort_index(inplace=True)

        # Select Fold samples
        if self.options.folds is not None:
            meta_patch = pd.concat([
                meta_patch[meta_patch["Fold"] == f] for f in self.options.folds
            ])

        # We only keep patches with all modalities available
        meta_patch = meta_patch[meta_patch.ID_PATCH.isin(self.patches_to_keep())]
        return meta_patch

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(
            self.date_tables[sat][id_patch] == 1)[0]]

    def read_data_from_disk(
        self,
        item: int,
        id_patch: int,
    ) -> tuple[dict[str, MMDCDataStruct], torch.Tensor, dict[
            str, torch.Tensor], dict[str, torch.Tensor]]:
        """Get id_patch from disk and cache it as item in the dataset"""
        data_pastis = {
            satellite:
            torch.load(
                os.path.join(
                    self.options.dataset_path_oe,
                    satellite,
                    f"{satellite}_{id_patch}.pt",
                ))
            for satellite in self.sats
        }  # pastis_eo.dataclass.PASTISItem
        data: dict[str, MMDCDataStruct] = {}
        for satellite in self.sats:
            data_all = data_pastis[satellite]
            dem = torch.load(
                os.path.join(
                    self.options.dataset_path_oe,
                    "DEM",
                    f"DEM_{id_patch}.pt",
                ))
            data_one_sat = {
                "img": data_all.sits,
                "mask": data_all.mask,
                "angles": data_all.angles.nan_to_num(),
                "dem": dem.nan_to_num()
            }

            data_one_sat.update({
                f"meteo_{k}":
                torch.load(
                    os.path.join(
                        self.options.dataset_path_oe,
                        f"{satellite}_METEO",
                        f"{satellite}_{k}_{id_patch}.pt",
                    ))
                for k in METEO_BANDS
            })

            data[satellite] = MMDCDataStruct.init_empty().fill_empty_from_dict(
                data_one_sat).concat_meteo()

        # Retrieve date sequences
        dates = {s: a.doy for s, a in data_pastis.items()}
        true_doys = {s: a.true_doy for s, a in data_pastis.items()}
        if self.options.task == "semantic":
            target = np.load(
                os.path.join(
                    self.options.dataset_path_pastis,
                    "ANNOTATIONS",
                    f"TARGET_{id_patch}.npy",
                ))
            target = torch.from_numpy(target[0].astype(int))
        else:
            raise NotImplementedError
        if self.options.cache:  # Not sure it works
            if self.options.mem16:
                self.memory[item] = [
                    {
                        k: v.casting(torch.float32)
                        for k, v in data.items()
                    },
                    target,
                    {
                        k: d
                        for k, d in dates.items()
                    },
                    {
                        k: d
                        for k, d in true_doys.items()
                    },
                ]
            else:
                self.memory[item] = [data, target, dates, true_doys]

        return data, target, dates, true_doys

    def __getitem__(self, item) -> OutClassifItem:
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.options.cache or item not in self.memory.keys():
            data, target, dates, true_doys = self.read_data_from_disk(
                item, id_patch)

        else:
            data, target, dates, true_doys = self.memory[item]
            if self.options.mem16:
                data = {k: v.casting(torch.float32) for k, v in data.items()}

        if len(self.sats) == 1:  # TODO change that for multimodal
            sits = data[self.sats[0]]
            doy = dates[self.sats[0]]
            if self.extract_true_doy:
                true_doy = true_doys[self.sats[0]]
            else:
                true_doy = None
            t, c, h, w = sits.data.img.shape
            y, x = self.get_crop_idx(
                rows=sits.data.img.shape[2], cols=sits.data.img.shape[3]
            )  # so that same crop for all the bands of a sits

            sits = sits.crop(x, y, self.crop_size)
            target = target[y:y + self.crop_size, x:x + self.crop_size]
            target[target == 19] = 0  # merge background and void class
            # if self.transform is not None:
            #     my_logger.debug("Apply transform per SITS")
            #     sits = rearrange(sits, "t c h w -> c t h w")
            #     sits = self.transform(sits)
            #     sits = rearrange(sits, "c t h w -> t c h w")
            sits, doy, padd_index = apply_padding(self.allow_pad, self.max_len,
                                                  t, sits, doy)
            if true_doy is not None:
                padd_true_doy = F.pad(true_doy, (0, self.max_len - t))
            else:
                padd_true_doy = None
            return OutClassifItem(
                sits=sits,
                input_doy=doy.to(sits.data.img.device),
                padd_index=padd_index.bool(),
                target=target,
                padd_val=torch.tensor(self.max_len - t),
                item=item,
                mask=(target != 0) & (target != 19),
                true_doy=padd_true_doy,
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

        return int(rows // 2 - self.crop_size // 2), int(cols // 2 -
                                                         self.crop_size // 2)

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


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(lambda x: (datetime(int(str(x)[:4]), int(str(x)[4:6]),
                                       int(str(x)[6:])) - reference_date).days)
    return d.values


def custom_collate_classif(batch: Iterable[OutClassifItem]) -> ClassifBInput:
    dict_collate = {}
    # TODO add padding here
    for key in batch[0].__dict__:
        if type(getattr(batch[0], key)) is MMDCDataStruct:
            to_collate = [getattr(b, key) for b in batch]
            sits_col = {}
            for k in to_collate[0].__dict__:
                if type(getattr(to_collate[0], k)) is torch.Tensor:
                    sits_col[k] = torch.stack(
                        [getattr(c, k) for c in to_collate])
                else:
                    #type is MMDCData
                    to_collate2 = [getattr(c, k) for c in to_collate]
                    # sits_col[k] = {k2: torch.stack([getattr(c, k2) for c in to_collate2])
                    #                for k2 in to_collate2[0].__dict__}
                    sits_col.update({
                        k2:
                        torch.stack([getattr(c, k2) for c in to_collate2])
                        for k2 in to_collate2[0].__dict__
                    })
            dict_collate[key] = MMDCDataStruct.init_empty_concat_meteo(
            ).fill_empty_from_dict(sits_col)
        elif type(getattr(batch[0], key)) is torch.Tensor:
            dict_collate[key] = torch.stack([getattr(b, key) for b in batch])

        elif getattr(batch[0], key) is None:
            dict_collate[key] = None

    return ClassifBInput.init_empty().fill_empty_from_dict(dict_collate)


class PastisDataModule(LightningDataModule):
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
        folds: PastisFolds,
        sats: list[str] = ["S2"],
        reference_date: str = "2018-10-01",
        task: Literal["semantic"] = "semantic",
        batch_size: int = 2,
        crop_size=64,
        crop_type: Literal["Center", "Random"] = "Random",
        num_workers: int = 1,
        # path_dir_csv: str | None = None,
        max_len: int = 60,
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

        self.sats = sats

        self.num_workers = num_workers
        self.max_len = max_len

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
                PASTISDataset(PASTISOptions(
                    task=self.task,
                    folds=self.folds.train,
                    dataset_path_oe=self.dataset_path_oe,
                    dataset_path_pastis=self.dataset_path_pastis,
                ),
                              sats=self.sats,
                              reference_date=self.reference_date,
                              crop_size=self.crop_size,
                              crop_type=self.crop_type,
                              max_len=self.max_len),
                PASTISDataset(PASTISOptions(
                    task=self.task,
                    folds=self.folds.val,
                    dataset_path_oe=self.dataset_path_oe,
                    dataset_path_pastis=self.dataset_path_pastis,
                ),
                              sats=self.sats,
                              reference_date=self.reference_date,
                              crop_size=self.crop_size,
                              crop_type=self.crop_type,
                              max_len=self.max_len),
            )
        if stage == "test":
            assert self.data
            self.data = PastisDataSets(
                self.data.train,
                self.data.val,
                PASTISDataset(PASTISOptions(
                    task=self.task,
                    folds=self.folds.test,
                    dataset_path_oe=self.dataset_path_oe,
                    dataset_path_pastis=self.dataset_path_pastis,
                ),
                              sats=self.sats,
                              reference_date=self.reference_date,
                              crop_size=self.crop_size,
                              crop_type=self.crop_type,
                              max_len=self.max_len),
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
        return self.instanciate_data_loader(self.data.test, shuffle=False)

    def predict_dataloader(self) -> tdata.DataLoader:
        """Predict dataloader"""
        return self.test_dataloader()

    def instanciate_data_loader(
        self,
        dataset: tdata.Dataset,
        shuffle: bool = False,
    ) -> tdata.DataLoader:
        """Return a data loader with the PASTIS data set"""

        return tdata.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=lambda x: custom_collate_classif(x),
        )

#
# def build_dm(sats) -> PastisDataModule:
#     """Builds datamodule"""
#     return PastisDataModule(dataset_path_oe=dataset_path_oe,
#                             dataset_path_pastis=dataset_path_pastis,
#                             folds=PastisFolds([1, 2, 3], [4], [5]),
#                             sats=sats,
#                             task="semantic",
#                             batch_size=1)
#
#
# dataset_path_oe = "/work/CESBIO/projects/DeepChange/Ekaterina/Pastis_OE"
# dataset_path_pastis = "/work/CESBIO/projects/DeepChange/Iris/PASTIS"
#
# def pastisds_dataloader(sats) -> None:
#     """Use a dataloader with PASTIS dataset"""
#     dm = build_dm(sats)
#     dm.setup(stage='fit')
#     dm.setup(stage='test')
#     assert hasattr(dm, 'train_dataloader') and \
#            hasattr(dm, 'val_dataloader') and hasattr(dm, 'test_dataloader')   # type: ignore[truthy-function]
#     for loader in (dm.train_dataloader(), dm.val_dataloader(),
#                    dm.test_dataloader()):
#         assert loader
#         for ((inputs, dates), labels), _ in zip(loader, range(4)):
#             assert list(inputs.keys()) == sats
#             assert list(dates.keys()) == sats
#             for sat in sats:
#                 if sat == "S2_MSK":
#                     b_s_x, t_s_x, p_s_x, _ = inputs[sat].shape
#                     nb_b = 1
#                 else:
#                     b_s_x, t_s_x, nb_b, p_s_x, _ = inputs[sat].shape
#                 b_s_d, t_s_d = dates[sat].shape
#                 b_s_y, p_s_y, _ = labels.shape
#                 assert t_s_d > 2
#                 assert b_s_x == b_s_d == b_s_y
#                 assert t_s_d == t_s_x
#                 # assert nb_b == PASTIS_BANDS[sat]
#                 assert p_s_x == p_s_y
#
# pastisds_dataloader(["S2"])
