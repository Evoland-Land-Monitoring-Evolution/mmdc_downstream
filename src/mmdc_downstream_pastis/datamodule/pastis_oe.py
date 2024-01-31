# Taken from https://github.com/VSainteuf/utae-paps/blob/3d83dc50e67e7ae204559e819cfbb432b1b21e10/src/dataset.py#L85 , some adaptations were made
import json
import logging
import os
from datetime import datetime
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


from ..constant.dataset import S2_BAND # TODO change
from ..constant.pastis import LABEL_PASTIS
from .template_dataset import TemplateDataset
from .utils import OutClassifItem, apply_padding, MMDCDataStruct

my_logger = logging.getLogger(__name__)

@dataclass
class PASTISOptions:
    """
    folder (str): Path to the dataset
    folds (list, optional): List of ints specifying which of the 5 official
        folds to load. By default (when None is specified) all folds are loaded.
    norm (bool): If true, images are standardised using pre-computed
        channel-wise means and standard deviations.
    cache (bool): If True, the loaded samples stay in RAM, default False.
    mem16 (bool): Additional argument for cache. If True, the image time
        series tensors are stored in half precision in RAM for efficiency.
        They are cast back to float32 when returned by __getitem__.
    """

    # task: PASTISTask
    target: Literal["semantic"] = "semantic",
    dataset_path_oe: str
    dataset_path_pastis: str
    # folder: str
    folds: list[int] | None = None
    norm: bool = True
    cache: bool = False
    mem16: bool = False


class PASTIS_Dataset_OE(TemplateDataset):
    def __init__(
            self,
            dataset_path_oe: str,
            dataset_path_pastis: str,
            norm: bool = False,
            target: Literal["semantic"] = "semantic",
            cache: bool = False,
            mem16: bool = False,
            folds: None | list = None,
            reference_date: str = "2014-03-03",
            class_mapping=None,
            mono_date=None,
            sats=None,
            crop_size=64,
            s2_band=None,
            dict_classes=None,
            require_weights=False,
            path_subdataset=None,
            crop_type: Literal["Center", "Random"] = "Random",
            transform: nn.Module | None = None,
            # dataset_type: Literal["train", "val", "test"] = "test",
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
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
        path_subdataset (None or str). if not None indicates a path to a csv which contains a column id_patch. All the patches from get_item will be from this subset
        """

        super().__init__(crop_size, crop_type, transform, dataset_type)

        self.extract_true_doy = extract_true_doy
        self.allow_pad = allow_padd
        if sats is None:
            sats = ["S2"]
        if s2_band is None:
            self.s2_band = S2_BAND
        else:
            self.s2_band = s2_band

        self.folder = dataset_path_oe
        self.folder_orig = dataset_path_pastis
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.max_len = max_len
        self.target = target
        self.sats = sats
        print("before metadata")
        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(
            os.path.join(dataset_path_pastis, "metadata.geojson")
        )
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 5000))
        if dict_classes is None:
            dict_classes = LABEL_PASTIS
        self.dict_classes = dict_classes
        self.labels = list(self.dict_classes.values())

        print("Done.")

        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        #        self.len = self.meta_patch.shape[0]
        if path_subdataset is not None:
            df_selected_patch = pd.read_csv(path_subdataset)
            self.id_patches = df_selected_patch["id_patch"].astype("int")
        else:
            self.id_patches = self.meta_patch.index
        self.fold = folds
        self.len = len(self.id_patches)
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

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[
            np.where(self.date_tables[sat][id_patch] == 1)[0]
        ]

    def get_metadata(self) -> gpd.GeoDataFrame:
        """Return a GeoDatFrame with the patch metadata"""

    def read_data_from_disk(
        self,
        item: int,
        id_patch: int,
    ) -> tuple[dict[SatId, torch.Tensor], torch.Tensor]:
        """Get id_patch from disk and cache it as item in the dataset"""

    def __getitem__(self, item) -> OutClassifItem:
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data_pastis = {
                satellite:
                    torch.load(
                        os.path.join(
                            self.folder,
                            f"{satellite}_{id_patch}.pt",
                        )
                    )
                for satellite in self.sats
            }  # pastis_eo.dataclass.PASTISItem
            data: dict[str, MMDCDataStruct] = {}
            for satellite in self.sats:
                data_all = data_pastis[satellite]
                dem = torch.load(
                    os.path.join(
                        self.folder,
                        f"DEM_{id_patch}.pt",
                    )
                )
                data_one_sat = {"data": data_all.sits,
                                "mask": data_all.mask,
                                "angles": data_all.angles.nan_to_num(),
                                "dem": dem.nan_to_num()}

                METEO_BANDS = {
                    "dew_temp": "dewpoint-temperature",
                    "prec": "precipitation-flux",
                    "sol_rad": "solar-radiation-flux",
                    "temp_max": "temperature-max",
                    "temp_mean": "temperature-mean",
                    "temp_min": "temperature-min",
                    "vap_press": "vapour-pressure",
                    "wind_speed": "wind-speed",
                }

                data_one_sat.update({
                    f"meteo_{k}": torch.load(
                        os.path.join(
                            self.folder,
                            f"{satellite}_{k}_{id_patch}.pt",
                        )
                    )
                    for k in METEO_BANDS
                })

                data[satellite] = MMDCDataStruct.init_empty().fill_empty_from_dict(data_one_sat).concat_meteo()

            # Retrieve date sequences
            dates = {s: a.doy for s, a in data_pastis.items()}
            true_doys = {s: a.true_doy for s, a in data_pastis.items()}
            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder_orig,
                        "ANNOTATIONS",
                        f"TARGET_{id_patch}.npy",
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            else:
                raise NotImplementedError
            if self.cache:  # Not sure it works
                if self.mem16:
                    self.memory[item] = [
                        {k: v.casting(torch.float32) for k, v in data.items()},
                        target,
                        {k: d for k, d in dates.items()},
                        {k: d for k, d in true_doys.items()},
                    ]
                else:
                    self.memory[item] = [data, target, dates, true_doys]

        else:
            data, target, dates, true_doys = self.memory[item]
            if self.mem16:
                data = {k: v.casting(torch.float32) for k, v in data.items()}

        if len(self.sats) == 1:  # TODO change that for multimodal
            sits = data[self.sats[0]]
            doy = dates[self.sats[0]]
            if self.extract_true_doy:
                true_doy = true_doys[self.sats[0]]
            else:
                true_doy = None
            t, c, h, w = sits.data.data.shape
            y, x = self.get_crop_idx(
                rows=sits.data.data.shape[2], cols=sits.data.data.shape[3]
            )  # so that same crop for all the bands of a sits


            #TODO: crop
            # sits = sits[:, :, y: y + self.crop_size, x: x + self.crop_size]
            # target = target[y: y + self.crop_size, x: x + self.crop_size]
            target[target == 19] = 0  # merge background and void class
            # if self.transform is not None:
            #     my_logger.debug("Apply transform per SITS")
            #     sits = rearrange(sits, "t c h w -> c t h w")
            #     sits = self.transform(sits)
            #     sits = rearrange(sits, "c t h w -> t c h w")
            # TODO: padding
            sits, doy, padd_index = apply_padding(
                self.allow_pad, self.max_len, t, sits, doy
            )
            if true_doy is not None:
                padd_true_doy = F.pad(true_doy, (0, self.max_len - t))
            else:
                padd_true_doy = None
            return OutClassifItem(
                sits=sits,
                doy=doy.to(sits),
                padd_index=padd_index.bool(),
                target=target,
                padd_val=torch.tensor(self.max_len - t),
                item=item,
                mask_loss=(target != 0) & (target != 19),
                s2_path=os.path.join(
                    self.folder,
                    f"DATA_{self.sats[0]}",
                    f"{self.sats[0]}_{id_patch}.pt",
                ),
                true_doy=padd_true_doy,
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


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
                datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                - reference_date
        ).days
    )
    return d.values



@dataclass
class PastisDataSets:
    """Struct for the train, val and test datasets"""

    train: tdata.Dataset
    val: tdata.Dataset
    test: tdata.Dataset | None = None


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
        data_dir,
        batch_size,
        config_dataset,
        dict_classes=None,
        crop_size=64,
        num_workers: int = 1,
        path_dir_csv: str | None = None,
        s2_band: list | None = None,
        max_len: int = 60,
        data_folder: str,
        folds: PastisFolds,
        task: PASTISTask,
        batch_size: int = 2,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # init different variables
        self.data_folder = data_folder
        self.folds = folds
        self.batch_size = batch_size
        self.task = task

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
                PASTISDataset(
                    self.config_dataset.train,
                    dict_classes=self.dict_classes,
                    transform=None,
                    s2_band=self.s2_band,
                    dataset_path=path_dataset_final,
                    dataset_type="train",
                    require_weights=False,
                    crop_size=self.crop_size,
                    max_len=self.max_len,
),
                PASTISDataset(
                    PASTISOptions(task=self.task,
                                  folds=self.folds.val,
                                  folder=self.data_folder)),
            )

        if stage == "test":
            assert self.data
            self.data = PastisDataSets(
                self.data.train,
                self.data.val,
                PASTISDataset(
                    PASTISOptions(task=self.task,
                                  folds=self.folds.test,
                                  folder=self.data_folder)),
            )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader"""
        assert self.data is not None
        assert self.data.train is not None
        return self.instanciate_data_loader(self.data.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        assert self.data is not None
        assert self.data.val is not None
        return self.instanciate_data_loader(self.data.val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
        assert self.data is not None
        assert self.data.test is not None
        return self.instanciate_data_loader(self.data.test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Predict dataloader"""
        return self.test_dataloader()

    def instanciate_data_loader(
        self,
        dataset: tdata.Dataset,
        shuffle: bool = False,
    ) -> DataLoader:
        """Return a data loader with the PASTIS data set"""

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=lambda x: pad_collate(x, pad_value=2),
        )

class PASTISDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        config_dataset,
        dict_classes=None,
        crop_size=64,
        num_workers: int = 1,
        path_dir_csv: str | None = None,
        s2_band: list | None = None,
        max_len: int = 60,
    ):
        self.max_len = max_len
        if dict_classes is None:
            dict_classes = LABEL_PASTIS
        super().__init__(
            data_dir,
            batch_size,
            config_dataset,
            dict_classes,
            crop_size=crop_size,
            num_workers=num_workers,
            path_dir_csv=path_dir_csv,
            s2_band=s2_band,
            max_len=max_len,
        )

        self.dataset_path = os.path.join(
            self.data_dir, self.config_dataset.train.dataset_path
        )

        self.num_channels = PASTIS_BAND

    def setup(
        self, stage: str | None = None
    ) -> None:  # operations you might want to perform on every GPU.
        if stage == "fit" or stage is None:
            path_dataset_final = os.path.join(
                self.data_dir, self.config_dataset.train.dataset_path
            )
            my_logger.info("load train ... ")

            self.data_train: PASTIS_Dataset = instantiate(
                self.config_dataset.train,
                dict_classes=self.dict_classes,
                transform=None,
                s2_band=self.s2_band,
                dataset_path=path_dataset_final,
                # dataset_type="train",
                require_weights=False,
                crop_size=self.crop_size,
                max_len=self.max_len,
            )
            my_logger.info("train loaded")
            self.data_val: PASTIS_Dataset = instantiate(
                self.config_dataset.val,
                dict_classes=self.dict_classes,
                transform=None,
                s2_band=self.s2_band,
                dataset_path=os.path.join(
                    self.data_dir, self.config_dataset.val.dataset_path
                ),
                # dataset_type="val",
                require_weights=False,
                crop_size=self.crop_size,
                max_len=self.max_len,
            )  # maybe unecessary
            my_logger.info("val loaded")
        if stage == "test":
            self.data_test: PASTIS_Dataset = instantiate(
                self.config_dataset.test,
                dict_classes=self.dict_classes,
                transform=None,
                s2_band=self.s2_band,
                dataset_path=os.path.join(
                    self.data_dir, self.config_dataset.test.dataset_path
                ),
                # dataset_type="test",
                require_weights=False,
                crop_size=self.crop_size,
                max_len=self.max_len,
            )
            my_logger.info("test loaded")
