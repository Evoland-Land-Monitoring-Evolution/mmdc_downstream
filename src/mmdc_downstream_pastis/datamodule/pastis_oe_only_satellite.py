"""
PASTIS-R Dataset module
https://raw.githubusercontent.com/VSainteuf/pastis-benchmark/main/code/dataloader.py
Original Author: Vivien Sainte Fare Garnot (github.com/VSainteuf)
Original License MIT
"""

import collections.abc
import json
import logging
import os
import re
from collections.abc import Callable, Sized
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Literal, TypeAlias, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mmdc_downstream_pastis.datamodule.datatypes import BatchInputUTAE

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def pad_tensor(x_t: torch.Tensor, final_size: int, pad_value: int = 0) -> torch.Tensor:
    """
    Pad the first dimension of a tensor with to have final_size,
    If final size is zero or negative, no padding is done
    """
    if final_size <= 0:
        return x_t
    padlen = final_size - x_t.shape[0]
    padding = [0 for _ in range(2 * len(x_t.shape[1:]))] + [0, padlen]
    return F.pad(x_t, pad=padding, value=pad_value)


np_str_obj_array_pattern = re.compile(r"[SaUO]")

PaddedBatch: TypeAlias = (
    torch.Tensor
    | tuple[np.ndarray, np.memmap]
    | collections.abc.Mapping
    | collections.abc.Sequence
)

UnpaddedBatch = (
    list[torch.Tensor]
    | list[tuple[np.ndarray, np.memmap]]
    | list[collections.abc.Mapping]
    | list[collections.abc.Sequence]
    | tuple[torch.Tensor, ...]
    | tuple[tuple[np.ndarray, np.memmap]]
    | tuple[collections.abc.Mapping, ...]
    | tuple[collections.abc.Sequence, ...]
)


def pad_collate_dataclass(
    batch: UnpaddedBatch,
    pad_value: int = 0,
) -> BatchInputUTAE:
    """
    Embed collated data into a dataclass
    """
    (data, dates), target = pad_collate(batch, pad_value)

    return BatchInputUTAE(
        sits=data, doy=dates, gt=target, gt_mask=(target == 0) | (target == 19)
    )


def pad_collate(
    batch: UnpaddedBatch,
    pad_value: int = 0,
) -> PaddedBatch:
    """Utility function to be used as collate_fn for the PyTorch dataloader
    to handle sequences of varying length.
    Sequences are padded with zeros by default.

    Modified default_collate from the official pytorch repo
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch if isinstance(e, torch.Tensor)]
            max_size = max(sizes)
            if not all(s == max_size for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [
                    pad_tensor(e, max_size, pad_value=pad_value)
                    for e in batch
                    if isinstance(e, torch.Tensor)
                ]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch if isinstance(x, torch.Tensor))
            storage = elem.storage()._new_shared(numel)  # pylint: disable=W0212
            out = elem.new(storage)
        return torch.stack([*batch], 0, out=out)
    if isinstance(elem, (np.ndarray, np.memmap)):
        # array of string classes and object
        if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
            raise TypeError(f"Format not managed : {elem.dtype}")
        return pad_collate([torch.as_tensor(b) for b in batch])
    if isinstance(elem, collections.abc.Mapping):
        return {
            key: pad_collate(
                [d[key] for d in batch if isinstance(d, collections.abc.Mapping)]
            )
            for key in elem
        }
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return [*(pad_collate(samples) for samples in zip(*batch))]
    if isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        batch_it = iter(batch)
        elem_size = len(batch[0])
        if not all(
            len(elem) == elem_size for elem in batch_it if isinstance(elem, Sized)
        ):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError(f"Format not managed : {elem_type}")


SatId: TypeAlias = Literal["S2", "S1_ASC", "S1_DESC"]

PASTIS_BANDS: dict[SatId, int] = {"S2": 10, "S1_ASC": 3, "S1_DESC": 3}


@dataclass
class PASTISDates:
    """Date table and date range for the PASTIS dataset"""

    tables: dict[str, dict[int, np.ndarray] | None]
    range_: np.ndarray


ClassMapping: TypeAlias = dict[int, int] | Callable[[np.ndarray], int]


class PASTISTarget(Enum):
    """The target task for the PASTIS data set"""

    SEMANTIC = auto()
    INSTANCE = auto()


@dataclass
class PASTISTask:
    """
    target (PASTISTarget): 'SEMANTIC' or 'INSTANCE'. Defines which type of target is
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
    reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
        based on which all observation dates are expressed. Along with the image
        time series and the target tensor, this dataloader yields the sequence
        of observation dates (in terms of number of days since the reference
        date). This sequence of dates is used for instance for the positional
        encoding in attention based approaches.
    class_mapping (dict, optional): Dictionary to define a mapping between the
        default 18 class nomenclature and another class grouping, optional.
    mono_date (int or str, optional): If provided only one date of the
        available time series is loaded. If argument is an int it defines the
        position of the date that is loaded. If it is a string, it should be
        in format 'YYYY-MM-DD' and the closest available date will be selected.
    sats (list): defines the satellites to use. If you are using PASTIS-R, you
         have access to
        Sentinel-2 imagery and Sentinel-1 observations in Ascending and
        Descending orbits,
        respectively S2, S1A, and S1D.
        For example use sats=['S2', 'S1A'] for Sentinel-2 + Sentinel-1 ascending
        time series,
        or sats=['S2', 'S1A','S1D'] to retrieve all time series.
        If you are using PASTIS, only  S2 observations are available."""

    target: PASTISTarget | str = PASTISTarget.SEMANTIC
    reference_date: str | datetime = "2018-09-01"
    class_mapping: ClassMapping | None = None
    mono_date: int | str | datetime | None = None
    sats: list[SatId] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.reference_date, str):
            year, month, day = map(int, self.reference_date.split("-"))
            self.reference_date = datetime(year, month, day)
        if isinstance(self.mono_date, str):
            if "-" in self.mono_date:
                year, month, day = map(int, self.mono_date.split("-"))
                self.mono_date = datetime(year, month, day)
            else:
                int(self.mono_date)

        if isinstance(self.class_mapping, dict):
            self.class_mapping = cast(
                Callable[[np.ndarray], int],
                np.vectorize(lambda x: self.class_mapping[x]),  # type: ignore
            )

        if self.sats is None:
            self.sats = ["S2"]

        if isinstance(self.target, str):
            if self.target.lower() == "semantic":
                self.target = PASTISTarget.SEMANTIC
            elif self.target.lower() == "instance":
                self.target = PASTISTarget.INSTANCE
            else:
                self.target = PASTISTarget.SEMANTIC


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

    task: PASTISTask
    folder: str
    folder_oe: str
    folds: list[int] | None = None
    norm: bool = True
    cache: bool = False
    mem16: bool = False


@dataclass
class PastisFolds:
    """Pastis folds for different tasks"""

    train: list[int]
    val: list[int]
    test: list[int] | None = None


class PASTISDataset(tdata.Dataset):
    """Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
       panoptic segmentation.

    The Dataset yields ((data, dates), target) tuples, where:
        - data contains the image time series
        - dates contains the date sequence of the observations expressed in number
          of days since a reference date
        - target is the semantic or instance target"""

    def __init__(self, options: PASTISOptions):
        """
        Initialise the dataset with the config, then
        - get metadata from the geojson file in the dataset folder
        - build dates table
        - build folds
        - get normalisation values and build the corresponding numpy arrays
        """
        super().__init__()
        self.options = options
        self.memory: dict[int, Any] = {}
        # PatchId: TypeAlias = str
        # self.memory_dates: dict[PatchId, dict[SatId, torch.Tensor]] = {}

        # Get metadata
        self.strict_s1 = True
        meta_patch = self.get_metadata()

        # Select Fold samples
        if self.options.folds is not None:
            meta_patch = pd.concat(
                [meta_patch[meta_patch["Fold"] == f] for f in self.options.folds]
            )

        self.len: int = meta_patch.shape[0]
        self.id_patches = meta_patch.index

        # Get normalisation values
        assert self.options.task.sats is not None
        self.norm: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
        if self.options.norm:
            self.norm = {}
            for sat in self.options.task.sats:
                if sat != "S2_MSK":
                    if sat == "S2":
                        s_name = "S2"
                    elif sat == "S1_ASC":
                        s_name = "S1A"
                    elif sat == "S1_DESC":
                        s_name = "S1D"

                    with open(
                        os.path.join(self.options.folder, f"NORM_{s_name}_patch.json"),
                        encoding="utf-8",
                    ) as file:
                        normvals = json.loads(file.read())
                    selected_folds = (
                        self.options.folds
                        if self.options.folds is not None
                        else range(1, 6)
                    )
                    means = [normvals[f"Fold_{f}"]["mean"] for f in selected_folds]
                    stds = [normvals[f"Fold_{f}"]["std"] for f in selected_folds]
                    tmp_norm = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                    self.norm[sat] = (
                        torch.from_numpy(tmp_norm[0]).float(),
                        torch.from_numpy(tmp_norm[1]).float(),
                    )
        logger.info("Dataset ready.")

    def get_one_satellite_patches(self, satellite):
        path = self.options.folder_oe
        sat_patches = [
            int(file[:-3].split("_")[-1])
            for file in os.listdir(os.path.join(path, satellite))
        ]

        meteo_patches = [
            int(file[:-3].split("_")[-1])
            for file in os.listdir(os.path.join(path, f"{satellite}_METEO"))
        ]

        dem_patches = [
            int(file[:-3].split("_")[-1])
            for file in os.listdir(os.path.join(path, "DEM"))
        ]

        return np.intersect1d(
            np.intersect1d(sat_patches, meteo_patches, assume_unique=False),
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
        for satellite in self.options.task.sats:
            valid = self.get_one_satellite_patches(satellite)
            if len(self.options.task.sats) == 1:
                return valid

        if len(self.options.task.sats) == 1:
            return self.get_one_satellite_patches(self.options.task.sats[0])

        elif len(self.options.task.sats) == 2:
            for satellite in self.options.task.sats:
                valid = self.get_one_satellite_patches(satellite)
                valid_patches.append(valid)

            if (
                (np.sort(self.options.task.sats) == ["S1_ASC", "S2"]).all()
                or (np.sort(self.options.task.sats) == ["S1_DESC", "S2"]).all()
                or self.strict_s1
            ):
                return np.intersect1d(
                    valid_patches[0], valid_patches[1], assume_unique=True
                )
            else:  # ["S1_ASC", "S1_DESC"]
                return np.union1d(valid_patches[0], valid_patches[1])

        else:
            for satellite in np.sort(self.options.task.sats):
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
            os.path.join(self.options.folder, "metadata.geojson")
        )
        meta_patch.index = meta_patch["ID_PATCH"].astype(int)
        meta_patch.sort_index(inplace=True)

        assert self.options.task.sats is not None
        self.dates = PASTISDates(
            {s: None for s in self.options.task.sats}, np.array(range(-200, 600))
        )
        for sat in self.options.task.sats:
            s_name = sat
            if sat == "S2_MSK":
                s_name = "S2"
            elif sat == "S1_ASC":
                s_name = "S1A"
            elif sat == "S1_DESC":
                s_name = "S1D"
            dates = meta_patch[f"dates-{s_name}"]
            meta_patch = meta_patch[meta_patch.ID_PATCH.isin(self.patches_to_keep())]

            date_table = pd.DataFrame(
                index=meta_patch.index, columns=self.dates.range_, dtype=int
            )
            assert isinstance(self.options.task.reference_date, datetime)
            for pid, date_seq in dates.items():
                dates = pd.DataFrame().from_dict(date_seq, orient="index")
                dates = dates[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.options.task.reference_date
                    ).days
                )
                date_table.loc[pid, dates.values] = 1
            date_table = date_table.fillna(0)
            self.dates.tables[sat] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        logger.info("Done.")
        return meta_patch

    def __len__(self) -> int:
        return self.len

    def get_dates(self, id_patch: int | np.integer, sat: str) -> np.ndarray:
        """Return the dates for the selected patch and satellite"""
        sat_table = cast(np.ndarray, self.dates.tables[sat])
        return self.dates.range_[np.where(sat_table[id_patch] == 1)[0]]

    def read_data_from_disk(
        self,
        item: int,
        id_patch: int,
    ) -> tuple[dict[SatId, torch.Tensor], dict[SatId, np.ndarray], torch.Tensor]:
        """Get id_patch from disk and cache it as item in the dataset"""
        assert self.options.task.sats is not None
        data_all = {
            satellite: torch.load(
                os.path.join(
                    self.options.folder_oe,
                    satellite,
                    f"{satellite}_{id_patch}.pt",
                )
            )
            for satellite in self.options.task.sats
        }  # T x C x H x W arrays
        data = {s: a.sits.to(torch.float32) for s, a in data_all.items()}
        dates = {s: a.doy for s, a in data_all.items()}

        if self.norm is not None:
            data.update(
                {
                    s: (
                        (d - self.norm[s][0][None, :, None, None])
                        / self.norm[s][1][None, :, None, None]
                    ).nan_to_num()
                    for s, d in data.items()
                    if s in self.norm
                }
            )

        if self.options.task.target == PASTISTarget.SEMANTIC:
            target = np.load(
                os.path.join(
                    self.options.folder, "ANNOTATIONS", f"TARGET_{id_patch}.npy"
                )
            )
            target = torch.from_numpy(target[0].astype(int))

            if self.options.task.class_mapping is not None:
                target = cast(
                    Callable[[np.ndarray], int], self.options.task.class_mapping
                )(target)

        elif self.options.task.target == PASTISTarget.INSTANCE:
            heatmap = np.load(
                os.path.join(
                    self.options.folder,
                    "INSTANCE_ANNOTATIONS",
                    f"HEATMAP_{id_patch}.npy",
                )
            )

            instance_ids = np.load(
                os.path.join(
                    self.options.folder,
                    "INSTANCE_ANNOTATIONS",
                    f"INSTANCES_{id_patch}.npy",
                )
            )
            pixel_to_object_mapping = np.load(
                os.path.join(
                    self.options.folder,
                    "INSTANCE_ANNOTATIONS",
                    f"ZONES_{id_patch}.npy",
                )
            )

            pixel_semantic_annotation = np.load(
                os.path.join(
                    self.options.folder, "ANNOTATIONS", f"TARGET_{id_patch}.npy"
                )
            )

            if self.options.task.class_mapping is not None:
                pixel_semantic_annotation = cast(
                    Callable[[np.ndarray], int], self.options.task.class_mapping
                )(pixel_semantic_annotation[0])
            else:
                pixel_semantic_annotation = pixel_semantic_annotation[0]

            size = np.zeros((*instance_ids.shape, 2))
            object_semantic_annotation = np.zeros(instance_ids.shape)
            for instance_id in np.unique(instance_ids):
                if instance_id != 0:
                    height = (instance_ids == instance_id).any(axis=-1).sum()
                    width = (instance_ids == instance_id).any(axis=-2).sum()
                    size[pixel_to_object_mapping == instance_id] = (height, width)
                    object_semantic_annotation[
                        pixel_to_object_mapping == instance_id
                    ] = pixel_semantic_annotation[instance_ids == instance_id][0]

            target = torch.from_numpy(
                np.concatenate(
                    [
                        heatmap[:, :, None],  # 0
                        instance_ids[:, :, None],  # 1
                        pixel_to_object_mapping[:, :, None],  # 2
                        size,  # 3-4
                        object_semantic_annotation[:, :, None],  # 5
                        pixel_semantic_annotation[:, :, None],  # 6
                    ],
                    axis=-1,
                )
            ).float()

        if self.options.cache:
            if self.options.mem16:
                self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
            else:
                self.memory[item] = [data, dates, target]

        return data, dates, target

    def __getitem__(
        self, item: int
    ) -> (
        tuple[tuple[dict[SatId, torch.Tensor], dict[SatId, torch.Tensor]], torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        assert self.options.task.sats is not None
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.options.cache or item not in self.memory:
            data, dates, target = self.read_data_from_disk(item, id_patch)
        else:
            data, dates, target = self.memory[item]
            if self.options.mem16:
                data = {k: v.float() for k, v in data.items()}

        # # Retrieve date sequences
        # if not self.options.cache or id_patch not in self.memory_dates:
        #     if self.options.cache:
        #         self.memory_dates[id_patch] = dates
        # else:
        #     dates = self.memory_dates[id_patch]

        if self.options.task.mono_date is not None:
            if isinstance(self.options.task.mono_date, int):
                data = {
                    s: data[s][self.options.task.mono_date].unsqueeze(0)
                    for s in self.options.task.sats
                }
                dates = {
                    s: dates[s][self.options.task.mono_date]
                    for s in self.options.task.sats
                }
            elif not isinstance(self.options.task.mono_date, str) and not isinstance(
                self.options.task.reference_date, str
            ):
                mono_delta = (
                    self.options.task.mono_date - self.options.task.reference_date
                ).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin())
                    for s in self.options.task.sats
                }
                data = {
                    s: data[s][mono_date[s]].unsqueeze(0)
                    for s in self.options.task.sats
                }
                dates = {s: dates[s][mono_date[s]] for s in self.options.task.sats}

        if self.options.mem16:
            return ({k: v.float() for k, v in data.items()}, dates), target

        # TODO change for multimodal
        if len(self.options.task.sats) == 1:
            data = data[self.options.task.sats[0]]
            dates = dates[self.options.task.sats[0]]

        return (data, dates), target


def prepare_dates(date_dict: dict[str, str], reference_date: datetime) -> np.ndarray:
    """Date formating."""
    dates = pd.DataFrame().from_dict(date_dict, orient="index")
    dates = dates[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return cast(np.ndarray, dates.values)


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
        dataset_path_pastis: str,
        dataset_path_oe: str,
        folds: PastisFolds,
        task: PASTISTask,
        batch_size: int = 2,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # init different variables
        self.dataset_path_pastis = dataset_path_pastis
        self.dataset_path_oe = dataset_path_oe
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
                    PASTISOptions(
                        task=self.task,
                        folds=self.folds.train,
                        folder=self.dataset_path_pastis,
                        folder_oe=self.dataset_path_oe,
                    )
                ),
                PASTISDataset(
                    PASTISOptions(
                        task=self.task,
                        folds=self.folds.val,
                        folder=self.dataset_path_pastis,
                        folder_oe=self.dataset_path_oe,
                    )
                ),
            )

        if stage == "test":
            assert self.data
            self.data = PastisDataSets(
                self.data.train,
                self.data.val,
                PASTISDataset(
                    PASTISOptions(
                        task=self.task,
                        folds=self.folds.test,
                        folder=self.dataset_path_pastis,
                        folder_oe=self.dataset_path_oe,
                    )
                ),
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
            collate_fn=lambda x: pad_collate_dataclass(x, pad_value=0),
        )


# def build_dm(sats: list[SatId]) -> PastisDataModule:
#     """Builds datamodule"""
#     return PastisDataModule(
#         dataset_path_pastis=DATA_FOLDER,
#         dataset_path_oe=DATA_FOLDER_OE,
#         folds=PastisFolds([1, 2, 3], [4], [5]),
#         task=PASTISTask(sats=["S1_ASC", "S1_DESC", "S2"]),
#         batch_size=2,
#     )
#
#
# DATA_FOLDER = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"
# DATA_FOLDER_OE = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis_OE"
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
#         for batch, _ in zip(loader, range(4)):
#             print(batch.sits.keys())
#
#
# pastisds_dataloader(["S1_ASC", "S1_DESC"])
