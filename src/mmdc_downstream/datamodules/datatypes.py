""" Types and classes for the MMDC data modules """
import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class MMDCDataChannels:
    """Number of channels for each MMDC data source"""

    sen1: int = 6
    s1_angles: int = 4
    sen2: int = 10
    s2_angles: int = 6
    dem: int = 4
    meteo: int = 8


@dataclass
class ShiftScale:
    """Data for linear scaling"""

    shift: torch.Tensor
    scale: torch.Tensor


@dataclass
class MMDCShiftScales:
    """Scaling info for MMDC data"""

    sen2: ShiftScale
    sen1: ShiftScale
    meteo: ShiftScale
    dem: ShiftScale


@dataclass
class MMDCS2Data:
    """Struct for Sentinel-2 data, masks and angles"""

    s2_set: torch.Tensor
    s2_masks: torch.Tensor
    s2_angles: torch.Tensor


@dataclass
class MMDCS1Data:
    """Struct for Sentinel-1 data, masks and angles"""

    s1_set: torch.Tensor
    s1_valmasks: torch.Tensor
    s1_angles: torch.Tensor


@dataclass
class MMDCMeteoData:
    # pylint: disable=too-many-instance-attributes
    """Struct for Sentinel-2 data, masks and angles"""

    dew_temp: torch.Tensor
    prec: torch.Tensor
    sol_rad: torch.Tensor
    temp_max: torch.Tensor
    temp_mean: torch.Tensor
    temp_min: torch.Tensor
    vap_press: torch.Tensor
    wind_speed: torch.Tensor

    def concat_data(self):
        """Concatenate meteo data into 8-features image series"""
        return torch.cat(
            [getattr(self, field.name).unsqueeze(-3) for field in fields(self)], -3
        )


@dataclass
class MMDCDataStruct:
    """Dataclass holding the tensors for image and auxiliary data in MMDC"""

    s2_data: MMDCS2Data
    s1_data: MMDCS1Data
    meteo: MMDCMeteoData
    dem: torch.Tensor

    def __getitem__(self, item):
        copied = self.init_empty()
        for field in fields(self):
            if field.name == "dem":
                setattr(copied, field.name, getattr(self, field.name)[item])
            else:
                for field_ in fields(getattr(self, field.name)):
                    setattr(
                        getattr(copied, field.name),
                        field_.name,
                        getattr(getattr(self, field.name), field_.name)[item],
                    )
        return copied

    def filter_data(self, idx):
        """Filter MMDC data by index, in place"""
        for field in fields(self):
            if isinstance(getattr(self, field.name), torch.Tensor):
                setattr(self, field.name, getattr(self, field.name)[idx])
            else:
                for field_ in fields(getattr(self, field.name)):
                    setattr(
                        getattr(self, field.name),
                        field_.name,
                        getattr(getattr(self, field.name), field_.name)[idx],
                    )

    def fill_empty_from_dict(self, dictionary: dict):
        """Fill an empty dataclass instance from dictionary"""
        for key, value in dictionary.items():
            key = key[6:] if key.startswith("meteo") else key
            if key in self.meteo.__dict__:
                setattr(self.meteo, key, value)
            elif key in self.s1_data.__dict__:
                setattr(self.s1_data, key, value)
            elif key in self.s2_data.__dict__:
                setattr(self.s2_data, key, value)
            elif key == "dem":
                setattr(self, key, value)
        return self

    @staticmethod
    def init_empty():
        """Create an empty dataclass instance"""
        return MMDCDataStruct(
            MMDCS2Data(None, None, None),
            MMDCS1Data(None, None, None),
            MMDCMeteoData(None, None, None, None, None, None, None, None),
            None,
        )


@dataclass
class MMDCDataSets:
    """Struct for the train, val and test datasets"""

    train: Dataset
    val: Dataset
    test: Dataset | None = None


@dataclass
class MMDCTensorStats:
    """Stats for a data tensor"""

    mean: torch.Tensor
    std: torch.Tensor
    qmin: torch.Tensor
    median: torch.Tensor
    qmax: torch.Tensor


@dataclass
class MMDCMeteoStats:
    # pylint: disable=too-many-instance-attributes
    """MMDC Stats for meteo data"""

    dew_temp: MMDCTensorStats
    prec: MMDCTensorStats
    sol_rad: MMDCTensorStats
    temp_max: MMDCTensorStats
    temp_mean: MMDCTensorStats
    temp_min: MMDCTensorStats
    vap_press: MMDCTensorStats
    wind_speed: MMDCTensorStats

    def concat_stats(self):
        """Concatenate meteo stats"""
        return MMDCTensorStats(
            *[
                torch.cat(
                    [getattr((getattr(self, field.name)), k) for field in fields(self)],
                    1,
                )
                for k in MMDCTensorStats.__dict__["__dataclass_fields__"].keys()
            ]
        )


@dataclass
class MMDCDataStats:
    """Struct for the MMDCTensorStats for the whole data set"""

    sen2: MMDCTensorStats
    sen1: MMDCTensorStats
    meteo: MMDCMeteoStats
    dem: MMDCTensorStats


@dataclass
class MMDCDataLoaderConfig:
    """Configuration parameters for the data loaders"""

    max_open_files: int
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class MMDCDataStatsConfig:
    """Struct for stats computation configuration"""

    batch_size: int
    max_open_files: int


@dataclass
class MMDCDataPaths:
    """Struct for the paths to tensors and rois"""

    tensors_dir: Path
    train_rois: Path
    val_rois: Path
    test_rois: Path

    def __post_init__(self) -> None:
        """Ensure that fields are of the correct type in case the
        client passed str instead of Path"""
        for field in fields(self):
            setattr(self, field.name, field.type(getattr(self, field.name)))


@dataclass
class MMDCDataModuleFiles:
    """Several file lists used by the data module"""

    stats_files: list[list[str]]
    train_data_files: list[list[Iterator[tuple]]]
    val_data_files: list[list[Iterator[tuple]]]
    test_data_files: list[list[Iterator[tuple]]]


@dataclass
class MMDCBatch:
    # pylint: disable=too-many-instance-attributes
    """Lightning module batch"""

    s2_x: torch.Tensor
    s2_m: torch.Tensor
    s2_a: torch.Tensor
    s1_x: torch.Tensor
    s1_vm: torch.Tensor
    s1_a: torch.Tensor
    meteo_x: torch.Tensor
    dem_x: torch.Tensor

    def copy(self):
        """Makes a MMDCBatch copy"""
        return dataclasses.replace(self)
