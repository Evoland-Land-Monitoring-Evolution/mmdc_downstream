"""Datatypes"""

import random
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
import torch
import torch.utils.data as tdata
from einops import rearrange
from mmdc_singledate.datamodules.datatypes import MMDCMeteoData
from torch.nn import functional as F


@dataclass
class PastisFolds:
    """Pastis folds for different tasks"""

    train: list[int]
    val: list[int]
    test: list[int] | None = None


@dataclass
class PastisDataSets:
    """Struct for the train, val and test datasets"""

    train: tdata.Dataset
    val: tdata.Dataset
    test: tdata.Dataset | None = None


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

    dataset_path_oe: str
    dataset_path_pastis: str
    folds: list[int] | None = None
    norm: bool = True
    cache: bool = False
    mem16: bool = False
    task: Literal["semantic"] = "semantic"


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


@dataclass
class MMDCData:
    """Dataclass holding the tensors for image and auxiliary data in MMDC"""

    img: torch.Tensor | None
    mask: torch.Tensor | None
    angles: torch.Tensor | None


@dataclass
class MMDCDataStruct:
    """Dataclass holding the tensors for image and auxiliary data in MMDC"""

    data: MMDCData
    meteo: MMDCMeteoData | torch.Tensor
    dem: torch.Tensor

    def fill_empty_from_dict(self, dictionary: dict, flatten_meteo: bool = False):
        """Fill an empty dataclass instance from dictionary"""
        for key, value in dictionary.items():
            key = key[6:] if (key.startswith("meteo") and len(key) > 5) else key
            if type(self.meteo) is MMDCMeteoData and key in self.meteo.__dict__:
                setattr(self.meteo, key, value)
            elif key in self.data.__dict__:
                setattr(self.data, key, value)
            else:
                if key == "meteo" and flatten_meteo:
                    setattr(self, key, rearrange(value, "b t d c h w -> b t (d c) h w"))
                else:
                    setattr(self, key, value)
        return self

    @staticmethod
    def init_empty():
        """Create an empty dataclass instance"""
        return MMDCDataStruct(
            MMDCData(None, None, None),
            MMDCMeteoData(None, None, None, None, None, None, None, None),
            None,
        )

    @staticmethod
    def init_empty_concat_meteo():
        """
        Create an empty dataclass instance,
        with concatenated meteo data
        """
        return MMDCDataStruct(
            MMDCData(None, None, None),
            None,
            None,
        )

    @staticmethod
    def init_empty_zeros_s1(t_max: int, H: int, W: int):
        """
        Create an empty dataclass instance,
        with concatenated meteo data
        """
        return MMDCDataStruct(
            MMDCData(
                torch.zeros(t_max, 3, H, W),
                torch.ones(t_max, 1, H, W),
                torch.zeros(t_max, 1, H, W),
            ),
            torch.zeros(t_max, 6, 8, H, W),
            torch.zeros(4, H, W),
        )

    def concat_meteo(self):
        """Concatenate meteo data into 8-features image series"""
        setattr(
            self,
            "meteo",
            torch.cat(
                [
                    getattr(self.meteo, field.name).unsqueeze(-3)
                    for field in fields(self.meteo)
                ],
                -3,
            ),
        )
        return self

    def casting(self, dtype: torch.dtype):
        """Convert all values to defined datatype, except masks that remain int"""
        setattr(self, "meteo", getattr(self, "meteo").to(dtype))
        setattr(self, "dem", getattr(self, "dem").to(dtype))
        setattr(self.data, "data", getattr(self.data, "data").to(dtype))
        setattr(self.data, "angles", getattr(self.data, "angles").to(dtype))
        setattr(self.data, "masks", getattr(self.data, "masks").to(torch.int8))

    def crop(self, x, y, cs):
        """Crop satellite image, angles, mask, dem and corresponding meteo data"""
        setattr(self, "meteo", getattr(self, "meteo")[:, :, :, y : y + cs, x : x + cs])
        setattr(self, "dem", getattr(self, "dem")[:, y : y + cs, x : x + cs])
        for field in fields(self.data):
            setattr(
                self.data,
                field.name,
                getattr(self.data, field.name)[:, :, y : y + cs, x : x + cs],
            )
        return self

    def padding(self, padd_tensor):
        """Pad satellite image, angles, mask and corresponding meteo data"""
        setattr(self, "meteo", F.pad(getattr(self, "meteo"), (0, 0) + padd_tensor))
        for field in fields(self.data):
            setattr(
                self.data,
                field.name,
                F.pad(getattr(self.data, field.name), padd_tensor),
            )
        return self


def randomcropindex(
    img_h: int, img_w: int, cropped_h: int, cropped_w: int
) -> tuple[int, int]:
    """
    Generate random numbers for window cropping of the patch (used in rasterio window)
    Args:
        img_h ():
        img_w ():
        cropped_h ():
        cropped_w ():

    Returns:

    """
    assert img_h >= cropped_h
    assert img_w >= cropped_w
    height = random.randint(0, img_h - cropped_h)
    width = random.randint(0, img_w - cropped_w)
    return height, width


@dataclass
class OneSatellitePatch:
    """One modality"""

    sits: MMDCDataStruct
    true_doy: np.ndarray | None


@dataclass
class PastisBatch:
    """Pastis input batch"""

    sits: MMDCDataStruct
    true_doy: torch.Tensor | np.ndarray = None

    def w(self):
        """width"""
        return self.sits.data.img.shape[-1]

    def h(self):
        """height"""
        return self.sits.data.img.shape[-2]

    @staticmethod
    def init_empty():
        """Create an empty dataclass instance"""
        return PastisBatch(None, None)

    def fill_empty_from_dict(self, dictionary: dict):
        """Fill an empty dataclass instance from dictionary"""
        for key, value in dictionary.items():
            setattr(self, key, value)
        return self

    def concat_aux_data(self, pad_value: int = 0) -> torch.Tensor:
        """Legacy function. Concatenate all aux data to images"""
        b, t, _, h, w = self.sits.data.img.shape
        dem_repeat = (
            self.sits.dem.unsqueeze(1)
            .expand(b, t, self.sits.dem.shape[-3], h, w)
            .clone()
        )
        pad_index = self.true_doy == pad_value
        dem_repeat[pad_index] = pad_value

        return torch.concat(
            [self.sits.data.img, self.sits.data.angles, self.sits.meteo, dem_repeat],
            dim=-3,
        )


@dataclass
class BatchInputUTAE:
    """Input to UTAE algorithm"""

    sits: torch.Tensor | dict[str, torch.Tensor] | PastisBatch
    gt: torch.Tensor
    doy: torch.Tensor | dict[str, torch.Tensor] | None = None
    sits_mask: torch.Tensor | dict[str, torch.Tensor] | None = None
    gt_mask: torch.Tensor | None = None
    id_patch: torch.Tensor | None = None
