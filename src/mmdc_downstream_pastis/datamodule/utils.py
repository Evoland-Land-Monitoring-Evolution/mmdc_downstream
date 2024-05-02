import logging.config
import random
from dataclasses import dataclass, fields

import numpy as np
import torch
from mmdc_singledate.datamodules.datatypes import MMDCMeteoData
from torch import Tensor
from torch.nn import functional as F

from .dataclass import InputInterp

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)
my_logger = logging.getLogger(__name__)


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
                    setattr(self, key, torch.flatten(value, start_dim=2, end_dim=3))
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
    def init_empty_zeros_s1(t_max, H, W):
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
class OutputMaskTensor:
    cld_mask: Tensor | None = None
    valid_mask: Tensor | None = None
    nan_mask: Tensor | None = None


@dataclass
class OutputSplitTensorin2:
    array_before: Tensor
    array_after: Tensor
    time_before: Tensor
    time_after: Tensor
    padd_index: Tensor
    mask_before: Tensor
    mask_after: Tensor


def split_before_after_tensor(tensor: Tensor, middle: int):
    array_before = torch.unsqueeze(tensor[:, :middle, ...], dim=1)  # c f t h w
    array_after = torch.unsqueeze(tensor[:, middle:, ...], dim=1)
    return array_before, array_after


def temp_padd_split(
    array: Tensor, time: Tensor, len_sits: int
) -> tuple[Tensor, Tensor, Tensor]:
    """

    Args:
        array ():shape (c,f,t,h,w)
        time ():(f,t)
        len_sits ():(len_sits)

    Returns:

    """
    my_logger.debug(array.shape)
    c, f, t, h, w = array.shape
    padd_index = torch.zeros(len_sits)
    if len_sits - t != 0:
        padd_index[-len_sits + t :] = 1
    if -len_sits + t == 0:
        return array, time, padd_index

    return (
        padd_sits_split(array, len_sits=len_sits, t=t),
        padd_time_split(time, len_sits=len_sits, t=t),
        padd_index.bool(),
    )


def padd_time_split(time: Tensor, len_sits: int, t: int):
    # print("in shape",time.shape)
    if len_sits - t != 0:
        padd_tensor = (0, len_sits - t, 0, 0)
        time = F.pad(time, padd_tensor)
    return time


def padd_sits_split(array: Tensor, len_sits: int, t: int):
    my_logger.debug(f"arr {array.shape} padded require {len_sits - t}")
    if len_sits - t != 0:
        if len(array.shape) == 5:
            padd_tensor = (0, 0, 0, 0, 0, len_sits - t, 0, 0, 0, 0)
        elif len(array.shape) == 4:  # c t h w
            padd_tensor = (0, 0, 0, 0, 0, len_sits - t, 0, 0)
        else:
            raise NotImplementedError
        return F.pad(array, padd_tensor)
    return array


def time_delta(date: np.ndarray, reference_date: np.datetime64 | None = None):
    if reference_date is None:
        reference_date = np.datetime64("2014-03-03", "D")
    duration = date - reference_date
    return duration


def temp_padd(
    array: Tensor, time: Tensor, len_sits: int
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        array (): shape c,t,h,w
        time (): t
        len_sits (): output temporal dimension
    Returns: tuple of tensors of dimension [c,len_sits,h,w], [len_sits],[len_sits]
    """
    assert len(array.shape) == 4, f"Wrong input dim {array.shape}"
    c, t, h, w = array.shape
    padd_tensor = (0, 0, 0, 0, 0, len_sits - t, 0, 0)
    padd_t = (0, len_sits - t)
    padd_index = torch.zeros(len_sits)
    if len_sits - t != 0:
        padd_index[-len_sits + t :] = 1
    return F.pad(array, padd_tensor), F.pad(time, padd_t), padd_index.bool()


def apply_padding(allow_padd, max_len, t, sits, doy):
    if allow_padd:
        padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
        padd_doy = (0, max_len - t)
        # padd_label = (0, 0, 0, 0, 0, self.max_len - t)
        sits = sits.padding(padd_tensor)
        doy = F.pad(doy, padd_doy)
        padd_index = torch.zeros(max_len)
        padd_index[t:] = 1
        padd_index = padd_index.bool()
    else:
        padd_index = None

    return sits, doy, padd_index


def concat_input(input1: InputInterp, input2: InputInterp) -> InputInterp:
    """

        Args:
            input1 ():
            input2 ():

        Returns:
    Input Interp with 5D sits f,t,c,h,w
    """
    sits = torch.stack([input1.sits, input2.sits], dim=0)
    doy = torch.stack([input1.doy, input2.doy], dim=0)
    mask_valid = torch.stack([input1.mask_valid, input2.mask_valid])
    padd_index = torch.stack([input1.padd_index, input2.padd_index])
    if input1.true_doy is not None:
        true_doy = torch.stack([input1.true_doy, input2.true_doy])
    else:
        true_doy = None
    return InputInterp(
        sits,
        doy=doy,
        mask_valid=mask_valid,
        padd_index=padd_index,
        true_doy=true_doy,
    )
