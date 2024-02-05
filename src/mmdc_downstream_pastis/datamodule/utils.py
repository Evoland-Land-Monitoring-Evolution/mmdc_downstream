import logging.config
import random
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from openeo_mmdc.dataset.dataclass import MaskMod
from torch import Tensor
from torch.nn import functional as F

from .dataclass import InputInterp


from mmdc_singledate.datamodules.datatypes import MMDCMeteoData
from dataclasses import dataclass, fields

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

    def fill_empty_from_dict(self, dictionary: dict):
        """Fill an empty dataclass instance from dictionary"""
        for key, value in dictionary.items():
            key = key[6:] if key.startswith("meteo") else key
            if key in self.meteo.__dict__:
                setattr(self.meteo, key, value)
            elif key in self.data.__dict__:
                setattr(self.data, key, value)
            elif key == "dem":
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

    def concat_meteo(self):
        """Concatenate meteo data into 8-features image series"""
        setattr(self,
                "meteo",
                torch.cat([getattr(self.meteo, field.name).unsqueeze(-3)
                           for field in fields(self.meteo)], -3)
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
        setattr(self, "meteo", getattr(self, "meteo")[:, :, :, y: y + cs, x: x + cs])
        setattr(self, "dem", getattr(self, "dem")[:, y: y + cs, x: x + cs])
        for field in fields(self.data):
            setattr(self.data, field.name, getattr(self.data, field.name)[:, :, y: y + cs, x: x + cs])
        return self

    def padding(self, padd_tensor):
        setattr(self, "meteo", F.pad(getattr(self, "meteo"), (0, 0) + padd_tensor))
        # setattr(self, "dem", F.pad(getattr(self, "dem"), padd_tensor))
        for field in fields(self.data):
            setattr(self.data, field.name, F.pad(getattr(self.data, field.name), padd_tensor))
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


def split_tensor_in_two(
    sits: Tensor,
    time: Tensor,
    middle: int,
    len_sits: int,
    mask: Tensor | None = None,
) -> OutputSplitTensorin2:
    """
    COnvert array and time np.array into torch Tensor as well as separate the input in two given the middle position.
    When required temporal padding is achieved
    Args:
        sits ():
        time ():
        middle ():
        len_sits ():

    Returns:

    """

    array_before, array_after = split_before_after_tensor(sits, middle)
    my_logger.debug(f"before {array_before.shape} after {array_after.shape}")
    time_before = torch.unsqueeze(time[:middle], dim=0)  # f t
    time_after = torch.unsqueeze(time[middle:], dim=0)  # f t
    if mask is not None:
        my_logger.debug(f"in split fn {mask[0, :, 0, 0]}")
        mask_before, mask_after = split_before_after_tensor(mask, middle)
        my_logger.debug(f"{mask_before[0, 0, :, 0, 0]}")
    else:
        mask_before, mask_after = None, None

    if middle != len_sits:
        my_logger.debug("Add temporal padding")
        array_before, time_before, padd_before = temp_padd_split(
            array_before, time_before, len_sits=len_sits
        )
        array_after, time_after, padd_after = temp_padd_split(
            array_after, time_after, len_sits=len_sits
        )
        my_logger.debug(
            f"before {array_before.shape} after {array_after.shape}"
        )
        padd_index = torch.stack([padd_before, padd_after], dim=0)
        if mask is not None:
            mask_before = padd_sits_split(
                mask_before, len_sits=len_sits, t=mask_before.shape[2]
            )
            mask_after = padd_sits_split(
                mask_after, len_sits=len_sits, t=mask_after.shape[2]
            )
    else:
        my_logger.debug(
            "No temporal padding has been applied as the SITS can be"
            " split perfectly in two"
        )
        padd_index = (torch.zeros(2, len_sits)).bool()
    # print(f"mask bef in split {mask_after[0,0,:,0,0]}")
    return OutputSplitTensorin2(
        array_before=array_before,
        array_after=array_after,
        time_before=time_before,
        time_after=time_after,
        padd_index=padd_index,
        mask_before=mask_before,
        mask_after=mask_after,
    )


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


@dataclass
class OutLoadSits:
    array: Tensor
    time: Tensor
    max_len: int



@dataclass
class OutPaddBERT:
    padd_array: Tensor
    padd_time: Tensor
    padd_corruption_mask: Tensor
    padd_val: int
    padd_index: Tensor


@dataclass
class OutClassifItem:
    sits: MMDCDataStruct
    doy: Tensor
    target: Tensor
    padd_index: Tensor | None
    padd_val: Tensor
    item: int
    mask_loss: Tensor | None = None
    s2_path: str | None = None
    true_doy: Tensor | None = None


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
