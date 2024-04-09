from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.utils.data as tdata

from mmdc_downstream_pastis.datamodule.utils import MMDCDataStruct


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

    # task: PASTISTask
    dataset_path_oe: str
    dataset_path_pastis: str
    # folder: str
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
class PastisBatch:
    sits: MMDCDataStruct
    input_doy: torch.Tensor
    padd_val: torch.Tensor
    padd_index: torch.Tensor
    true_doy: torch.Tensor = None

    def w(self):
        return self.sits.data.img.shape[-1]

    def h(self):
        return self.sits.data.img.shape[-2]

    @staticmethod
    def init_empty():
        """Create an empty dataclass instance"""
        return PastisBatch(None, None, None, None, None)

    def fill_empty_from_dict(self, dictionary: dict):
        """Fill an empty dataclass instance from dictionary"""
        for key, value in dictionary.items():
            setattr(self, key, value)
        return self

    #
    #
    # # TODO: cloud mask
    # def pin_memory(self):
    #     self.sits = self.sits.pin_memory()
    #     self.input_doy = self.input_doy.pin_memory()
    #     self.padd_index = self.padd_index.pin_memory()
    #     if self.true_doy is not None:
    #         self.true_doy.pin_memory()
    #     return self


@dataclass
class ModuleInput:
    sits: torch.Tensor
    input_doy: torch.Tensor
    padd_index: torch.Tensor | None
    cld_mask: torch.Tensor | None = None
    true_doy: torch.Tensor | None = None

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.sits = self.sits.pin_memory()
        self.input_doy = self.input_doy.pin_memory()
        self.padd_index = self.padd_index.pin_memory()
        if self.cld_mask is not None:
            self.cld_mask = self.cld_mask.pin_memory()
        if self.true_doy is not None:
            self.true_doy.pin_memory()
        return self


@dataclass
class OneSatellitePatch:
    sits: MMDCDataStruct
    input_doy: torch.Tensor
    padd_index: torch.Tensor | None
    padd_val: torch.Tensor
    true_doy: np.ndarray | None
