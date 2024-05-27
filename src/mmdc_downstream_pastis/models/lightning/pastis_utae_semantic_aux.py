#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
""" Lightning module for lai regression prediction """

import logging
from pathlib import Path

import omegaconf
import torch

# from einops import rearrange
from mmdc_singledate.datamodules.datatypes import ShiftScale
from torch import nn

from mmdc_downstream_pastis.models.lightning.pastis_utae_semantic import PastisUTAE

from ...datamodule.datatypes import BatchInputUTAE
from ..torch.utae import UTAE
from ..torch.utae_fusion import UTAEFusion

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logging.getLogger(__name__).setLevel(logging.INFO)


def compute_stats(data_type: str, sat_stats: dict) -> ShiftScale:
    scale_regul = nn.Threshold(1e-10, 1.0)
    return ShiftScale(
        sat_stats[data_type][1],
        scale_regul((sat_stats[data_type][2] - sat_stats[data_type][0]) / 2.0),
    )


def set_aux_stats(
    stats_path: dict[str | Path] | str | Path | None,
    device: torch.device,
    sat: str,
) -> dict[str, ShiftScale] | ShiftScale:
    if type(stats_path) not in (dict, omegaconf.dictconfig.DictConfig):
        sat_stats = torch.load(stats_path)
        stats_sat = {d: compute_stats(d, sat_stats) for d in sat_stats}
        angles_len = 6 if sat == "S2" else 1
        return ShiftScale(
            torch.concat(
                [
                    stats_sat["img"].shift,
                    torch.zeros(angles_len),
                    stats_sat["meteo"].shift.repeat(6),
                    stats_sat["dem"].shift,
                ],
                0,
            ).to(device),
            torch.concat(
                [
                    stats_sat["img"].scale,
                    torch.ones(angles_len),
                    stats_sat["meteo"].scale.repeat(6),
                    stats_sat["dem"].scale,
                ],
                0,
            ).to(device),
        )
    return {sat: set_aux_stats(stats_path[sat], device, sat) for sat in stats_path}


def standardize_data(
    data: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    pad_index: bool,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Standardize 5d image tensor data by removing a shift and then scaling
    :param data: Tensor of data with shape [C] or [B,T,C,H,W]
    :param shfit: 1D tensor with shifting data
    :param scale: 1D tensor with scaling data
    :pad_index: IDX of Padded SITS dates
    :pad_value: padding value
    """
    # print(shift)
    # print(scale)
    # print("NON-NORMALIZED")
    # resh = rearrange(data, "b t c h w -> (b t h w) c ")
    # print(torch.round(resh.mean(0), decimals=1),
    #       torch.round(resh.median(0).values, decimals=1),
    #       torch.round(resh.std(0), decimals=1),
    #       torch.round(resh.min(0).values, decimals=1),
    #       torch.round(resh.max(0).values, decimals=1))
    # print("NORMALIZED")
    shift = shift[None, None, :, None, None].expand_as(data)
    scale = scale[None, None, :, None, None].expand_as(data)
    data_stand = torch.full_like(data, fill_value=pad_value)
    data_stand[~pad_index] = ((data - shift) / scale)[~pad_index]
    # resh = rearrange(data_stand, 'b t c h w -> (b t h w) c ').nan_to_num()
    # print(torch.round(resh.mean(0), decimals=1),
    #       torch.round(resh.median(0).values, decimals=1),
    #       torch.round(resh.std(0), decimals=1),
    #       torch.round(resh.min(0).values, decimals=1),
    #       torch.round(resh.max(0).values, decimals=1))
    # We replace all nans by zeros after standardization
    return data_stand.nan_to_num()


class PastisAuxUTAE(PastisUTAE):
    """
    Pastis Semantic segmentation module.
    Raw image data is concatenated with auxiliary data
    """

    def __init__(
        self,
        model: UTAE | UTAEFusion,
        metrics_list: list[str] = ["cross"],
        losses_list: list[str] = ["cross"],
        lr: float = 0.001,
        lr_type: str = "constant",
        resume_from_checkpoint: str | None = None,
        stats_aux_path: dict[str | Path] | None = None,
    ):
        super().__init__(
            model, metrics_list, losses_list, lr, lr_type, resume_from_checkpoint
        )

        self.stats_aux_path = stats_aux_path
        self.stats = None

    def forward(self, batch: BatchInputUTAE) -> torch.Tensor:
        """Forward step"""
        sits = batch.sits

        if self.stats is not None:
            if type(sits) is dict:
                sits = {
                    sat: standardize_data(
                        sits[sat].clone(),
                        shift=self.stats[sat].shift,
                        scale=self.stats[sat].scale,
                        pad_index=batch.doy[sat] == self.trainer.datamodule.pad_value,
                        pad_value=self.trainer.datamodule.pad_value,
                    )
                    for sat in sits
                }
            else:
                sits = standardize_data(
                    sits.clone(),
                    shift=self.stats.shift,
                    scale=self.stats.scale,
                    pad_index=batch.doy == self.trainer.datamodule.pad_value,
                    pad_value=self.trainer.datamodule.pad_value,
                )
        return self.model.forward(sits, batch_positions=batch.doy)

    def on_fit_start(self) -> None:
        """On fit start, get the stats, and set them into the model"""
        assert hasattr(self.trainer, "datamodule")
        if self.stats_aux_path is not None:
            self.stats = set_aux_stats(
                self.stats_aux_path,
                device="cuda",
                sat=self.trainer.datamodule.sats[0]
                if type(self.stats_aux_path) is not dict
                else None,
            )
