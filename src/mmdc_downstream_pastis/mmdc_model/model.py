#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""Pretrained MMDC model module"""

from pathlib import Path
from typing import Any
from einops import rearrange

import torch
from mmdc_singledate.models.torch.full_experts import AuxData
from mmdc_downstream.mmdc_model.model import PretrainedMMDC
from mmdc_downstream.models.datatypes import LatentPred

from mmdc_singledate.datamodules.datatypes import MMDCShiftScales, ShiftScale
from mmdc_singledate.utils.train_utils import standardize_data

def reshape_sits(sits: torch.Tensor):
    return rearrange(sits, 'b t c h w -> (b t) c h w')


def reshape_back_sits(sits: torch.Tensor, batch_size: int):
    return rearrange(sits, '(b t) c h w -> b t c h w', b=batch_size)


class PretrainedMMDCPastis(PretrainedMMDC):
    """Pretrained MMDC model class"""

    def __init__(
            self,
            pretrained_path: str | Path,
            model_name: str,
            model_type: str,
    ):
        super().__init__(pretrained_path, model_name, model_type)


    def get_latent_s1_mmdc(self, data: Any) -> torch.tensor:
        """Predict latent variable"""
        with torch.no_grad():
            self.model_mmdc.eval()
            bs = data.s1_x.shape[0]
            s1_x, s2_x, meteo_x, dem_x = self.model_mmdc.standardize_inputs(
                reshape_sits(data.s1_x),
                reshape_sits(data.s2_x),
                reshape_sits(data.meteo_x),
                reshape_sits(data.dem_x))
            embs = self.model_mmdc.compute_aux_embeddings(
                AuxData(reshape_sits(data.s1_a),
                        reshape_sits(data.s2_a),
                        reshape_sits(meteo_x),
                        reshape_sits(dem_x)))
            latents = self.model_mmdc.generate_latents(s1_x, s2_x, embs)
            if self.model_type == "baseline":
                return LatentPred(reshape_back_sits(latents.sen1.mean, bs),
                                  reshape_back_sits(latents.sen1.logvar, bs),
                                  reshape_back_sits(latents.sen2.mean, bs),
                                  reshape_back_sits(latents.sen2.logvar, bs))
            else:
                latent_experts = self.model_mmdc.generate_latent_experts(latents)
                return LatentPred(reshape_back_sits(latents.sen1.mean, bs),
                                  reshape_back_sits(latents.sen1.logvar, bs),
                                  reshape_back_sits(latents.sen2.mean, bs),
                                  reshape_back_sits(latents.sen2.logvar, bs),
                                  reshape_back_sits(latent_experts.latent_experts.mean, bs),
                                  reshape_back_sits(latent_experts.latent_experts.logvar, bs))

    def get_latent_mmdc(self, data: Any) -> torch.tensor:
        """Predict latent variable"""
        with torch.no_grad():
            self.model_mmdc.eval()
            bs = data.s1_x.shape[0]
            s1_x, s2_x, meteo_x, dem_x = self.model_mmdc.standardize_inputs(
                reshape_sits(data.s1_x),
                reshape_sits(data.s2_x),
                reshape_sits(data.meteo_x),
                reshape_sits(data.dem_x))
            embs = self.model_mmdc.compute_aux_embeddings(
                AuxData(reshape_sits(data.s1_a),
                        reshape_sits(data.s2_a),
                        reshape_sits(meteo_x),
                        reshape_sits(dem_x)))
            latents = self.model_mmdc.generate_latents(s1_x, s2_x, embs)
            if self.model_type == "baseline":
                return LatentPred(reshape_back_sits(latents.sen1.mean, bs),
                                  reshape_back_sits(latents.sen1.logvar, bs),
                                  reshape_back_sits(latents.sen2.mean, bs),
                                  reshape_back_sits(latents.sen2.logvar, bs))
            else:
                latent_experts = self.model_mmdc.generate_latent_experts(latents)
                return LatentPred(reshape_back_sits(latents.sen1.mean, bs),
                                  reshape_back_sits(latents.sen1.logvar, bs),
                                  reshape_back_sits(latents.sen2.mean, bs),
                                  reshape_back_sits(latents.sen2.logvar, bs),
                                  reshape_back_sits(latent_experts.latent_experts.mean, bs),
                                  reshape_back_sits(latent_experts.latent_experts.logvar, bs))

    def standardize_inputs(
        self,
        s1_x: torch.Tensor | None = None,
        s2_x: torch.Tensor | None = None,
        meteo_x: torch.Tensor = None,
        dem_x: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        """Standardize the input data tensors using the scale information"""
        assert self.model_mmdc.scales is not None
        if s1_x is not None:
            self.model_mmdc.scales.sen1.shift = self.model_mmdc.scales.sen1.shift.type_as(s1_x)
            self.model_mmdc.scales.sen1.scale = self.model_mmdc.scales.sen1.scale.type_as(s1_x)
            s1_x = standardize_data(
                s1_x,
                shift=self.model_mmdc.scales.sen1.shift,
                scale=self.model_mmdc.scales.sen1.scale,
                # null_value=np.log10(S1_NOISE_LEVEL),
            )
        if s2_x is not None:
            self.model_mmdc.scales.sen2.shift = self.model_mmdc.scales.sen2.shift.type_as(s2_x)
            self.model_mmdc.scales.sen2.scale = self.model_mmdc.scales.sen2.scale.type_as(s2_x)
            s2_x = standardize_data(
                s2_x,
                shift=self.model_mmdc.scales.sen2.shift,
                scale=self.model_mmdc.scales.sen2.scale,
            )

        self.model_mmdc.scales.meteo.shift = self.model_mmdc.scales.meteo.shift.type_as(meteo_x)
        self.model_mmdc.scales.meteo.scale = self.model_mmdc.scales.meteo.scale.type_as(meteo_x)
        self.model_mmdc.scales.dem.shift = self.model_mmdc.scales.dem.shift.type_as(dem_x)
        self.model_mmdc.scales.dem.scale = self.model_mmdc.scales.dem.scale.type_as(dem_x)

        meteo_x = standardize_data(
            meteo_x,
            shift=self.model_mmdc.scales.meteo.shift,
            scale=self.model_mmdc.scales.meteo.scale,
        )

        dem_x = standardize_data(
            dem_x,
            shift=self.model_mmdc.scales.dem.shift,
            scale=self.model_mmdc.scales.dem.scale,
        )
        return s1_x, s2_x, meteo_x, dem_x