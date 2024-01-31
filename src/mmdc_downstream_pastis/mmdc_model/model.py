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
