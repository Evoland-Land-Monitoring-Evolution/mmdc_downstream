#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""Pretrained MMDC model module"""

import os
from pathlib import Path
from typing import Any

import torch
from hydra import utils as hydra_utils
from mmdc_singledate.models.lightning.full_experts import (
    MMDCFullExpertsLitModule,
    MMDCFullLitModule,
)
from mmdc_singledate.models.torch.full_experts import (
    AuxData,
    MMDCFullExpertsModule,
    MMDCFullModule,
)
from omegaconf import OmegaConf

from ..models.datatypes import LatentPred


class PretrainedMMDC:
    """Pretrained MMDC model class"""

    def __init__(
        self,
        pretrained_path: str | Path,
        model_name: str,
        model_type: str,
    ):
        if pretrained_path is not None:
            self.lightning_module: MMDCFullExpertsLitModule | MMDCFullLitModule
            self.pretrained_path = os.path.abspath(pretrained_path)
            self.model_name = model_name
            self.model_type = model_type
            self.device = "cpu"
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model_mmdc: MMDCFullModule | MMDCFullExpertsModule = (
                self.init_mmdc_model()
            )
            self.set_mmdc_stats()

            assert self.lightning_module is not None
            self.lightning_module.freeze()

            (
                self.model_mmdc.nb_enc_cropped_hw,
                self.model_mmdc.nb_cropped_hw,
            ) = self.model_mmdc.compute_cropping()
        else:
            self.model_mmdc = None

    def init_mmdc_model(self) -> MMDCFullModule | MMDCFullExpertsModule:
        """Load pretrained MMDC model from checkpoint"""
        ckpt_path = os.path.join(self.pretrained_path, self.model_name + ".ckpt")
        cfg = OmegaConf.load(os.path.join(self.pretrained_path, "config.yaml"))
        model = hydra_utils.instantiate(cfg.model.model)
        if self.model_type == "baseline":
            self.lightning_module = MMDCFullLitModule.load_from_checkpoint(
                ckpt_path, model=model, map_location=self.device
            )
        else:  # experts
            self.lightning_module = MMDCFullExpertsLitModule.load_from_checkpoint(
                ckpt_path, model=model, map_location=self.device
            )
        return self.lightning_module.model.to(self.device)

    def set_mmdc_stats(self):
        """Set stats from file associated with checkpoint"""
        stats = torch.load(
            os.path.join(self.pretrained_path, "stats.pt"), weights_only=False
        )
        self.lightning_module.set_stats(stats)

    # def get_latent_mmdc_pred(self, data: Any) -> torch.tensor:
    #     # self.model_mmdc.eval()
    #     return self.get_latent_mmdc(data)

    def get_latent_mmdc(self, data: Any) -> torch.tensor:
        """Predict latent variable"""
        with torch.no_grad():
            self.model_mmdc.eval()

            s1_x, s2_x, meteo_x, dem_x = self.model_mmdc.standardize_inputs(
                data.s1_x, data.s2_x, data.meteo_x, data.dem_x
            )
            embs = self.model_mmdc.compute_aux_embeddings(
                AuxData(data.s1_a, data.s2_a, meteo_x, dem_x)
            )
            latents = self.model_mmdc.generate_latents(s1_x, s2_x, embs)
            if self.model_type == "baseline":
                return LatentPred(
                    latents.sen1.mean,
                    latents.sen1.logvar,
                    latents.sen2.mean,
                    latents.sen2.logvar,
                )
            latent_experts = self.model_mmdc.generate_latent_experts(latents)
            return LatentPred(
                latents.sen1.mean,
                latents.sen1.logvar,
                latents.sen2.mean,
                latents.sen2.logvar,
                latent_experts.latent_experts.mean,
                latent_experts.latent_experts.logvar,
            )
