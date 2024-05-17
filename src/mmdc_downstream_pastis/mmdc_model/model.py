#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""Pretrained MMDC model module"""
from pathlib import Path

import torch
from einops import rearrange, repeat
from mmdc_singledate.models.datatypes import S1S2VAEAuxiliaryEmbeddings, VAELatentSpace
from mmdc_singledate.utils.train_utils import standardize_data

from mmdc_downstream_lai.mmdc_model.model import PretrainedMMDC
from mmdc_downstream_pastis.utils.utils import MMDCPartialBatch


def repeat_dem(dem: torch.Tensor, time: int) -> torch.Tensor:
    """Adds temporal dimension to each batch dem element"""
    return repeat(dem, "b c h w -> b t c h w", t=time)


def reshape_sits(sits: torch.Tensor) -> torch.Tensor:
    """Reshape pastis TS"""
    return rearrange(sits, "b t c h w -> (b t) c h w")


def reshape_back_sits(sits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reshape embedded images to TS"""
    return rearrange(sits, "(b t) c h w -> b t c h w", b=batch_size)


class PretrainedMMDCPastis(PretrainedMMDC):
    """Pretrained MMDC model class"""

    def __init__(
        self,
        pretrained_path: str | Path,
        model_name: str,
        model_type: str,
    ):
        super().__init__(pretrained_path, model_name, model_type)

        self.compute_aux_embeddings = {
            "S1": self.compute_aux_embeddings_s1,
            "S2": self.compute_aux_embeddings_s2,
        }
        assert self.model_mmdc.scales is not None

        # TODO checks nans in angles, dem and masks

    def prepare_aux_embeddings(self, data: MMDCPartialBatch) -> torch.Tensor:
        """Prepare auxillary embeddings for S1 or S2 data"""
        with torch.no_grad():
            self.model_mmdc.eval()
            meteo_x = self.standardize_meteo(reshape_sits(data.meteo))
            dem_x = self.standardize_dem(
                reshape_sits(repeat_dem(data.dem, time=data.img.shape[1]))
            )
            return self.compute_aux_embeddings[data.type](
                reshape_sits(data.angles).nan_to_num(), meteo_x, dem_x
            )

    def compute_aux_embeddings_s1(
        self,
        s1_angles: torch.Tensor,
        meteo: torch.Tensor,
        dem: torch.Tensor,
    ) -> S1S2VAEAuxiliaryEmbeddings:
        self.model_mmdc.eval()
        """Generate the embeddings for the S1 auxiliary data"""
        s1_a_emb = self.model_mmdc.nets.embedders.s1_angles(s1_angles)
        s1_dem_emb = self.model_mmdc.nets.embedders.s1_dem(dem, s1_a_emb)
        meteo_emb = self.model_mmdc.nets.embedders.meteo(meteo)
        return S1S2VAEAuxiliaryEmbeddings(s1_a_emb, None, s1_dem_emb, None, meteo_emb)

    def compute_aux_embeddings_s2(
        self,
        s2_angles: torch.Tensor,
        meteo: torch.Tensor,
        dem: torch.Tensor,
    ) -> S1S2VAEAuxiliaryEmbeddings:
        """Generate the embeddings for the S2 auxiliary data"""
        self.model_mmdc.eval()
        s2_a_emb = self.model_mmdc.nets.embedders.s2_angles(s2_angles)
        s2_dem_emb = self.model_mmdc.nets.embedders.s2_dem(dem, s2_a_emb)
        meteo_emb = self.model_mmdc.nets.embedders.meteo(meteo)
        return S1S2VAEAuxiliaryEmbeddings(None, s2_a_emb, None, s2_dem_emb, meteo_emb)

    def get_latent_s1_mmdc(self, data: MMDCPartialBatch) -> VAELatentSpace:
        """Predict latent S1 variable"""
        with torch.no_grad():
            self.model_mmdc.eval()
            batch_size = data.img.shape[0]
            embs = self.prepare_aux_embeddings(data)
            latent_s1 = self.model_mmdc.generate_latent_s1(
                self.standardize_s1(reshape_sits(data.img)), embs
            )
            return VAELatentSpace(
                reshape_back_sits(latent_s1.mean, batch_size),
                reshape_back_sits(latent_s1.logvar, batch_size),
            )

    def get_latent_s2_mmdc(self, data: MMDCPartialBatch) -> VAELatentSpace:
        """Predict latent S2 variable"""
        with torch.no_grad():
            self.model_mmdc.eval()
            batch_size = data.img.shape[0]
            embs = self.prepare_aux_embeddings(data)

            latent_s2 = self.model_mmdc.generate_latent_s2(
                self.standardize_s2(reshape_sits(data.img)), embs
            )
            return VAELatentSpace(
                reshape_back_sits(latent_s2.mean, batch_size),
                reshape_back_sits(latent_s2.logvar, batch_size),
            )

    def standardize_s1(self, s1_x: torch.Tensor) -> torch.Tensor:
        """Standardize S1"""
        assert self.model_mmdc.scales is not None
        self.model_mmdc.scales.sen1.shift = self.model_mmdc.scales.sen1.shift.type_as(
            s1_x
        )
        self.model_mmdc.scales.sen1.scale = self.model_mmdc.scales.sen1.scale.type_as(
            s1_x
        )
        return standardize_data(
            s1_x,
            shift=self.model_mmdc.scales.sen1.shift,
            scale=self.model_mmdc.scales.sen1.scale,
        ).nan_to_num()

    def standardize_s2(self, s2_x: torch.Tensor) -> torch.Tensor:
        """Standardize S2"""
        assert self.model_mmdc.scales is not None
        self.model_mmdc.scales.sen2.shift = self.model_mmdc.scales.sen2.shift.type_as(
            s2_x
        )
        self.model_mmdc.scales.sen2.scale = self.model_mmdc.scales.sen2.scale.type_as(
            s2_x
        )
        return standardize_data(
            s2_x,
            shift=self.model_mmdc.scales.sen2.shift,
            scale=self.model_mmdc.scales.sen2.scale,
        ).nan_to_num()

    def standardize_meteo(self, meteo_x: torch.Tensor) -> torch.Tensor:
        """Standardize meteo"""
        assert self.model_mmdc.scales is not None
        self.model_mmdc.scales.meteo.shift = self.model_mmdc.scales.meteo.shift.type_as(
            meteo_x
        )
        self.model_mmdc.scales.meteo.scale = self.model_mmdc.scales.meteo.scale.type_as(
            meteo_x
        )

        return standardize_data(
            meteo_x,
            shift=self.model_mmdc.scales.meteo.shift,
            scale=self.model_mmdc.scales.meteo.scale,
        ).nan_to_num()

    def standardize_dem(self, dem_x: torch.Tensor) -> torch.Tensor:
        """Standardize DEM"""
        assert self.model_mmdc.scales is not None
        self.model_mmdc.scales.dem.shift = self.model_mmdc.scales.dem.shift.type_as(
            dem_x
        )
        self.model_mmdc.scales.dem.scale = self.model_mmdc.scales.dem.scale.type_as(
            dem_x
        )

        return standardize_data(
            dem_x,
            shift=self.model_mmdc.scales.dem.shift,
            scale=self.model_mmdc.scales.dem.scale,
        ).nan_to_num()
