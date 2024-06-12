#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
import os
from pathlib import Path

import torch
from einops import rearrange
from mmdc_singledate.models.datatypes import VAELatentSpace

from mmdc_downstream_pastis.encode_series.encode import back_to_date, build_dm
from mmdc_downstream_tcd.tile_processing.encode_tile_pvae import mmdc2pvae_batch

log = logging.getLogger(__name__)


def encode_series_pvae(
    path_jit_model: str | Path,
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sats: list[str],
    output_path: str | Path,
):
    ts_net = torch.jit.load(path_jit_model)
    ts_net.eval()

    """Encode PASTIS SITS S2 with prosailVAE"""
    dm = build_dm(dataset_path_oe, dataset_path_pastis, sats)

    dataset = dm.instanciate_dataset(fold=None)
    loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)
    log.info("Loader is ready")
    for batch in loader:
        log.info(batch.id_patch)
        sits = batch.sits["S2"].sits

        s2_ref = sits.data.img.squeeze(0)
        s2_angles = sits.data.angles.squeeze(0)
        s2_mask = sits.data.mask.squeeze(0)
        t, c, h, w = s2_ref.shape

        # Prepare patch
        s2_ref, s2_angles = mmdc2pvae_batch(s2_ref, s2_angles)

        s2_ref_reshape = rearrange(s2_ref, "t c h w ->  (t h w) c")
        s2_angles_reshape = rearrange(s2_angles, "t c h w ->  (t h w) c")

        res = ts_net(s2_ref_reshape, s2_angles_reshape)
        res_shaped = rearrange(res, "(t h w) c ml  ->  t c ml h w", h=h, w=w)
        res_shaped[s2_mask.unsqueeze(2).expand_as(res_shaped).bool()] = torch.nan

        encoded_pastis_s2 = {
            "latents": VAELatentSpace(
                res_shaped[:, :, 0, :, :].detach(), res_shaped[:, :, 1, :, :].detach()
            ),
            "mask": s2_mask,
            "dates": back_to_date(batch.sits["S2"].true_doy, dm.reference_date)[0],
        }
        torch.save(
            encoded_pastis_s2,
            os.path.join(output_path, "S2", f"S2_{batch.id_patch[0]}.pt"),
        )
