#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime
import pandas as pd
import torch
from einops import rearrange
from openeo_mmdc.dataset.to_tensor import load_transform_one_mod

from mmdc_downstream_pastis.encode_series.encode import back_to_date, build_dm

log = logging.getLogger(__name__)


def encode_series_alise_s1(
    path_alise_model: str | Path,
    path_csv: str | Path,
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sat: str,
    output_path: str | Path,
):
    sess_opt = onnxruntime.SessionOptions()
    sess_opt.intra_op_num_threads = 8
    transform = load_transform_one_mod(path_csv, mod=sat.lower()).transform
    print(transform)
    ort_session = onnxruntime.InferenceSession(
        path_alise_model,
        sess_opt,
        providers=["CPUExecutionProvider"],
    )

    """Encode PASTIS SITS S2 with prosailVAE"""
    dm = build_dm(dataset_path_oe, dataset_path_pastis, [sat])

    dataset = dm.instanciate_dataset(fold=None)
    loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)
    log.info("Loader is ready")
    for batch in loader:
        log.info(batch.id_patch)
        sits = batch.sits[sat].sits

        img = sits.data.img.squeeze(0)
        mask = sits.data.mask.squeeze(0)

        reference_date = "2014-03-03"

        doy = torch.Tensor(
            (
                pd.to_datetime(
                    back_to_date(batch.sits[sat].true_doy, dm.reference_date)[0]
                )
                - pd.to_datetime(reference_date)
            ).days
        )

        if sat == "S1_ASC":
            s1_asc_2b = 10 ** img[:, :2]

            ratio = s1_asc_2b[:, 0] / s1_asc_2b[:, 1]

            s1_asc = torch.cat([s1_asc_2b, ratio[:, None, :, :]], dim=1)

            s1_asc = torch.log(s1_asc)

            s1_asc[mask.expand_as(s1_asc).bool()] = 0

            prepared_sits = s1_asc
        else:  # S2
            prepared_sits = img

        # Prepare patch
        ref_norm = rearrange(
            transform(rearrange(prepared_sits, "t c h w -> c t h w")),
            "c t h w -> 1 t c h w",
        )

        input = {
            "sits": ref_norm.cpu().numpy(),
            "tpe": doy[None, :].numpy(),
            "padd_mask": torch.zeros(1, ref_norm.shape[1]).bool().numpy(),
        }

        ort_out = ort_session.run(None, input)[0]  # (1, 10, 64, 64, 64)

        for i in range(ort_out[0].shape[0]):
            for j in range(ort_out[0].shape[1]):
                print(np.quantile(ort_out[0, i, j], q=[0.01, 0.5, 0.99]))

        torch.save(
            ort_out[0],
            os.path.join(output_path, sat, f"{sat}_{batch.id_patch[0]}.pt"),
        )
