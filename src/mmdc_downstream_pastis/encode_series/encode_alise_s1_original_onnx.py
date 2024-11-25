# !/usr/bin/env python3
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

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def back_to_date(
    days_int: torch.Tensor, ref_date: str = "2018-09-01"
) -> list[np.array]:
    """
    Go back from integral doy (number of days from ref date)
    to calendar doy
    """
    return [
        np.asarray(
            pd.to_datetime(
                [
                    pd.Timedelta(dd, "d") + pd.to_datetime(ref_date)
                    for dd in days_int.cpu().numpy()[d]
                ]
            )
        )
        for d in range(len(days_int))
    ]


def encode_series_alise_onnx(
    path_alise_model: str | Path,
    path_csv: str | Path,
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sat: str,
    output_path: str | Path,
    postfix: str,
):
    """Encode PASTIS SITS S2 with MALICE"""

    log.info("Loader is ready")

    sess_opt = onnxruntime.SessionOptions()
    sess_opt.intra_op_num_threads = 16
    transform = load_transform_one_mod(path_csv, mod=sat.lower()).transform

    ort_session = onnxruntime.InferenceSession(
        path_alise_model,
        sess_opt,
        providers=["CPUExecutionProvider"],
    )

    for patch_path in np.sort(os.listdir(os.path.join(dataset_path_oe, sat))):
        id_patch = patch_path.split("_")[-1].split(".")[0]
        if not Path(
            os.path.join(output_path, sat + "_" + postfix, f"{sat}_{id_patch}.pt")
        ).exists():
            batch = torch.load(
                os.path.join(dataset_path_oe, sat, patch_path), map_location=DEVICE
            )
            log.info(id_patch)
            img = batch.sits

            mask = batch.mask

            true_doy = batch.dates_dt.values[:, 0]

            reference_date = "2014-03-03"

            doy = torch.Tensor(
                (pd.to_datetime(true_doy) - pd.to_datetime(reference_date)).days
            )

            if sat == "S1_ASC":
                s1_asc_2b = img[:, [1, 0]]

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
            # for i in range(ort_out[0].shape[0]):
            #     for j in range(ort_out[0].shape[1]):
            #         print(np.nanquantile(ort_out[0, i, j].cpu(), q=[0.01, 0.5, 0.99]))

            torch.save(
                torch.Tensor(ort_out[0]),
                os.path.join(output_path, sat + "_" + postfix, f"{sat}_{id_patch}.pt"),
            )
