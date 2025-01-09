import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from mmmv_ssl.data.dataclass import BatchOneMod, SITSOneMod
from openeo_mmdc.dataset.to_tensor import load_all_transforms

from mmdc_downstream_pastis.datamodule.datatypes import MMDCDataStruct
from mmdc_downstream_pastis.encode_series.encode_alise_s1_original_pt import (
    load_checkpoint,
)
from mmdc_downstream_pastis.encode_series.encode_malise_aux_pt import (
    apply_transform_basic,
)

from .utils import (
    extract_gt_points,
    generate_coord_matrix,
    get_index,
    get_slices,
    get_tcd_gt,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_tile_malice_aux(
    folder_prepared_data: str | Path | None,
    folder_data: str | Path | None,
    path_alise_model: str | Path,
    path_csv: str | Path,
    output_folder: str | Path,
    gt_file: str | Path = None,
    satellites: list[str] = ["s1_asc"],
):
    """
    Encode a tile with ALISE algorithm with a sliding window
    We divide the sliding window of 768 pixels into 4 overlapping patches
    to overcome memory problems
    """

    margin = int(re.search(r"m_([0-9]*)", str(folder_prepared_data)).group(1))
    window = int(re.search(r"w_([0-9]*)", str(folder_prepared_data)).group(1))
    log.info(f"Margin={margin}")
    log.info(f"Window={window}")

    transforms = load_all_transforms(
        path_csv, modalities=["s1_asc", "s2", "agera5", "dem"]
    )

    # if path_alise_model.endswith("onnx"):
    #     import onnxruntime
    # else:

    for sat in satellites:
        # if path_alise_model.endswith("onnx"):
        #     sess_opt = onnxruntime.SessionOptions()
        #     sess_opt.intra_op_num_threads = 16
        #     ort_session = onnxruntime.InferenceSession(
        #         path_alise_model,
        #         sess_opt,
        #         providers=["CUDAExecutionProvider"],
        #     )
        # else:
        hydra_conf = path_alise_model.split(".")[0] + "_config.yaml"

        repr_encoder = load_checkpoint(hydra_conf, path_alise_model, satellites[0]).to(
            DEVICE
        )

        q, f = repr_encoder.nq, repr_encoder.encoder.ubarn.d_model

        if sat == "s1_asc":
            dates_final = pd.DataFrame(
                torch.load(os.path.join(folder_data, "s1_asc_dates.pt"))
            )["date_s1_asc"].values
        else:
            dates_final = np.load(os.path.join(folder_data, "dates_s2.npy"))[1:]
        print(dates_final)

        reference_date = "2014-03-03"
        doy = torch.Tensor(
            (pd.to_datetime(dates_final) - pd.to_datetime(reference_date)).days
        )
        print(doy)
        tcd_values = None
        if gt_file is not None:
            tcd_values = get_tcd_gt(gt_file)

        sliced_gt = {}
        # for enum, sliced in enumerate(slices_list[::-1]):
        #     enum = len(slices_list) - enum - 1

        Path(os.path.join(output_folder, sat)).mkdir(exist_ok=True, parents=True)

        Path(os.path.join(folder_prepared_data, sat)).mkdir(exist_ok=True, parents=True)

        months_folders = ["3", "4", "5", "6", "7", "8", "9", "10"]

        slices_list = get_slices(
            folder_data, months_folders, window, margin=margin, drop_incomplete=True
        )

        for enum, sliced in enumerate(slices_list):
            if not Path(
                os.path.join(
                    output_folder,
                    sat,
                    f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv",
                )
            ).exists():
                log.info(f"Opening patch {enum}")
                data_files = {}
                data_files[sat] = torch.load(
                    os.path.join(folder_prepared_data, sat, f"{sat}_tile_{enum}.pt")
                )

                data = {}
                data[sat] = data_files[sat]["data"]

                xy_matrix = generate_coord_matrix(sliced)[
                    :, margin:-margin, margin:-margin
                ]

                log.info(f"Encoding patch {enum}")

                ind = []
                tcd_values_sliced = None
                if tcd_values is not None:
                    ind, tcd_values_sliced = get_index(tcd_values, xy_matrix)

                img = data[sat]["img"]  # [:, :, 32:-32, 32:-32]
                angles = data[sat]["angles"]
                img = torch.concat((img, angles), dim=-3)

                meteo = (
                    MMDCDataStruct.init_empty()
                    .fill_empty_from_dict(data[sat])
                    .concat_meteo()
                    .meteo
                )
                meteo = rearrange(meteo, "b t d c h w -> b t c d h w", c=8)
                dem = data[sat]["dem"]

                print(len(doy))
                print(doy)

                # Prepare patch
                ref_norm = apply_transform_basic(
                    img, getattr(transforms, sat.lower()).transform
                )
                meteo_norm = rearrange(
                    apply_transform_basic(
                        rearrange(meteo, "b t c d h w -> b t c d (h w)"),
                        transforms.agera5.transform,
                    ),
                    "b t c d (h w) -> b t c d h w",
                    w=img.shape[-1],
                )
                dem_norm = apply_transform_basic(
                    dem[:, None, :, :, :], transforms.dem.transform
                )

                del data

                encoded_patch = {}

                encoded_patch[f"{sat}_lat_mu"] = np.zeros((q, f, 768, 768))

                local_patch = int(768 / 2)
                margin_local = 32
                for x in range(0, window, local_patch - margin_local):
                    if x + local_patch + margin_local > window:
                        continue
                    for y in range(0, window, local_patch - margin_local):
                        if y + local_patch + margin_local > window:
                            continue
                        log.info(
                            f"Patch {x}:{x + local_patch + margin_local}, "
                            f"{y}:{y + local_patch + margin_local}"
                        )

                        # if path_alise_model.endswith("onnx"):
                        #     input = {
                        #         "sits": sits.numpy(),
                        #         "tpe": doy[None, :].numpy(),
                        #         "padd_mask": torch.zeros(1, sits.shape[1])
                        #         .bool()
                        #         .numpy(),
                        #     }
                        #
                        #     ort_out = ort_session.run(None, input)[
                        #         0
                        #     ]  # (1, 10, 64, 64, 64)
                        # else:

                        input = SITSOneMod(
                            sits=ref_norm.to(DEVICE)[
                                0,
                                :,
                                :,
                                y : y + local_patch + margin_local,
                                x : x + local_patch + margin_local,
                            ],
                            input_doy=doy.to(DEVICE),
                            meteo=meteo_norm.to(DEVICE)[
                                0,
                                :,
                                :,
                                :,
                                y : y + local_patch + margin_local,
                                x : x + local_patch + margin_local,
                            ].mean((-1, -2)),
                        ).apply_padding(ref_norm.shape[1] + 1)

                        input = BatchOneMod(
                            sits=input.sits[None, ...],
                            input_doy=input.input_doy[None, ...],
                            padd_index=input.padd_mask[None, ...].to(DEVICE),
                            meteo=input.meteo[None, ...],
                        )

                        dem_patch = dem_norm[
                            :,
                            :,
                            :,
                            y : y + local_patch + margin_local,
                            x : x + local_patch + margin_local,
                        ]

                        with torch.no_grad():
                            ort_out = repr_encoder.forward_keep_input_dim(
                                input, dem=dem_patch.to(DEVICE)
                            ).repr.cpu()

                        if margin_local > 0:
                            encoded_patch[f"{sat}_lat_mu"][
                                :,
                                :,
                                y + margin_local : y + local_patch,
                                x + margin_local : x + local_patch,
                            ] = ort_out[
                                0,
                                :,
                                :,
                                margin_local:-margin_local,
                                margin_local:-margin_local,
                            ]
                        else:
                            encoded_patch[f"{sat}_lat_mu"][
                                :,
                                :,
                                y : y + local_patch,
                                x : x + local_patch,
                            ] = ort_out[0]

                encoded_patch["matrix"] = xy_matrix
                encoded_patch[f"{sat}_lat_mu"] = encoded_patch[f"{sat}_lat_mu"][
                    :, :, margin:-margin, margin:-margin
                ]

                encoded_patch[f"{sat}_doy"] = np.arange(
                    encoded_patch[f"{sat}_lat_mu"].shape[0]
                )

                # Save encoded patch
                torch.save(
                    {
                        "img": torch.Tensor(encoded_patch[f"{sat}_lat_mu"]),
                        "matrix": {
                            "x_min": xy_matrix[0, 0].min(),
                            "x_max": xy_matrix[0, 0].max(),
                            "y_min": xy_matrix[1, :, 0].min(),
                            "y_max": xy_matrix[1, :, 0].max(),
                        },
                    },
                    os.path.join(output_folder, sat, f"{sat}_{enum}.pt"),
                )

                # Save GT data
                if ind.size > 0 and tcd_values_sliced is not None:
                    df_gt, days = extract_gt_points(
                        encoded_patch,
                        sat,
                        ind,
                        tcd_values_sliced,
                        sliced_gt,
                    )
                    df_gt.to_csv(
                        os.path.join(
                            output_folder,
                            sat,
                            f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv",
                        )
                    )
                    if not Path(
                        os.path.join(
                            output_folder, sat, f"gt_tcd_t32tnt_days_{sat}.npy"
                        )
                    ).exists():
                        np.save(
                            os.path.join(
                                output_folder, sat, f"gt_tcd_t32tnt_days_{sat}.npy"
                            ),
                            days,
                        )

                del encoded_patch
