import logging
import os

# from mmdc_downstream_tcd.tile_processing.utils import (
#     create_netcdf_encoded,
#     create_netcdf_one_date_encoded,
#     create_netcdf_s2_one_date,
# )
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmdc_singledate.datamodules.components.datamodule_components import (
    build_meteo,
    dem_height_aspect,
)

from mmdc_downstream_pastis.datamodule.datatypes import METEO_BANDS, PastisBatch
from mmdc_downstream_pastis.datamodule.utils import MMDCDataStruct
from mmdc_downstream_pastis.encode_series.encode import encode_one_batch
from mmdc_downstream_pastis.mmdc_model.model import PretrainedMMDCPastis
from mmdc_downstream_tcd.tile_processing.utils import (
    generate_coord_matrix,
    get_index,
    get_meteo_dates_modality,
    get_one_slice,
    get_s1,
    get_s2,
    get_sliced_modality,
    get_slices,
    get_tcd_gt,
    rearrange_ts,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


log = logging.getLogger(__name__)

satellites = ["s2", "s1_asc", "s1_desc"]
meteo_modalities = METEO_BANDS.keys()


def encode_tile(
    folder_data: str | Path,
    months_folders: list[str | int],
    window: int,
    output_folder: str | Path,
    mmdc_model: PretrainedMMDCPastis,
    gt_file: str | Path = None,
    s1_join: bool = True,
):
    """
    Encode a tile with MMDC algorithm with a sliding window
    """

    margin = mmdc_model.model_mmdc.nb_cropped_hw

    slices_list = get_slices(
        folder_data, months_folders, window, margin=margin, drop_incomplete=True
    )

    dates_meteo: dict[str, pd.DataFrame] = {}

    s2_dates = np.load(os.path.join(folder_data, "dates_s2.npy"))
    s1_dates_asc = np.load(os.path.join(folder_data, "dates_s1_asc_good.npy"))
    s1_dates_desc = np.load(os.path.join(folder_data, "dates_s1_desc_good.npy"))

    # print("computing s1 asc")
    # s2_dates = check_s2_clouds(folder_data, months_folders)
    # s1_dates_asc = check_s1_nans(folder_data, months_folders,
    #                          orbit="ASCENDING")
    # np.save(os.path.join(folder_data, "dates_s1_asc.npy"), s1_dates_asc)
    # print("computing s1 desc")
    # s1_dates_desc = check_s1_nans(folder_data, months_folders,
    #                          orbit="DESCENDING")
    # #
    # np.save(os.path.join(folder_data, "dates_s1_desc.npy"), s1_dates_desc)
    tcd_values = None
    if gt_file is not None:
        tcd_values = get_tcd_gt(gt_file)

    # Whether we produce separate embeddings for s1_asc and s1_desc or not
    satellites_to_write = ["s2", "s1"] if s1_join else satellites

    sliced_gt = {}
    # for enum, sliced in enumerate(slices_list[::-1]):
    #     enum = len(slices_list) - enum - 1
    for enum, sliced in enumerate(slices_list):
        if not np.all(
            [
                Path(
                    os.path.join(
                        output_folder, f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv"
                    )
                ).exists()
                for sat in satellites_to_write
            ]
        ):
            xy_matrix = generate_coord_matrix(sliced)[:, margin:-margin, margin:-margin]

            ind = []
            tcd_values_sliced = None
            if tcd_values is not None:
                ind, tcd_values_sliced = get_index(tcd_values, xy_matrix)

            if tcd_values is None or (tcd_values is not None and ind.size > 0):
                encoded_patch = {}

                log.info(f"Slice {sliced}")
                images = {}
                data = {sat: {} for sat in satellites}
                for sat in satellites:
                    t_slice = [
                        f"2018-{months_folders[0]}-05",
                        f"2018-{months_folders[-1]}-30",
                    ]

                    if sat == "s2":
                        modality = "Sentinel2"
                        log.info(f"Slicing {modality}")
                        sliced_modality = get_sliced_modality(
                            months_folders,
                            folder_data,
                            sliced,
                            modality,
                            meteo=False,
                            dates=s2_dates,
                        )
                        log.info(f"Sliced {modality}")
                        sliced_modality = sliced_modality.sel(
                            t=slice(t_slice[0], t_slice[1])
                        )

                        # create_netcdf_s2_one_date(
                        #     xy_matrix, sliced_modality, margin, enum, date=2)

                        images, data, dates = get_s2(
                            s2_dates, t_slice, data, images, sliced_modality
                        )
                    else:
                        modality = (
                            "Sentinel1_ASCENDING"
                            if sat == "s1_asc"
                            else "Sentinel1_DESCENDING"
                        )
                        log.info(f"Slicing {modality}")
                        sliced_modality = get_sliced_modality(
                            months_folders,
                            folder_data,
                            sliced,
                            modality,
                            meteo=False,
                            dates=s1_dates_asc if sat == "s1_asc" else s1_dates_desc,
                        )
                        log.info(f"Sliced {modality}")
                        sliced_modality = sliced_modality.sel(
                            t=slice(t_slice[0], t_slice[1])
                        )

                        images, data, dates = get_s1(
                            t_slice, data, images, sliced_modality, sat
                        )
                        del sliced_modality

                    if sat not in dates_meteo:
                        dates_meteo[sat] = get_meteo_dates_modality(dates, sat)

                for modality in meteo_modalities:
                    log.info(modality)
                    sliced_meteo = get_sliced_modality(
                        months_folders,
                        folder_data,
                        sliced,
                        modality.upper(),
                        meteo=True,
                    )
                    for sat, dates_sat in dates_meteo.items():
                        # print(modality)
                        data[sat][f"meteo_{modality}"] = build_meteo(
                            images, sliced_meteo, dates_sat, modality
                        )[modality].unsqueeze(0)
                del images, sliced_meteo
                # DEM
                nc_file = os.path.join(folder_data, "DEM", "openEO.nc")
                dem_slice = get_one_slice(nc_file, sliced)
                dem_dataset = torch.tensor(dem_slice.DEM.values)
                # We deal with the special case,
                # when dem band contains 2 dates and one of them is nan
                # So we choose the date with less nans than the other
                if len(dem_dataset) > 0:
                    dem_dataset = dem_dataset[
                        torch.argmin(torch.isnan(dem_dataset).sum((1, 2)))
                    ]
                dem = dem_height_aspect(dem_dataset.squeeze(0))
                for sat in data:
                    data[sat]["dem"] = dem.unsqueeze(0)
                del dem
                batch_dict = {}
                for sat in data:
                    sits = (
                        MMDCDataStruct.init_empty()
                        .fill_empty_from_dict(data[sat])
                        .concat_meteo()
                    )
                    setattr(
                        sits,
                        "meteo",
                        torch.flatten(getattr(sits, "meteo"), start_dim=2, end_dim=3),
                    )
                    tcd_batch = PastisBatch(
                        sits=sits,
                        true_doy=dates_meteo[sat][f"date_{sat}"].values[None, :],
                        padd_val=torch.Tensor([]),
                        padd_index=torch.Tensor([]),
                    )
                    batch_dict[sat.upper()] = tcd_batch
                    encoded_patch[f"{sat}_doy"] = dates_meteo[sat][f"date_{sat}"].values
                    encoded_patch[f"{sat}_mask"] = sits.data.mask[
                        0, :, :, margin:-margin, margin:-margin
                    ]
                log.info("encoding")

                del data, tcd_batch

                (
                    latents1,
                    latents1_asc,
                    latents1_desc,
                    latents2,
                    mask_s1,
                    days_s1,
                ) = encode_one_batch(mmdc_model, batch_dict)
                encoded_patch["matrix"] = xy_matrix
                encoded_patch["s2_lat_mu"] = latents2.mean[
                    0, :, :, margin:-margin, margin:-margin
                ]
                encoded_patch["s2_lat_logvar"] = latents2.logvar[
                    0, :, :, margin:-margin, margin:-margin
                ]

                if s1_join:
                    encoded_patch["s1_lat_mu"] = latents1.mean[
                        0, :, :, margin:-margin, margin:-margin
                    ]
                    encoded_patch["s1_lat_logvar"] = latents1.logvar[
                        0, :, :, margin:-margin, margin:-margin
                    ]
                    encoded_patch["s1_mask"] = mask_s1[
                        0, :, :, margin:-margin, margin:-margin
                    ]
                    encoded_patch["s1_doy"] = days_s1
                else:
                    encoded_patch["s1_asc_lat_mu"] = latents1_asc.mean[
                        0, :, :, margin:-margin, margin:-margin
                    ]
                    encoded_patch["s1_asc_lat_logvar"] = latents1_asc.logvar[
                        0, :, :, margin:-margin, margin:-margin
                    ]
                    encoded_patch["s1_desc_lat_mu"] = latents1_desc.mean[
                        0, :, :, margin:-margin, margin:-margin
                    ]
                    encoded_patch["s1_desc_lat_logvar"] = latents1_desc.logvar[
                        0, :, :, margin:-margin, margin:-margin
                    ]

                # create_netcdf_encoded(xy_matrix,
                #                       encoded_patch["s2_lat_mu"].cpu().numpy(),
                #                       enum,
                #                       dates=dates_meteo["s2"][f"date_s2"].values)
                # create_netcdf_one_date_encoded(xy_matrix,
                #                                encoded_patch[
                #                                "s2_lat_mu"].cpu().numpy(),
                #                                enum,
                #                                date=2)
                # create_netcdf_encoded(xy_matrix,
                #                       encoded_patch["s1_lat_mu"].cpu().numpy(),
                #                       enum,
                #                       dates=dates_meteo["s1"][f"date_s1"].values,
                #                       s1=True)
                # create_netcdf_one_date_encoded(xy_matrix,
                #                                encoded_patch[
                #                                "s2_lat_mu"].cpu().numpy(),
                #                                enum,
                #                                date=2,
                #                                s1=True)

                if ind.size > 0 and tcd_values_sliced is not None:
                    for sat in satellites_to_write:
                        mask = encoded_patch[f"{sat}_mask"].expand_as(
                            encoded_patch[f"{sat}_lat_mu"]
                        )

                        encoded_patch[f"{sat}_lat_mu"][mask.bool()] = torch.nan
                        encoded_patch[f"{sat}_lat_logvar"][mask.bool()] = torch.nan

                        gt_ref_values_mu = rearrange_ts(encoded_patch[f"{sat}_lat_mu"])[
                            :, ind[:, 0], ind[:, 1]
                        ].T

                        gt_ref_values_logvar = rearrange_ts(
                            encoded_patch[f"{sat}_lat_logvar"]
                        )[:, ind[:, 0], ind[:, 1]].T
                        days = encoded_patch[f"{sat}_doy"]

                        values_df_mu = pd.DataFrame(
                            data=gt_ref_values_mu.cpu().numpy(),
                            columns=[
                                f"mu_f{b}_d{m}"
                                for m in range(len(days))
                                for b in range(
                                    encoded_patch[f"{sat}_lat_logvar"].shape[-3]
                                )
                            ],
                        )
                        values_df_logvar = pd.DataFrame(
                            data=gt_ref_values_logvar.cpu().numpy(),
                            columns=[
                                f"logvar_f{b}_d{m}"
                                for m in range(len(days))
                                for b in range(
                                    encoded_patch[f"{sat}_lat_logvar"].shape[-3]
                                )
                            ],
                        )

                        df_gt = pd.concat(
                            [
                                tcd_values_sliced.reset_index(),
                                values_df_mu,
                                values_df_logvar,
                            ],
                            axis=1,
                        )

                        if sat not in sliced_gt:
                            sliced_gt[sat] = [df_gt]
                        else:
                            sliced_gt[sat].append(df_gt)

                        df_gt.to_csv(
                            os.path.join(
                                output_folder, f"gt_tcd_t32tnt_encoded_{sat}_{enum}.csv"
                            )
                        )
                        if not Path(
                            os.path.join(output_folder, f"gt_tcd_t32tnt_days_{sat}.npy")
                        ).exists():
                            np.save(
                                os.path.join(
                                    output_folder, f"gt_tcd_t32tnt_days_{sat}.npy"
                                ),
                                days,
                            )

                # torch.save(encoded_patch,
                #            os.path.join(output_folder, f"Encoded_patch_{enum}.pt"))

                del (
                    latents1,
                    latents1_asc,
                    latents1_desc,
                    latents2,
                    encoded_patch,
                    batch_dict,
                )
    #
    # for sat in satellites_to_write:
    #     final_df = pd.concat(sliced_gt[sat], ignore_index=True)
    #     final_df.to_csv(os.path.join(output_folder,
    #     f"gt_tcd_t32tnt_encoded_{sat}.csv"))
