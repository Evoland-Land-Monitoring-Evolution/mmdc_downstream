"""Script to compute NDVI per ROI to select tiles for LAI model training"""
import os
from collections import namedtuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from einops import rearrange

DatasetPaths = namedtuple("DatasetPaths", ["input_path", "output_path"])


def compute_variables(paths: DatasetPaths, tiles: list[str]) -> None:
    """
    General function to compute NDVI stats
    """

    df_all: pd.DataFrame = pd.DataFrame()
    for tile in tiles:
        print("Computing Tile ", tile)
        tile_path = os.path.join(paths.input_path, tile)
        if os.path.exists(tile_path):
            files_s2_set = [
                file for file in os.listdir(tile_path) if "s2_set" in file
            ]
            for file_s2_roi in files_s2_set:
                df_roi = process_file(tile_path, file_s2_roi)
                df_roi["roi"] = int(file_s2_roi.split('.')[0].split('_')[-1])
                df_roi['tile'] = tile

                if df_all is None:
                    df_all = df_roi
                else:
                    df_all = pd.concat([df_all, df_roi], ignore_index=True)

    print(df_all)

    df_all_roi = df_all.groupby(
        ['tile', 'roi']).mean().drop(columns=['patch']).reset_index()

    df_all.to_csv(os.path.join(paths.input_path, "NDVI_stats_patch.csv"))
    df_all_roi.to_csv(os.path.join(paths.input_path, "NDVI_stats_roi.csv"))


def process_temporal_pixels(temp_dict, patch_dict):
    """Compute statistics over statistics on temporal ndvi pixel values"""
    for k, v in temp_dict.items():  # pylint: : disable=C0103

        min_patch, median_patch, max_patch = v.quantile(
            torch.Tensor([0.05, 0.5, 0.95]))

        patch_dict[k + "_min"] = min_patch.item()
        patch_dict[k + "_median"] = median_patch.item()
        patch_dict[k + "_max"] = max_patch.item()
    return patch_dict

def process_patch(patch: str | int,
                  global_df: pd.DataFrame,
                  roi_df: pd.DataFrame,
                  ndvi_set: torch.Tensor) -> pd.DataFrame:
    """Compute temporal NDVI stats for one patch"""
    idx = roi_df.index[roi_df["patch_id"] == patch].to_list()

    ndvi_patch = ndvi_set[idx]

    # Per pixel
    # temp stands for temporal
    ndvi_patch_temp = rearrange(ndvi_patch, 't h w -> t (h w)')

    # Compute temporal stats per pixel
    min_temp, median_temp, max_temp = ndvi_patch_temp.nanquantile(
        torch.Tensor([0.05, 0.5, 0.95]), dim=0)

    temp_dict = {
        "min_temp": min_temp,
        "median_temp": median_temp,
        "max_temp": max_temp
    }

    patch_dict = {'patch': patch}
    patch_dict = process_temporal_pixels(temp_dict, patch_dict)

    if len(global_df) == 0:
        global_df = pd.DataFrame(patch_dict, index=[0])
    else:
        global_df = pd.concat(
            [global_df, pd.DataFrame(patch_dict, index=[0])],
            ignore_index=True)

    # mean_temp = ndvi_patch_temp.nanmean(0)
    # median_temp = ndvi_patch_temp.nanmedian(0).values
    # max_temp = ndvi_patch_temp.nan_to_num(-10).max(0).values
    # min_temp = ndvi_patch_temp.nan_to_num(10).min(0).values
    return global_df


def process_file(
    tile_path: str,
    file_s2: str,
) -> pd.DataFrame:
    """
    Compute variable for each individual roi file from selected tiles
    """
    s2_set = torch.load(os.path.join(tile_path, file_s2))

    s2_mask = torch.load(
        os.path.join(tile_path, file_s2.replace("set", "masks"))).to(int)

    ndvi_set = compute_ndvi(s2_set, s2_mask)

    roi_df = pd.read_csv(os.path.join(  # pylint: : disable=C0103
        tile_path,
        file_s2.replace("s2_set", "dataset").replace("pth", "csv")),
                     sep='\t')

    global_df = {}

    for patch in roi_df["patch_id"].unique():
        global_df = process_patch(patch, global_df, roi_df, ndvi_set)

    return global_df


def compute_ndvi(s2_set: torch.Tensor, s2_mask: torch.Tensor) -> torch.Tensor:
    """
    Prepare input data (concat S2 bands and angles) ->
    produce LAI with snap ->
    reshape output and set masked data to nan
    """
    ndvi = (s2_set[:, 6] - s2_set[:, 2]) / (s2_set[:, 6] + s2_set[:, 2])
    ndvi[s2_mask.squeeze(1).bool()] = torch.nan

    return ndvi


def visualize_lai_gt(
    output_set: torch.Tensor,
    s2_set: torch.Tensor,
    file_s2: str,
) -> None:
    """Visualize LAI next to S2 image"""
    plt.close()
    fig = plt.figure(figsize=(20, 20))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.imshow(output_set[0].detach().numpy() / 15, cmap='gray')
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.imshow(s2_set[0, [7, 2, 1]].permute(1, 2, 0).detach().numpy() /
               s2_set[0, [7, 2, 1]].reshape(3, -1).max(1).values)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ax3.hist(output_set[0].detach().numpy(), bins=20)
    print(file_s2.split('.')[0])
    fig.savefig(file_s2.split('.')[0] + ".png")
