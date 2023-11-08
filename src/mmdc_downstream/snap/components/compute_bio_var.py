"""Script to compute bio-physical variables"""
import os
from collections import namedtuple
from typing import Literal

import matplotlib.pyplot as plt
import torch

from src.mmdc_downstream.snap.lai_snap import BVNET

DatasetPaths = namedtuple("DatasetPaths", ["input_path", "output_path"])


def compute_variables(paths: DatasetPaths,
                      tiles: list[str],
                      variables: list[Literal["lai", "ccc", "cab", "cwc",
                                              "cw"]],
                      ver: Literal["2", "3A", "3B"] = "2") -> None:
    """
    General function to compute each bio-physical variable
    """
    for var in variables:

        # We instantiate snap model for each variable
        model = BVNET(ver=ver, variable=var)
        model.set_snap_weights()

        for tile in tiles:
            tile_path = os.path.join(paths.input_path, tile)
            if os.path.exists(tile_path):
                files_s2_set = [
                    file for file in os.listdir(tile_path) if "s2_set" in file
                ]
                for file_s2 in files_s2_set:
                    output_set = process_file(tile_path, file_s2, model)
                    # torch.save(
                    #     output_set,
                    #     os.path.join(os.path.join(paths.output_path, tile),
                    #                  file_s2.replace("set", model.variable)))


def process_file(tile_path: str, file_s2: str, model: BVNET) -> torch.Tensor:
    """
    Compute variable for each individual roi file from selected tiles
    """
    s2_set = torch.load(os.path.join(tile_path, file_s2))
    s2_angles = torch.load(
        os.path.join(tile_path, file_s2.replace("set", "angles")))

    input_set = prepare_image(s2_set, s2_angles)
    output_set = model.forward(input_set)
    output_set = process_output(
        output_set,
        torch.Size([s2_set.size(0),
                    s2_set.size(2),
                    s2_set.size(3)]))

    # plt.close()
    # fig = plt.figure(figsize=(20, 10))
    # ax1 = plt.subplot2grid((1, 2), (0, 0))
    # ax1.imshow(output_set[0].to(int).detach().numpy(), cmap='gray')
    # ax2 = plt.subplot2grid((1, 2), (0, 1))
    # ax2.imshow(s2_set[0, [7, 2, 1]].permute(1, 2, 0).detach().numpy() /
    #            s2_set[0, [7, 2, 1]].reshape(3, -1).max(1).values)
    # print(file_s2.split('.')[0])
    # fig.savefig(file_s2.split('.')[0] + ".png")

    return output_set


def prepare_image(
    s2_set: torch.Tensor,
    s2_angles: torch.Tensor,
) -> torch.Tensor:
    """
    Prepare input data for BVNET model
    S2 bands that we select :
    B03, B04, B05, B06, B07, B8A, B11, B12

    S2 bands we have in the image set:
    B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12

    S2 angles we select:
    cos Z_s2, cos Z_sun, cos(A_sun-A_s2)

    S2 angles we have in the set:
    cos Z_sun, cos A_sun, sin A_sun, cos Z_s2, cos A_s2, sin A_s2

    We compute cos(A_sun-A_s2) using the formula:
    cos(A-B) = cosA*cosB - sinA*sinB
    """

    s2_set_sel = s2_set[:, [1, 2, 3, 4, 5, 7, 8, 9]]
    s2_angles_sel = s2_angles[:, [3, 0]]

    # Compute cos(A_sun-A_s2)
    cos_diff = s2_angles[:, 1] * s2_angles[:, 4] - \
               s2_angles[:, 2] * s2_angles[:, 5]

    input_set = torch.cat((s2_set_sel, s2_angles_sel, cos_diff[:, None, :, :]),
                          1)

    # We transform to get input shape (B, 11)
    return input_set.permute(0, 2, 3, 1).reshape(-1, 11)


def process_output(output_set: torch.Tensor,
                   output_size: torch.Size) -> torch.Tensor:
    """
    Transforms output data from shape (B,) to (B, W, H)
    """
    return output_set.reshape(output_size)
