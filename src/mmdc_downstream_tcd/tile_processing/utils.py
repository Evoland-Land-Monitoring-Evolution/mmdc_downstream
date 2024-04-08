import numpy as np
import torch
import xarray as xr
from affine import Affine


def set_system_and_save(ds: xr.Dataset, xx: torch.Tensor, yy: torch.Tensor, name: str):
    ds.rio.write_transform(
        Affine(10.0, 0.0, xx.min() - 5, 0.0, -10.0, yy.max() + 5),
        grid_mapping_name="crs",
        inplace=True,
    )

    ds.rio.write_crs(32632, inplace=True)
    ds.to_netcdf(name, engine="h5netcdf")


def create_netcdf_s2_all_dates(
    xy_matrix: torch.Tensor, sliced_modality: xr.Dataset, margin: int, enum: int
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            B02=(
                ["t", "x", "y"],
                sliced_modality.B02.values[:, margin:-margin, margin:-margin],
            ),
            B03=(
                ["t", "x", "y"],
                sliced_modality.B03.values[:, margin:-margin, margin:-margin],
            ),
            B04=(
                ["t", "x", "y"],
                sliced_modality.B04.values[:, margin:-margin, margin:-margin],
            ),
            B08=(
                ["t", "x", "y"],
                sliced_modality.B08.values[:, margin:-margin, margin:-margin],
            ),
        ),
        coords={
            "x": ("x", xx[:, 0]),
            "y": ("y", yy[0]),
            "t": ("t", sliced_modality.t.values),
        },
        attrs=dict(description="Original image"),
    )

    set_system_and_save(ds, xx, yy, name=f"s2_patch_{enum}.nc")


def create_netcdf_s1_all_dates(
    xy_matrix: torch.Tensor,
    sliced_modality: xr.Dataset,
    margin: int,
    enum: int,
    orbit="asc",
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            VV=(
                ["t", "x", "y"],
                sliced_modality.VV.values[:, margin:-margin, margin:-margin],
            ),
            VH=(
                ["t", "x", "y"],
                sliced_modality.VH.values[:, margin:-margin, margin:-margin],
            ),
        ),
        coords={
            "x": ("x", xx[:, 0]),
            "y": ("y", yy[0]),
            "t": ("t", sliced_modality.t.values),
        },
        attrs=dict(description="Original image"),
    )

    set_system_and_save(ds, xx, yy, name=f"s1_{orbit}_patch_{enum}.nc")


def create_netcdf_s2_one_date(
    xy_matrix: torch.Tensor,
    sliced_modality: xr.Dataset,
    margin: int,
    enum: int,
    date: int,
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            B02=(
                ["x", "y"],
                sliced_modality.isel(t=date).B02.values[margin:-margin, margin:-margin],
            ),
            B03=(
                ["x", "y"],
                sliced_modality.isel(t=date).B03.values[margin:-margin, margin:-margin],
            ),
            B04=(
                ["x", "y"],
                sliced_modality.isel(t=date).B04.values[margin:-margin, margin:-margin],
            ),
            B08=(
                ["x", "y"],
                sliced_modality.isel(t=date).B08.values[margin:-margin, margin:-margin],
            ),
        ),
        coords={"x": ("x", xx[:, 0]), "y": ("y", yy[0])},
        attrs=dict(description="Original image"),
    )
    set_system_and_save(ds, xx, yy, name=f"s2_patch_{enum}_t{date}.nc")


def create_netcdf_s1_one_date(
    xy_matrix: torch.Tensor,
    sliced_modality: xr.Dataset,
    margin: int,
    enum: int,
    date: int,
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            B02=(
                ["x", "y"],
                sliced_modality.isel(t=date).B02.values[margin:-margin, margin:-margin],
            ),
            B03=(
                ["x", "y"],
                sliced_modality.isel(t=date).B03.values[margin:-margin, margin:-margin],
            ),
            B04=(
                ["x", "y"],
                sliced_modality.isel(t=date).B04.values[margin:-margin, margin:-margin],
            ),
            B08=(
                ["x", "y"],
                sliced_modality.isel(t=date).B08.values[margin:-margin, margin:-margin],
            ),
        ),
        coords={"x": ("x", xx[:, 0]), "y": ("y", yy[0])},
        attrs=dict(description="Original image"),
    )
    set_system_and_save(ds, xx, yy, name=f"s2_patch_{enum}_t{date}.nc")


def create_netcdf_encoded(
    xy_matrix: torch.Tensor,
    s2_mu: np.ndarray,
    enum: int,
    dates: np.ndarray,
    s1: bool = False,
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            f0=(["t", "x", "y"], s2_mu[:, 0]),
            f1=(["t", "x", "y"], s2_mu[:, 1]),
            f2=(["t", "x", "y"], s2_mu[:, 2]),
            f3=(["t", "x", "y"], s2_mu[:, 3]),
            f4=(["t", "x", "y"], s2_mu[:, 4]),
            f5=(["t", "x", "y"], s2_mu[:, 5]),
        ),
        coords={"x": ("x", xx[:, 0]), "y": ("y", yy[0]), "t": ("t", dates)},
        attrs=dict(description="Encoded image"),
    )
    if not s1:
        set_system_and_save(ds, xx, yy, name=f"s2_patch_encoded_{enum}.nc")
    else:
        set_system_and_save(ds, xx, yy, name=f"s1_patch_encoded_{enum}.nc")


def create_netcdf_one_date_encoded(
    xy_matrix: torch.Tensor, s2_mu: np.ndarray, enum: int, date: int, s1: bool = False
):
    xx = xy_matrix[0]
    yy = xy_matrix[1]
    ds = xr.Dataset(
        data_vars=dict(
            f0=(["x", "y"], s2_mu[date, 0]),
            f1=(["x", "y"], s2_mu[date, 1]),
            f2=(["x", "y"], s2_mu[date, 2]),
            f3=(["x", "y"], s2_mu[date, 3]),
            f4=(["x", "y"], s2_mu[date, 4]),
            f5=(["x", "y"], s2_mu[date, 5]),
        ),
        coords={"x": ("x", xx[:, 0]), "y": ("y", yy[0])},
        attrs=dict(description="Encoded image one date"),
    )
    if not s1:
        set_system_and_save(ds, xx, yy, name=f"s2_patch_encoded_{enum}_t{date}.nc")
    else:
        set_system_and_save(ds, xx, yy, name=f"s1_patch_encoded_{enum}_t{date}.nc")
