import os

import xarray as xr

folder_data = "/work/scratch/data/kalinie/TCD/OpenEO_data/t32tnt/"
month = "4"
nc_file = os.path.join(folder_data, month, "Sentinel1_ASCENDING", "openEO.nc")

file = xr.open_dataset(
    nc_file,
    decode_coords="all",
    mask_and_scale=False,
    decode_times=True,
    engine="h5netcdf",
)

file.isel(t=0).to_netcdf(os.path.join(folder_data, "S1_4_1.nc"))
