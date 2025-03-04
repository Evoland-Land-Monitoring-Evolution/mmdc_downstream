import json
import os
import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import openeo
import pyproj
import shapely

warnings.filterwarnings("ignore")

connection = openeo.connect("https://openeo.vito.be/openeo/1.2").authenticate_oidc()

launched_jobs = [job["title"] for job in connection.list_jobs()]

# Copy json to output folder
job_options = {
    "executor-memory": "10G",
    "executor-memoryOverhead": "16G",  # default 2G
    "executor-cores": 2,
    "task-cpus": 1,
    "executor-request-cores": "400m",
    "max-executors": "100",
    "driver-memory": "16G",
    "driver-memoryOverhead": "16G",
    "driver-cores": 5,
    "udf-dependency-archives": [],
    "logging-threshold": "info",
}


folder = "/home/kalinichevae/Documents/BioMass/Sweden_NFI"
output_folder = "/home/kalinichevae/scratch_data/BioMass/images/Sweden_all"

assert Path(output_folder).exists()

tiles_df = gpd.read_file(os.path.join(folder, "S2_tiles_32633.gpkg")).explode(
    ignore_index=True
)
nfi_year = [f for f in os.listdir(folder) if f.endswith("_bb_dissolved_bb.gpkg")]

tiles_df = tiles_df[~tiles_df["Name"].isin(["32VPJ"])].reset_index(drop=True)

for f in np.sort(nfi_year):
    bb_df = gpd.read_file(os.path.join(folder, f))
    year = int(re.match(r".*([1-3][0-9]{3})", f).group(1))
    # we put here bb indices that are already attributed to a tile,
    # so it is not used twice
    used_idx = []
    if year > 2014:
        for i in range(len(tiles_df)):
            tile_name = tiles_df["Name"].values[i]
            print(
                Path(
                    os.path.join(
                        output_folder, str(year), tile_name, "job-results.json"
                    )
                ).exists()
            )
            if (
                f"Sweden_S1_ASC_{tile_name}_{year}" not in launched_jobs
                and not Path(
                    os.path.join(
                        output_folder, str(year), tile_name, "job-results.json"
                    )
                ).exists()
            ):
                sel_idx = [
                    t
                    for t in range(len(bb_df))
                    if (
                        tiles_df.iloc[i]
                        .geometry.contains(bb_df.iloc[[t], :].geometry)
                        .sum()
                        > 0
                        and t not in used_idx
                    )
                ]
                used_idx.extend(sel_idx)
                if sel_idx:
                    print(tile_name, year)

                    selected_bb = (
                        bb_df.copy()
                        .iloc[sel_idx, :]
                        .reset_index(drop=True)
                        .drop(
                            columns=[
                                "ID",
                                "TractID",
                                "PlotID",
                                "Inventory_Date",
                                "Easting",
                                "Northing",
                                "PlotArea",
                                "LandUseClass",
                                "Treelayers",
                                "MeanHeight",
                                "StandAge",
                                "Stand_density",
                                "VolPine",
                                "VolContorta",
                                "VolSpruce",
                                "VolBirch",
                                "VolOtherDec",
                                "AGB+foliage_t/ha",
                                "AGB_t/ha",
                                "width",
                                "height",
                                "area",
                                "perimeter",
                            ]
                        )
                    )
                    selected_bb["Tile"] = tile_name

                    selected_bb_name = os.path.join(
                        folder,
                        "FEATURES",
                        f"Sweden_{tile_name}_{year}.geojson",
                    )

                    transformer = pyproj.Transformer.from_crs(
                        selected_bb.crs, 4326, always_xy=True
                    )

                    selected_bb_wgs84 = [
                        shapely.ops.transform(transformer.transform, b)
                        for b in selected_bb.geometry
                    ]

                    selected_bb = gpd.GeoDataFrame(
                        {
                            "ID": range(len(sel_idx)),
                            "year": year,
                            "geometry": selected_bb_wgs84,
                            "tile": tile_name,
                        },
                        crs="epsg:4326",
                    )

                    selected_bb.to_file(
                        selected_bb_name,
                        driver="GeoJSON",
                        crs="epsg:4326",
                        index=False,
                    )

                    with open(selected_bb_name) as feat:
                        features = json.load(feat)

                    orbit = "ASCENDING"

                    properties = {"sat:orbit_state": lambda od: od == orbit}
                    sentinel1_cube = (
                        connection.load_collection(
                            "SENTINEL1_GRD",
                            temporal_extent=[f"{year}-01-01", f"{year + 1}-01-01"],
                            bands=["VV", "VH"],
                            properties=properties,
                        )
                        .sar_backscatter(
                            coefficient="gamma0-terrain",
                            elevation_model=None,
                            mask=False,
                            contributing_area=False,
                            local_incidence_angle=True,
                            ellipsoid_incidence_angle=False,
                            noise_removal=True,
                            options=None,
                        )
                        .filter_spatial(features)
                    )

                    s1_download_job = sentinel1_cube.create_job(
                        title=f"Sweden_S1_ASC_{tile_name}_{year}",
                        job_options=job_options,
                        out_format="netCDF",
                        sample_by_feature=True,
                    )
                    s1_download_job.start_job()
                    print(f"Sent job {s1_download_job.job_id}")
