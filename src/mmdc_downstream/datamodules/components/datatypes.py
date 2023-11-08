""" Types for the datamodule components"""

from collections import namedtuple

DFRow = namedtuple("DFRow", ["df", "row"])
S1Config = namedtuple(
    "S1Config", ["missing_asc", "asc_name", "missing_desc", "desc_name"]
)

PatchSize = namedtuple("PatchSize", ["size", "margin"])
PatchCreationParams = namedtuple("PatchCreationParams", ["size", "margin", "threshold"])

DFRowData = namedtuple("DFRowData", ["s1_config", "name_s2", "name_srtm"])

PatchFilterConditions = namedtuple(
    "PatchFilterConditions",
    [
        "keep_s2",
        "keep_s1asc",
        "keep_s1desc",
        "keep_dem",
        "keep_meteo",
        "s2_keep_count",
        "s1asc_keep_count",
        "s1desc_keep_count",
        "dem_keep_count",
        "meteo_keep_count",
    ],
)

KeepCounts = namedtuple(
    "KeepCounts",
    [
        "s2_keep_count",
        "s1asc_keep_count",
        "s1desc_keep_count",
        "dem_keep_count",
        "meteo_keep_count",
    ],
)

PatchesToFilter = namedtuple("PatchesToFilter", ["s2m_p", "s1m_p", "dem_p", "meteo_p"])
PatchConfig = namedtuple(
    "PatchConfig", ["roi", "tile", "patch_size", "patch_margin", "threshold"]
)

DatasetPaths = namedtuple("DatasetPaths", ["samples_dir", "export_path"])

SeriesDays = namedtuple(
    "SeriesDays",
    [
        "days_gap",
        "meteo_days_before",
        "meteo_days_after",
        "start_date_series",
        "end_date_series",
    ],
)

MatchParams = namedtuple("MatchParams", "s1_asc_desc_strict prefilter_clouds only_csv")
