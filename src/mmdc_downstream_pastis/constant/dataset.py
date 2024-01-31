S2_BAND = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

DROP_VARIABLE = [
    "sunAzimuthAngles",
    "sunZenithAngles",
    "viewAzimuthMean",
    "viewZenithMean",
    "dataMask",
]
CLD_MASK = ["CLM", "SCL"]
LOAD_VARIABLE = S2_BAND + CLD_MASK
DATA_MASK = "dataMask"
ENCODE_PERMUTATED = -5
VAL_SEED = 5
AGERA_DATA = [
    "DEW_TEMP",
    "PREC",
    "SOL_RAD",
    "TEMP_MAX",
    "TEMP_MEAN",
    "TEMP_MIN",
    "VAP_PRESS",
    "WIND_SPEED",
]
SIMPLE_MOD_MMDC = ["s2", "s1_asc", "s1_desc", "dem"]
L_MOD_MMDC = SIMPLE_MOD_MMDC + [
    "dew_temp",
    "prec",
    "sol_rad",
    "temp_max",
    "temp_mean",
    "temp_min",
    "val_press",
    "wind_speed",
]
MERGED_MODALITIES = SIMPLE_MOD_MMDC + ["agera5"]
D_MODALITY = dict(
    zip(
        L_MOD_MMDC,
        ["Sentinel2", "Sentinel1_ASCENDING", "Sentinel1_DESCENDING", "DEM"]
        + AGERA_DATA,
    )
)

S2_TILE = ["31TEK"]
FORMAT_SITS = "clip.nc"
PRETRAIN_BEG = "2017-01-01"
PRETRAIN_END = "2020-12-31"
