""" Constants describing MMDC data"""


S1_NOISE_LEVEL = 3e-05  # sigma0 value un linear scale considered to be noise

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

MODALITIES = [
    "s2",
    "s1_asc",
    "s1_desc",
    "dem",
    "dew_temp",
    "prec",
    "sol_rad",
    "temp_max",
    "temp_mean",
    "temp_min",
    "vap_press",
    "wind_speed",
]

MMDC_DATA = [
    "s2_set",
    "s2_masks",
    "s2_angles",
    "s1_set",
    "s1_valmasks",
    "s1_angles",
    "meteo_dew_temp",
    "meteo_prec",
    "meteo_sol_rad",
    "meteo_temp_max",
    "meteo_temp_mean",
    "meteo_temp_min",
    "meteo_vap_press",
    "meteo_wind_speed",
    "dem",
]

D_MODALITY = dict(
    zip(
        MODALITIES,
        ["Sentinel2", "Sentinel1_ASCENDING", "Sentinel1_DESCENDING", "DEM"]
        + AGERA_DATA,
    )
)

BANDS_S2 = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
ANGLES_S2 = ["sunAzimuthAngles", "sunZenithAngles", "viewAzimuthMean", "viewZenithMean"]
MASKS_S2 = ["CLM", "SCL"]

BANDS_S1 = ["VV", "VH"]
ANGLES_S1 = ["local_incidence_angle"]

METEO_BANDS = {
    "dew_temp": "dewpoint-temperature",
    "prec": "precipitation-flux",
    "sol_rad": "solar-radiation-flux",
    "temp_max": "temperature-max",
    "temp_mean": "temperature-mean",
    "temp_min": "temperature-min",
    "vap_press": "vapour-pressure",
    "wind_speed": "wind-speed",
}
