import matplotlib.pyplot as plt
import numpy as np
import rasterio
from encode_tile import get_tcd_gt
from sklearn.metrics import mean_squared_error

gt_path = "/work/scratch/data/kalinie/TCD/OpenEO_data/SAMPLES_t32tnt_32632.geojson"
gt_path = "/work/scratch/data/kalinie/TCD/OpenEO_data/SAMPLES_t32tnt.geojson"

their_results = "/work/scratch/data/kalinie/TCD/OpenEO_data/their_tcd_t32tnt.tif"
their_results = (
    "/work/scratch/data/kalinie/TCD/TDC-VLCC-2018_raw/Central_EU_TCD_VLCC_2018_LAEA.tif"
)

# TCD_2018_010m_eu_03035_V2_0_Central_EU

src = rasterio.open(their_results)
img = src.read()[0]
df_gt = get_tcd_gt(gt_path)

rows, cols = src.shape

x_left, y_top = src.bounds[0], src.bounds[3]

transform = src.get_transform()

xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = -transform[5]

print(transform)
points_list = df_gt[["x", "y"]].values  # list of X,Y coordinates
# df_gt = df_gt[(df_gt['x'] >= xy_matrix[0].min()) &
#                                (df_gt['x'] <= xy_matrix[0].max()) &
#                                (df_gt['y'] >= xy_matrix[1].min()) &
#                                (df_gt['y'] <= xy_matrix[1].max())]


col = ((df_gt["x"].values - xOrigin) / pixelWidth).astype(int)
row = ((yOrigin - df_gt["y"].values) / pixelHeight).astype(int)
# coords = np.array([col, row])
pred = img[row, col].clip(0, 100)
print(pred)

plt.scatter(pred, df_gt["TCD"].values, s=1, c="red", alpha=0.5)
plt.plot(
    [df_gt["TCD"].values.min(), df_gt["TCD"].values.max()],
    [df_gt["TCD"].values.min(), df_gt["TCD"].values.max()],
)
plt.xlabel("pred")
plt.ylabel("gt")
plt.savefig("their_plot.png")

mse = mean_squared_error(df_gt["TCD"].values, np.array(pred), squared=False)
print(mse)
