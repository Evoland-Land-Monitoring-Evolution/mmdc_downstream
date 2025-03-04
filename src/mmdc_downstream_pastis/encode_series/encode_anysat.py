# flake8: noqa

#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import json
import logging
import ssl
from datetime import datetime

import numpy as np
import torch

ssl._create_default_https_context = ssl._create_stdlib_context

log = logging.getLogger(__name__)

# SATS = ["S2", "S1_ASC"]
SATS = ["S1_ASC"]

model = torch.hub.load(
    "gastruc/anysat",
    "anysat",
    pretrained=True,
    force_reload=False,
    flash_attn=False,
    device="cuda",
)


folder = "/work/CESBIO/projects/DeepChange/Ekaterina/Pastis-R"
output_path = "/work/scratch/data/kalinie/results/Pastis_encoded/anysat"

with open(folder + "/NORM_S1A_patch.json") as f:
    norm_s1 = json.load(f)["Fold_1"]
    MEAN_S1 = torch.tensor(norm_s1["mean"]).float()
    STD_S1 = torch.tensor(norm_s1["std"]).float()
print(norm_s1)

with open(folder + "/NORM_S2_patch.json") as f:
    norm_s2 = json.load(f)["Fold_1"]
    MEAN_S2 = torch.tensor(norm_s2["mean"]).float()
    STD_S2 = torch.tensor(norm_s2["std"]).float()
print(norm_s2)


def day_number_in_year_pastis(date_arr):
    day_number = []
    for date_string in date_arr:
        date_object = datetime.strptime(str(date_string), "%Y%m%d")
        day_number.append(date_object.timetuple().tm_yday)  # Get the day of the year
    return torch.tensor(day_number)


with open(folder + "/metadata.geojson") as f:
    meta = json.load(f)


for i in range(len(meta["features"])):
    patch_id = meta["features"][i]["id"]
    # if not Path(os.path.join(output_path, "S1_ASC", f"S1_ASC_{patch_id}.pt")).exists():
    if True:
        print(f"Patch {patch_id}")

        if "S1_ASC" in SATS:
            s1 = torch.FloatTensor(np.load(folder + f"/DATA_S1A/S1A_{patch_id}.npy"))
            s1_dates = day_number_in_year_pastis(
                list(meta["features"][i]["properties"]["dates-S1A"].values())
            )
            s1 = (s1 - MEAN_S1[:, None, None]) / STD_S1[:, None, None]
        if "S2" in SATS:
            s2 = torch.FloatTensor(np.load(folder + f"/DATA_S2/S2_{patch_id}.npy"))
            s2_dates = day_number_in_year_pastis(
                list(meta["features"][i]["properties"]["dates-S2"].values())
            )
            s2 = (s2 - MEAN_S2[:, None, None]) / STD_S2[:, None, None]

        final_encoded = None
        size = 32
        for xx in range(0, 128, size):
            for yy in range(0, 128, size):
                data = {}
                if "S2" in SATS:
                    data = {
                        "s2": s2[..., yy : yy + size, xx : xx + size]
                        .unsqueeze(0)
                        .to("cuda"),
                        "s2_dates": s2_dates.unsqueeze(0).to("cuda"),
                    }
                if "S1_ASC" in SATS:
                    data = {
                        "s1": s1[..., yy : yy + size, xx : xx + size]
                        .unsqueeze(0)
                        .to("cuda"),
                        "s1_dates": s1_dates.unsqueeze(0).to("cuda"),
                    }
                with torch.no_grad():
                    features = model(
                        data,
                        patch_size=20,
                        output="dense",
                        output_modality="s2" if SATS[0] == "S2" else "s1",
                    )[0].permute(2, 0, 1)
                    # features_ = model(data, patch_size=10, output='patch')[0].permute(2, 0, 1)
                    # print(features[:768] - features_)
                    # exit()

                if final_encoded is None:
                    final_encoded = torch.zeros((features.shape[0], 128, 128))

                final_encoded[
                    :, yy : yy + size, xx : xx + size
                ] = features.cpu().detach()

        import matplotlib.pyplot as plt

        plt.rcParams["backend"]
        plt.close()
        rows, cols = 16, 16

        f, axarr = plt.subplots(
            nrows=rows, ncols=cols, subplot_kw={"xticks": [], "yticks": []}
        )

        vmin, vmax = np.percentile(final_encoded, q=[2, 98], axis=(-2, -1))
        vmin, vmax = vmin[:, None, None], vmax[:, None, None]
        final_encoded_norm = (
            (final_encoded.clip(torch.Tensor(vmin), torch.Tensor(vmax)) - vmin)
            / (vmax - vmin)
        ).clip(0, 1)
        for k in range(rows):
            for m in range(cols):
                print(k, m)
                to_show = final_encoded_norm[
                    768 + (k * cols + m) * 3 : 768 + (k * cols + m) * 3 + 3
                ]
                axarr[k, m].imshow(to_show.permute(1, 2, 0).numpy())
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        plt.savefig(f"{SATS[0]}_emb_solo.pdf", format="pdf", dpi=300)
        plt.show()

        exit()

        # torch.save(
        #     final_encoded,
        #     os.path.join(output_path, "S1_ASC", f"S1_ASC_{patch_id}.pt"),
        # )
