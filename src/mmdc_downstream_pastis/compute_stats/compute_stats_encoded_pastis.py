"""Compute stats for encoded pastis dataset"""

import logging
import os

import numpy as np
import torch
from einops import rearrange

model = "checkpoint_best"
dataset_path_oe = f"{os.environ['WORK']}/results/Pastis_encoded"
feat_nb = 6
sats = ["S1", "S2"]

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_stats(sum, sum_x2):
    mu = sum / count
    mu_of_x2 = sum_x2 / count
    std = torch.sqrt(mu_of_x2 - mu**2)
    return mu, std


for sat in sats:
    log.info(f"Starting process satellite {sat}")
    dataset_path_oe_sat = os.path.join(dataset_path_oe, model, sat)
    images = os.listdir(dataset_path_oe_sat)
    count = 0
    sum_of_mu = torch.zeros(feat_nb)
    sum_of_logvar = torch.zeros(feat_nb)
    sum_of_mu2 = torch.zeros(feat_nb)
    sum_of_logvar2 = torch.zeros(feat_nb)
    for im in np.sort(images):
        print(im)
        log.info(im)
        sits = torch.load(
            os.path.join(dataset_path_oe_sat, im), map_location="cpu"
        )  # T x C x H x W
        img = sits["latents"]
        mask = sits["mask"]
        mean = rearrange(img.mean, "t c h w -> c (h w t)")
        logvar = rearrange(img.logvar, "t c h w -> c (h w t)")
        mask = rearrange(mask, "t c h w -> (c h w t)")

        for feat in range(feat_nb):
            mean_feat = mean[feat][~mask.bool()]
            logvar_feat = logvar[feat][~mask.bool()]
            if feat == 0:
                count += len(mean_feat)
            sum_of_mu[feat] += mean_feat.sum()
            sum_of_logvar[feat] += logvar_feat.sum()
            sum_of_mu2[feat] += (mean_feat**2).sum()
            sum_of_logvar2[feat] += (logvar_feat**2).sum()
    # compute_stats(sum_of_mu, sum_of_mu2)
    # compute_stats(sum_of_logvar, sum_of_logvar2)
    torch.save(
        compute_stats(sum_of_mu, sum_of_mu2),
        os.path.join(dataset_path_oe, model, f"stats_mu_{sat}.pt"),
    )
    torch.save(
        compute_stats(sum_of_logvar, sum_of_logvar2),
        os.path.join(dataset_path_oe, model, f"stats_logvar_{sat}.pt"),
    )
