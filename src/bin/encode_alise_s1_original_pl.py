# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from hydra.utils import instantiate
from mmmv_ssl.data.datamodule.mm_datamodule import MMMaskDataModule
from mmmv_ssl.model.sits_encoder import MonoSITSEncoder
from mmmv_ssl.model.utils import build_encoder
from mmmv_ssl.module.alise_mm import AliseMM
from mt_ssl.data.classif_class import ClassifBInput
from mt_ssl.utils.open import open_yaml
from omegaconf import DictConfig
from openeo_mmdc.dataset.to_tensor import load_transform_one_mod
from torch.nn import functional as F

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK = "/work/scratch/data/kalinie"


def load_checkpoint(hydra_conf, pretrain_module_ckpt_path, mod):
    pretrain_module_config = DictConfig(open_yaml(hydra_conf))
    myconfig_module = DictConfig(
        open_yaml(f"{WORK}/results/alise_preentrained/fine_tune_one_mod.yaml")
    )
    myconfig_module.mod = "s2" if mod == "S2" else "s1"

    pretrain_module_config.datamodule.datamodule.path_dir_csv = (
        f"{WORK}/results/alise_preentrained/data/"
    )
    old_datamodule: MMMaskDataModule = instantiate(
        pretrain_module_config.datamodule.datamodule,
        config_dataset=pretrain_module_config.dataset,
        batch_size=pretrain_module_config.train.batch_size,
        _recursive_=False,
    )
    pretrained_pl_module: AliseMM = instantiate(
        pretrain_module_config.module,
        train_config=pretrain_module_config.train,
        input_channels=old_datamodule.num_channels,
        stats=(
            old_datamodule.all_transform.s2.stats,
            old_datamodule.all_transform.s1_asc.stats,
        ),  # TODO do better than that load stats of each mod
        _recursive_=False,
    )

    pretrained_module = pretrained_pl_module.load_from_checkpoint(
        pretrain_module_ckpt_path
    )

    repr_encoder: MonoSITSEncoder = build_encoder(
        pretrained_module=pretrained_module, mod=myconfig_module.mod
    )

    repr_encoder.eval()

    return repr_encoder


# def load_checkpoint(hydra_conf, pretrain_module_ckpt_path, mod):
#     path = f"/work/scratch/data/kalinie/results/alise_preentrained/malice_{mod}.pth"
#     return torch.load(path)


def back_to_date(
    days_int: torch.Tensor, ref_date: str = "2018-09-01"
) -> list[np.array]:
    """
    Go back from integral doy (number of days from ref date)
    to calendar doy
    """
    return [
        np.asarray(
            pd.to_datetime(
                [
                    pd.Timedelta(dd, "d") + pd.to_datetime(ref_date)
                    for dd in days_int.cpu().numpy()[d]
                ]
            )
        )
        for d in range(len(days_int))
    ]


def apply_padding(allow_padd, max_len, t, sits, doy):
    if allow_padd:
        padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
        padd_doy = (0, max_len - t)
        sits = F.pad(sits, padd_tensor)
        doy = F.pad(doy, padd_doy)
        padd_index = torch.zeros(max_len)
        padd_index[t:] = 1
        padd_index = padd_index.bool()
    else:
        padd_index = None

    return sits, doy, padd_index


def encode_series_alise_s1(
    path_alise_model: str | Path,
    path_csv: str | Path,
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sat: str,
    output_path: str | Path,
    postfix: str,
):
    transform = load_transform_one_mod(path_csv, mod=sat.lower()).transform

    """Encode PASTIS SITS S2 with prosailVAE"""

    log.info("Loader is ready")

    hydra_conf = os.path.join(
        os.path.dirname(path_alise_model), ".hydra", "config.yaml"
    )
    if not Path(hydra_conf).exists():
        hydra_conf = path_alise_model.split(".")[0] + "_config.yaml"

    repr_encoder = load_checkpoint(hydra_conf, path_alise_model, sat.lower())
    repr_encoder = repr_encoder.to(DEVICE)

    for patch_path in np.sort(os.listdir(os.path.join(dataset_path_oe, sat))):
        batch = torch.load(
            os.path.join(dataset_path_oe, sat, patch_path), map_location=DEVICE
        )

        id_patch = patch_path.split("_")[-1].split(".")[0]
        log.info(id_patch)
        img = batch.sits  # [:, :, 32:-32, 32:-32]

        mask = batch.mask  # [:, :, 32:-32, 32:-32]

        true_doy = batch.dates_dt.values[:, 0]

        reference_date = "2014-03-03"

        doy = torch.Tensor(
            (pd.to_datetime(true_doy) - pd.to_datetime(reference_date)).days
        )
        print(len(doy))
        print(doy)

        if sat == "S1_ASC":
            s1_asc_2b = img[:, [1, 0]]

            ratio = s1_asc_2b[:, 0] / s1_asc_2b[:, 1]

            s1_asc = torch.cat([s1_asc_2b, ratio[:, None, :, :]], dim=1)

            s1_asc = torch.log(s1_asc)

            s1_asc[mask.expand_as(s1_asc).bool()] = 0

            prepared_sits = s1_asc
        else:  # S2
            prepared_sits = img

        prepared_sits, doy, padd_index = apply_padding(
            allow_padd=True,
            max_len=70 if sat == "S2" else 100,
            t=img.shape[0],
            sits=prepared_sits,
            doy=doy,
        )

        # Prepare patch
        ref_norm = rearrange(
            transform(rearrange(prepared_sits, "t c h w -> c t h w")),
            "c t h w -> 1 t c h w",
        )

        input = ClassifBInput(
            sits=ref_norm.to(DEVICE),
            input_doy=doy[None, :].to(DEVICE),
            padd_index=padd_index[None, :].to(DEVICE),
            mask=None,
            labels=None,
        )
        repr_encoder.eval()
        with torch.no_grad():
            ort_out = repr_encoder.forward_keep_input_dim(input).repr

        # for i in range(ort_out[0].shape[0]):
        #     for j in range(ort_out[0].shape[1]):
        #         print(np.nanquantile(ort_out[0, i, j].cpu(), q=[0.01, 0.5, 0.99]))

        torch.save(
            ort_out[0],
            os.path.join(output_path, sat + "_" + postfix, f"{sat}_{id_patch}.pt"),
        )
