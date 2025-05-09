# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from hydra.utils import instantiate
from mmmv_ssl.data.dataclass import BatchOneMod
from mmmv_ssl.data.datamodule.mm_datamodule import MMMaskDataModule
from mmmv_ssl.model.sits_encoder import MonoSITSEncoder
from mmmv_ssl.model.utils import build_encoder
from mmmv_ssl.module.alise_mm import AliseMM
from mt_ssl.utils.open import open_yaml
from omegaconf import DictConfig
from openeo_mmdc.dataset.to_tensor import load_transform_one_mod
from torch.nn import functional as F

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORK = "/work/scratch/data/kalinie"


def load_checkpoint(
    hydra_conf: str | Path, pretrain_module_ckpt_path: str | Path, mod: str
) -> MonoSITSEncoder:
    """Load MALICE checkpoint and instantiate Mono-modal SITS encoder"""

    pretrain_module_config = DictConfig(open_yaml(hydra_conf))

    pretrain_module_config.datamodule.datamodule.path_dir_csv = (
        f"{WORK}/results/alise_preentrained/data/"
    )

    try:
        old_datamodule: MMMaskDataModule = instantiate(
            pretrain_module_config.datamodule.datamodule,
            config_dataset=pretrain_module_config.dataset,
            batch_size=pretrain_module_config.train.batch_size,
            _recursive_=False,
        )
        try:
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
        except AttributeError:
            pretrained_pl_module: AliseMM = instantiate(
                pretrain_module_config.module,
                train_config=pretrain_module_config.train,
                stats=(
                    old_datamodule.all_transform.s2.stats,
                    old_datamodule.all_transform.s1_asc.stats,
                ),  # TODO do better than that load stats of each mod
                _recursive_=False,
            )

    except AttributeError:
        pretrained_pl_module: AliseMM = instantiate(
            pretrain_module_config.module,
        )

    pretrained_module = pretrained_pl_module.load_from_checkpoint(
        pretrain_module_ckpt_path, map_location=torch.device(DEVICE)
    )

    repr_encoder: MonoSITSEncoder = build_encoder(
        pretrained_module=pretrained_module, mod="s2" if "2" in mod else "s1"
    )

    repr_encoder.eval()

    return repr_encoder


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


def encode_series_malice(
    path_alise_model: str | Path,
    path_csv: str | Path,
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sat: str,
    output_path: str | Path,
    postfix: str,
):
    transform = load_transform_one_mod(path_csv, mod=sat.lower()).transform

    """Encode PASTIS SITS with MALICE"""

    log.info("Loader is ready")

    hydra_conf = os.path.join(
        os.path.dirname(path_alise_model), ".hydra", "config.yaml"
    )
    if not Path(hydra_conf).exists():
        hydra_conf = path_alise_model.split(".")[0] + "_config.yaml"

    repr_encoder = load_checkpoint(hydra_conf, path_alise_model, sat.lower())
    repr_encoder = repr_encoder.to(DEVICE)

    for patch_path in np.sort(os.listdir(os.path.join(dataset_path_oe, sat))):
        id_patch = patch_path.split("_")[-1].split(".")[0]

        if not Path(
            os.path.join(output_path, sat + "_" + postfix, f"{sat}_{id_patch}.pt")
        ).exists():
            batch = torch.load(
                os.path.join(dataset_path_oe, sat, patch_path), map_location=DEVICE
            )
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
                max_len=prepared_sits.shape[0],
                t=prepared_sits.shape[0],
                sits=prepared_sits,
                doy=doy,
            )

            # Prepare patch
            ref_norm = rearrange(
                transform(rearrange(prepared_sits, "t c h w -> c t h w")),
                "c t h w -> 1 t c h w",
            )

            input = BatchOneMod(
                sits=ref_norm.to(DEVICE),
                input_doy=doy[None, :].to(DEVICE),
                padd_index=padd_index[None, :].to(DEVICE),
            )
            repr_encoder.eval()
            with torch.no_grad():
                ort_out = repr_encoder.forward_keep_input_dim(input).repr

            torch.save(
                ort_out[0],
                os.path.join(output_path, sat + "_" + postfix, f"{sat}_{id_patch}.pt"),
            )
