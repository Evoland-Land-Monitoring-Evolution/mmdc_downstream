# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

import logging
import os
from pathlib import Path

import pandas as pd
import torch
from einops import rearrange
from hydra.utils import instantiate
from mmmv_ssl.data.dataclass import BatchOneMod, SITSOneMod
from mmmv_ssl.model.sits_encoder import MonoSITSAuxEncoder
from mmmv_ssl.model.utils import build_encoder
from mmmv_ssl.module.alise_mm import AliseMM
from mt_ssl.utils.open import open_yaml
from omegaconf import DictConfig
from openeo_mmdc.dataset.to_tensor import load_all_transforms

from mmdc_downstream_pastis.encode_series.encode import back_to_date, build_dm

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(hydra_conf, pretrain_module_ckpt_path, mod):
    pretrain_module_config = DictConfig(open_yaml(hydra_conf))

    pretrained_pl_module: AliseMM = instantiate(
        pretrain_module_config.module,
    )

    pretrained_module = pretrained_pl_module.load_from_checkpoint(
        pretrain_module_ckpt_path, map_location=torch.device(DEVICE)
    )

    repr_encoder: MonoSITSAuxEncoder = build_encoder(
        pretrained_module=pretrained_module, mod="s2" if "2" in mod else "s1"
    )

    repr_encoder.eval()

    return repr_encoder


def apply_transform_basic(
    batch_sits: torch.Tensor, transform: torch.nn.Module
) -> torch.Tensor:
    b, *_ = batch_sits.shape
    batch_sits = rearrange(batch_sits, "b t c h w -> c (b t ) h w")
    batch_sits = transform(batch_sits)
    batch_sits = rearrange(batch_sits, "c (b t ) h w -> b t c h w", b=b)
    return batch_sits


def encode_series_malise_aux(
    path_alise_model: str | Path,
    path_csv: str | Path,
    dataset_path_oe: str | Path,
    dataset_path_pastis: str | Path,
    sat: str,
    output_path: str | Path,
    postfix: str,
):
    transforms = load_all_transforms(
        path_csv, modalities=["s1_asc", "s2", "agera5", "dem"]
    )

    """Encode PASTIS SITS with MALICE"""

    log.info("Loader is ready")

    hydra_conf = os.path.join(
        os.path.dirname(path_alise_model), ".hydra", "config.yaml"
    )
    if not Path(hydra_conf).exists():
        hydra_conf = path_alise_model.split(".")[0] + "_config.yaml"

    repr_encoder = load_checkpoint(hydra_conf, path_alise_model, sat.lower())
    repr_encoder = repr_encoder.to(DEVICE)

    dm = build_dm(dataset_path_oe, dataset_path_pastis, [sat])

    dataset = dm.instanciate_dataset(fold=None)
    loader = dm.instanciate_data_loader(dataset, shuffle=False, drop_last=False)
    log.info("Loader is ready")
    for batch in loader:
        log.info(batch.id_patch)
        id_patch = batch.id_patch[0]

        img = batch.sits[sat].sits.data.img[0]  # [:, :, 32:-32, 32:-32]
        angles = batch.sits[sat].sits.data.angles[0]
        img = torch.concat((img, angles), dim=-3)

        meteo = batch.sits[sat].sits.meteo
        meteo = rearrange(meteo, "b t (d c) h w -> b t c d h w", c=8).nanmean((-2, -1))
        dem = batch.sits[sat].sits.dem

        true_doy = back_to_date(batch.sits[sat].true_doy, dm.reference_date)[0]

        reference_date = "2014-03-03"

        doy = torch.Tensor(
            (pd.to_datetime(true_doy) - pd.to_datetime(reference_date)).days
        )
        print(len(doy))
        print(doy)

        # Prepare patch
        ref_norm = apply_transform_basic(
            img[None, ...], getattr(transforms, sat.lower()).transform
        )
        meteo_norm = apply_transform_basic(
            meteo[..., None], transforms.agera5.transform
        ).squeeze(-1)
        dem_norm = apply_transform_basic(
            dem[:, None, :, :, :], transforms.dem.transform
        )

        input = SITSOneMod(
            sits=ref_norm.to(DEVICE)[0],
            input_doy=doy.to(DEVICE),
            meteo=meteo_norm.to(DEVICE)[0],
        ).apply_padding(ref_norm.shape[1] + 1)

        input = BatchOneMod(
            sits=input.sits[None, ...],
            input_doy=input.input_doy[None, ...],
            padd_index=input.padd_mask[None, ...].to(DEVICE),
            meteo=input.meteo[None, ...],
        )

        repr_encoder.eval()
        with torch.no_grad():
            ort_out = repr_encoder.forward_keep_input_dim(
                input, dem=dem_norm.to(DEVICE)
            ).repr
        torch.save(
            ort_out[0],
            os.path.join(output_path, sat + "_" + postfix, f"{sat}_{id_patch}.pt"),
        )
