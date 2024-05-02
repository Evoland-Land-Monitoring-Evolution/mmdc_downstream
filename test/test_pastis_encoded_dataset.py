#!/usr/bin/env python3
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
import os

import pytest

from mmdc_downstream_pastis.datamodule.datatypes import PastisFolds
from mmdc_downstream_pastis.datamodule.pastis_oe import PastisDataModule

model = "2024-04-05_14-58-22"
dataset_path_oe = f"{os.environ['WORK']}/results/Pastis_encoded"
dataset_path_pastis = f"{os.environ['SCRATCH']}/scratch_data/Pastis"
# dataset_path_oe = "/home/kalinichevae/jeanzay/results/Pastis_encoded"
# dataset_path_pastis = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"
dataset_path_oe = os.path.join(dataset_path_oe, model)


def build_dm(sats) -> PastisDataModule:
    """Builds datamodule"""
    return PastisDataModule(
        dataset_path_oe=dataset_path_oe,
        dataset_path_pastis=dataset_path_pastis,
        folds=PastisFolds([1, 2, 3], [4], [5]),
        sats=sats,
        task="semantic",
        batch_size=2,
    )


@pytest.mark.parametrize(
    "sats",
    [
        ["S2"],
        ["S1"],
        ["S2", "S1"],
    ],
)
def test_pastisds_dataloader(sats) -> None:
    """Use a dataloader with PASTIS dataset"""
    dm = build_dm(sats)
    dm.setup(stage="fit")
    dm.setup(stage="test")
    assert (
        hasattr(dm, "train_dataloader")
        and hasattr(dm, "val_dataloader")
        and hasattr(dm, "test_dataloader")
    )  # type: ignore[truthy-function]
    for loader in (dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()):
        assert loader
        for (batch_dict, mask_dict, doys_dict, target, mask, id_patch), _ in zip(
            loader, range(4)
        ):
            if len(sats) > 1:
                assert list(batch_dict.keys()) == sats
                for sat in sats:
                    b_s_x, t_s_x, nb_b, p_s_x, _ = batch_dict[sat].shape

                    b_s_imsk, t_s_imsk, nb_b_imsk, p_s_imsk, _ = mask_dict[sat].shape

                    b_s_y, p_s_y, _ = target.shape
                    b_s_m, p_s_m, _ = mask.shape
                    b_s_id = len(id_patch)

                    assert b_s_x == b_s_imsk == b_s_y == b_s_m == b_s_id
                    assert t_s_x == t_s_imsk
                    assert p_s_x == p_s_imsk == p_s_y == p_s_m
            else:
                b_s_x, t_s_x, nb_b, p_s_x, _ = batch_dict.shape

                b_s_imsk, t_s_imsk, nb_b_imsk, p_s_imsk, _ = mask_dict.shape

                b_s_y, p_s_y, _ = target.shape
                b_s_m, p_s_m, _ = mask.shape
                b_s_id = len(id_patch)

                assert b_s_x == b_s_imsk == b_s_y == b_s_m == b_s_id
                assert t_s_x == t_s_imsk
                assert p_s_x == p_s_imsk == p_s_y == p_s_m
