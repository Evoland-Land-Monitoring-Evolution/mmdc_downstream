#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
import os

import pytest

from mmdc_downstream_pastis.datamodule.datatypes import PastisFolds
from mmdc_downstream_pastis.datamodule.pastis_oe_aux import PastisAuxDataModule

# dataset_path_oe = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis_OE"
# dataset_path_pastis = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"

dataset_path_oe = f"{os.environ['SCRATCH']}/scratch_data/Pastis_OE"
dataset_path_pastis = f"{os.environ['SCRATCH']}/scratch_data/Pastis"


def build_dm(sats) -> PastisAuxDataModule:
    """Builds datamodule"""
    return PastisAuxDataModule(
        dataset_path_oe=dataset_path_oe,
        dataset_path_pastis=dataset_path_pastis,
        folds=PastisFolds([1, 2, 3], [4], [5]),
        sats=sats,
        task="semantic",
        batch_size=5,
    )


@pytest.mark.parametrize(
    "sats",
    [
        ["S2"],
        ["S1_ASC"],
        ["S1_DESC"],
        ["S2", "S1_ASC"],
        ["S2", "S1_DESC"],
        ["S1_ASC", "S1_DESC"],
        ["S2", "S1_ASC", "S1_DESC"],
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
        for batch, _ in zip(loader, range(4)):
            batch_dict = batch.sits
            mask_sits = batch.sits_mask
            mask = batch.gt_mask
            target = batch.gt
            id_patch = batch.id_patch

            assert list(batch_dict.keys()) == sats

            for sat in sats:
                b_s_x, t_s_x, nb_b, p_s_x, _ = batch_dict[sat].shape
                b_s_imsk, t_s_imsk, nb_b_imsk, p_s_imsk, _ = mask_sits[sat].shape

                b_s_y, p_s_y, _ = target.shape
                b_s_m, p_s_m, _ = mask.shape
                b_s_id = len(id_patch)

                if "S2" in sat:
                    assert nb_b == 64
                else:
                    assert nb_b == 56

                assert b_s_x == b_s_imsk == b_s_y == b_s_m == b_s_id
                assert t_s_x == t_s_imsk
                # assert nb_b == PASTIS_BANDS[sat]
                assert p_s_x == p_s_imsk == p_s_y == p_s_m
