#!/usr/bin/env python3
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales

import pytest

from mmdc_downstream_pastis.datamodule.datatypes import PastisFolds
from mmdc_downstream_pastis.datamodule.pastis_oe import PastisDataModule

dataset_path_oe = "/work/CESBIO/projects/DeepChange/Ekaterina/Pastis_OE"
dataset_path_pastis = "/work/CESBIO/projects/DeepChange/Iris/PASTIS"


def build_dm(sats) -> PastisDataModule:
    """Builds datamodule"""
    return PastisDataModule(
        dataset_path_oe=dataset_path_oe,
        dataset_path_pastis=dataset_path_pastis,
        folds=PastisFolds([1, 2, 3], [4], [5]),
        sats=sats,
        task="semantic",
        batch_size=1,
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
        for (batch_dict, target, mask, id_patch), _ in zip(loader, range(4)):
            assert list(batch_dict.keys()) == sats
            for sat in sats:
                b_s_x, t_s_x, nb_b, p_s_x, _ = batch_dict[sat].sits.data.img.shape
                b_s_ang, t_s_ang, nb_b_ang, p_s_ang, _ = batch_dict[
                    sat
                ].sits.data.angles.shape
                b_s_imsk, t_s_imsk, nb_b_imsk, p_s_imsk, _ = batch_dict[
                    sat
                ].sits.data.mask.shape
                b_s_dem, nb_b_dem, p_s_dem, _ = batch_dict[sat].sits.dem.shape
                b_s_meteo, t_s_meteo, nb_b_meteo, p_s_meteo, _ = batch_dict[
                    sat
                ].sits.meteo.shape
                b_s_y, p_s_y, _ = target.shape
                b_s_m, p_s_m, _ = mask.shape
                b_s_id = len(id_patch)
                assert nb_b_dem == 4
                assert nb_b_meteo == 48
                assert (
                    b_s_x
                    == b_s_ang
                    == b_s_imsk
                    == b_s_dem
                    == b_s_meteo
                    == b_s_y
                    == b_s_m
                    == b_s_id
                )
                assert t_s_x == t_s_ang == t_s_imsk == t_s_meteo
                # assert nb_b == PASTIS_BANDS[sat]
                assert (
                    p_s_x
                    == p_s_ang
                    == p_s_imsk
                    == p_s_dem
                    == p_s_meteo
                    == p_s_y
                    == p_s_m
                )
