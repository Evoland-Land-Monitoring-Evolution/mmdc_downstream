#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO
"""
Tests for the datamodules.pastis_dataset module
"""
import os

import pytest
from torch.utils import data as tu_data

from mmdc_downstream_pastis.datamodule.pastis_oe_only_satellite import (
    PASTIS_BANDS,
    PastisDataModule,
    PASTISDataset,
    PastisFolds,
    PASTISOptions,
    PASTISTask,
    SatId,
    pad_collate,
)

# dataset_path_oe = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis_OE"
# dataset_path_pastis = "/home/kalinichevae/scratch_jeanzay/scratch_data/Pastis"

dataset_path_oe = f"{os.environ['SCRATCH']}/scratch_data/Pastis_OE"
dataset_path_pastis = f"{os.environ['SCRATCH']}/scratch_data/Pastis"


def instanciate_data_loader(
    sats: list[SatId], data_folder: str, data_folder_oe: str, batch_size: int
) -> tu_data.DataLoader:
    """Return a data loader with the PASTIS data set"""
    return tu_data.DataLoader(
        PASTISDataset(
            PASTISOptions(PASTISTask(sats=sats), data_folder, data_folder_oe)
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: pad_collate(x, pad_value=0),
    )


@pytest.mark.hpc
@pytest.mark.parametrize("sats", [["S2"], ["S1A"], ["S2", "S1D"], ["S2_MSK"]])
def test_pastisds_instantiate(sats: list[SatId]) -> None:
    """PASTIS dataset instanciation"""
    d_s = PASTISDataset(
        PASTISOptions(
            PASTISTask(sats=sats),
            dataset_path_pastis,
            dataset_path_oe,
            folds=[1, 2, 3, 4, 5],
        )
    )
    assert d_s is not None


def build_dm(sats: list[SatId]) -> PastisDataModule:
    """Builds datamodule"""
    return PastisDataModule(
        dataset_path_pastis=dataset_path_pastis,
        dataset_path_oe=dataset_path_oe,
        folds=PastisFolds([1, 2, 3], [4], [5]),
        task=PASTISTask(sats=sats),
        batch_size=5,
    )


@pytest.mark.trex
@pytest.mark.parametrize(
    "sats",
    [["S2"]],
    # ["S1_ASC"],
    # ["S1_DESC"],
    # ["S2", "S1_ASC"],
    # ["S2", "S1_DESC"],
    # ["S1_ASC", "S1_DESC"],
    # ["S2", "S1_ASC", "S1_DESC"]
)
def test_pastisds_dataloader(sats: list[SatId]) -> None:
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
        for batch, _ in zip(loader, range(10)):
            inputs = batch.sits
            dates = batch.doy
            labels = batch.gt

            print(inputs)
            assert list(inputs.keys()) == sats
            assert list(dates.keys()) == sats
            for sat in sats:
                if sat == "S2_MSK":
                    b_s_x, t_s_x, p_s_x, _ = inputs[sat].shape
                    nb_b = 1
                else:
                    b_s_x, t_s_x, nb_b, p_s_x, _ = inputs[sat].shape
                b_s_d, t_s_d = dates[sat].shape
                b_s_y, p_s_y, _ = labels.shape
                assert t_s_d > 2
                assert b_s_x == b_s_d == b_s_y
                assert t_s_d == t_s_x
                assert nb_b == PASTIS_BANDS[sat]
                assert p_s_x == p_s_y
