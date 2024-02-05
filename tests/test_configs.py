#!/usr/bin/env/ python
# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales

# Code taken from:
# https://github.com/ashleve/lightning-hydra-template/blob/main/tests/test_configs.py


import hydra
from hydra import initialize, compose
import pytest

@pytest.mark.hpc
def test_train_config() -> None:
    """Tests the training configuration."""
    with initialize(version_base=None, config_path="../configs"):
        cfg_train = compose(config_name="train")
        assert cfg_train
        assert cfg_train.datamodule
        assert cfg_train.model
        assert cfg_train.trainer

        hydra.utils.instantiate(cfg_train.datamodule)
        hydra.utils.instantiate(cfg_train.model)
        hydra.utils.instantiate(cfg_train.trainer)
