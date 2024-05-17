#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales

# based on
# :https://github.com/ashleve/lightning-hydra-template/blob/main/tests/test_train.py

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

from mmdc_downstream_lai.training_pipeline import train


@pytest.mark.slow
@pytest.mark.skip(reason="Have to look into hydra testing")
def test_train_fast_dev_run():
    """
    test training loop
    Run for 1 train, val, test step
    """

    with initialize(version_base=None, config_path="../configs"):
        cfg_train = compose(config_name="train")
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        HydraConfig().set_config(cfg_train)
        # instantiate(cfg_train.datamodule)
        # instantiate(cfg_train.model)

        train(cfg_train)
