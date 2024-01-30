#!/usr/bin/env python3
"""
Code to produce LAI GT with SNAP model.
This code is taken from
https://src.koda.cnrs.fr/yoel.zerah.1/prosailvae/-/blob/pvae/bvnet_regression/bvnet.py?ref_type=heads
"""

# pylint: skip-file
#  type: ignore

import os
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn

SNAP_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "weights")


def normalize(unnormalized: torch.Tensor, min_sample: torch.Tensor,
              max_sample: torch.Tensor):
    """
    Normalize with sample min and max of distribution
    """
    return 2 * (unnormalized - min_sample) / (max_sample - min_sample) - 1


def denormalize(normalized: torch.Tensor, min_sample: torch.Tensor,
                max_sample: torch.Tensor):
    """
    de-normalize with sample min and max of distribution
    """
    return 0.5 * (normalized + 1) * (max_sample - min_sample) + min_sample


@dataclass
class NormBVNETNNV2:
    """
    Min and max snap nn input and output for normalization
    """
    input_min: torch.Tensor = torch.tensor([
        0.0000, 0.0000, 0.0000, 0.00663797254225, 0.0139727270189,
        0.0266901380821, 0.0163880741923, 0.0000, 0.918595400582,
        0.342022871159, -1.0000
    ])

    input_max: torch.Tensor = torch.tensor([
        0.253061520472, 0.290393577911, 0.305398915249, 0.608900395798,
        0.753827384323, 0.782011770669, 0.493761397883, 0.49302598446, 1.0000,
        0.936206429175, 1.0000
    ])


class NormBVNETNNV3B:
    """
    Min and max snap nn input and output for normalization
    """
    input_min: torch.Tensor = torch.tensor([
        0.0000, 0.0000, 0.0000, 0.0119814116908, 0.0169060342706,
        0.0176448354545, 0.0147283842139, 0.0000, 0.979624800125,
        0.342108564072, -1.0000
    ])

    input_max: torch.Tensor = torch.tensor([
        0.247742161604, 0.305951681647, 0.327098829671, 0.599329840352,
        0.741682769861, 0.780987637826, 0.507673379171, 0.502205128583, 1.0000,
        0.927484749175, 1.0000
    ])


class NormBVNETNNV3A:
    """
    Min and max snap nn input and output for normalization
    """
    input_min: torch.Tensor = torch.tensor([
        0.0000, 0.0000, 0.0000, 0.008717364330310326, 0.019693160430621366,
        0.026217828282102625, 0.018931934894415213, 0.0000, 0.979624800125421,
        0.342108564072183, -1.0000
    ])

    input_max: torch.Tensor = torch.tensor([
        0.23901527463861838, 0.29172736471507876, 0.32652671459255694,
        0.5938903910368211, 0.7466909927207045, 0.7582393779705984,
        0.4929337190581187, 0.4877499217101771, 1.0000, 0.9274847491748729,
        1.0000
    ])


@dataclass
class DenormSNAPLAIV2:
    lai_min: torch.Tensor = torch.tensor(0.000319182538301)
    lai_max: torch.Tensor = torch.tensor(14.4675094548)


@dataclass
class DenormSNAPLAIV3B:
    lai_min: torch.Tensor = torch.tensor(0.000233773908827)
    lai_max: torch.Tensor = torch.tensor(13.834592547)


@dataclass
class DenormSNAPLAIV3A:
    lai_min: torch.Tensor = torch.tensor(0.00023377390882650673)
    lai_max: torch.Tensor = torch.tensor(13.834592547008839)


@dataclass
class DenormSNAPCCCV2:
    ccc_min: torch.Tensor = torch.tensor(0.00742669295987)
    ccc_max: torch.Tensor = torch.tensor(873.90822211)


@dataclass
class DenormSNAPCCCV3B:
    ccc_min: torch.Tensor = torch.tensor(0.0184770096032)
    ccc_max: torch.Tensor = torch.tensor(888.156665152)


@dataclass
class DenormSNAPCCCV3A:
    ccc_min: torch.Tensor = torch.tensor(0.01847700960324858)
    ccc_max: torch.Tensor = torch.tensor(888.1566651521919)


@dataclass
class DenormSNAPCWCV2:
    cwc_min: torch.Tensor = torch.tensor(3.85066859366e-06)
    cwc_max: torch.Tensor = torch.tensor(0.522417054645)


@dataclass
class DenormSNAPCWCV3B:
    cwc_min: torch.Tensor = torch.tensor(2.84352788861e-06)
    cwc_max: torch.Tensor = torch.tensor(0.419181347199)


@dataclass
class DenormSNAPCWCV3A:
    cwc_min: torch.Tensor = torch.tensor(4.227082600108468e-06)
    cwc_max: torch.Tensor = torch.tensor(0.5229998511245837)


def get_SNAP_norm_factors(ver: str = '2', variable='lai'):
    """
    Get normalization factor for BVNET NN
    """
    if ver == "2":
        bvnet_norm = NormBVNETNNV2()
        if variable == "lai":
            variable_min = DenormSNAPLAIV2().lai_min
            variable_max = DenormSNAPLAIV2().lai_max
        elif variable == "ccc":
            variable_min = DenormSNAPCCCV2().ccc_min
            variable_max = DenormSNAPCCCV2().ccc_max
        elif variable == "cab":
            variable_min = DenormSNAPCCCV2().ccc_min / (
                DenormSNAPLAIV2().lai_max - DenormSNAPLAIV2().lai_min)
            variable_max = DenormSNAPCCCV2().ccc_max / (
                DenormSNAPLAIV2().lai_max - DenormSNAPLAIV2().lai_min)
        elif variable == "cwc":
            variable_min = DenormSNAPCWCV2().cwc_min
            variable_max = DenormSNAPCWCV2().cwc_max
        elif variable == "cw":
            variable_min = DenormSNAPCWCV2().cwc_min / (
                DenormSNAPLAIV2().lai_max - DenormSNAPLAIV2().lai_min)
            variable_max = DenormSNAPCWCV2().cwc_max / (
                DenormSNAPLAIV2().lai_max - DenormSNAPLAIV2().lai_min)
        else:
            raise NotImplementedError
    elif ver == "3B":
        bvnet_norm = NormBVNETNNV3B()
        if variable == "lai":
            variable_min = DenormSNAPLAIV3B().lai_min
            variable_max = DenormSNAPLAIV3B().lai_max
        elif variable == "ccc":
            variable_min = DenormSNAPCCCV3B().ccc_min
            variable_max = DenormSNAPCCCV3B().ccc_max
        elif variable == "cab":
            variable_min = DenormSNAPCCCV3B().ccc_min / (
                DenormSNAPLAIV3B().lai_max - DenormSNAPLAIV3B().lai_min)
            variable_max = DenormSNAPCCCV3B().ccc_max / (
                DenormSNAPLAIV3B().lai_max - DenormSNAPLAIV3B().lai_min)
        elif variable == "cwc":
            variable_min = DenormSNAPCWCV3B().cwc_min
            variable_max = DenormSNAPCWCV3B().cwc_max
        elif variable == "cw":
            variable_min = DenormSNAPCWCV3B().cwc_min / (
                DenormSNAPLAIV3B().lai_max - DenormSNAPLAIV3B().lai_min)
            variable_max = DenormSNAPCWCV3B().cwc_max / (
                DenormSNAPLAIV3B().lai_max - DenormSNAPLAIV3B().lai_min)
        else:
            raise NotImplementedError
    elif ver == "3A":
        bvnet_norm = NormBVNETNNV3A()
        if variable == "lai":
            variable_min = DenormSNAPLAIV3A().lai_min
            variable_max = DenormSNAPLAIV3A().lai_max
        elif variable == "ccc":
            variable_min = DenormSNAPCCCV3A().ccc_min
            variable_max = DenormSNAPCCCV3A().ccc_max
        elif variable == "cab":
            variable_min = DenormSNAPCCCV3A().ccc_min / (
                DenormSNAPLAIV3A().lai_max - DenormSNAPLAIV3A().lai_min)
            variable_max = DenormSNAPCCCV3A().ccc_max / (
                DenormSNAPLAIV3A().lai_max - DenormSNAPLAIV3A().lai_min)
        elif variable == "cwc":
            variable_min = DenormSNAPCWCV3A().cwc_min
            variable_max = DenormSNAPCWCV3A().cwc_max
        elif variable == "cw":
            variable_min = DenormSNAPCWCV3A().cwc_min / (
                DenormSNAPLAIV3A().lai_max - DenormSNAPLAIV3A().lai_min)
            variable_max = DenormSNAPCWCV3A().cwc_max / (
                DenormSNAPLAIV3A().lai_max - DenormSNAPLAIV3A().lai_min)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return bvnet_norm.input_min, bvnet_norm.input_max, variable_min, variable_max


class BVNET(nn.Module):
    """
    Neural Network with BVNET architecture to predict LAI from S2 reflectances and angles
    """

    def __init__(self,
                 device: str = 'cpu',
                 ver: str = "3A",
                 variable='lai',
                 third_layer=False):
        super().__init__()
        assert ver in ["2", "3A", "3B"]
        input_min, input_max, variable_min, variable_max = get_SNAP_norm_factors(
            ver=ver, variable=variable)
        input_size = len(input_max)  # 8 bands + 3 angles
        hidden_layer_size = 5
        if not third_layer:
            layers = OrderedDict([
                ('layer_1',
                 nn.Linear(in_features=input_size,
                           out_features=hidden_layer_size).float()),
                ('tanh', nn.Tanh()),
                ('layer_2',
                 nn.Linear(in_features=hidden_layer_size,
                           out_features=1).float())
            ])
        else:
            layers = OrderedDict([
                ('layer_0',
                 nn.Linear(in_features=input_size,
                           out_features=input_size).float()),
                ('tanh', nn.Tanh()),
                ('layer_1',
                 nn.Linear(in_features=input_size,
                           out_features=hidden_layer_size).float()),
                ('tanh', nn.Tanh()),
                ('layer_2',
                 nn.Linear(in_features=hidden_layer_size,
                           out_features=1).float())
            ])
        self.input_min = input_min.float().to(device)
        self.input_max = input_max.float().to(device)
        self.variable_min = variable_min.float().to(device)
        self.variable_max = variable_max.float().to(device)
        self.net: nn.Sequential = nn.Sequential(layers).to(device)
        self.device = device
        self.ver = ver
        self.variable = variable


    def load_weights(self, dir, prefix=""):
        weights_1 = torch.from_numpy(
            pd.read_csv(os.path.join(
                dir, f"{prefix}{self.ver}_{self.variable}_weights_1.csv"),
                        header=None).values).float()
        weights_2 = torch.from_numpy(
            pd.read_csv(os.path.join(
                dir, f"{prefix}{self.ver}_{self.variable}_weights_2.csv"),
                        header=None).values).float()
        bias_1 = torch.from_numpy(
            pd.read_csv(os.path.join(
                dir, f"{prefix}{self.ver}_{self.variable}_bias_1.csv"),
                        header=None).values).float()
        bias_2 = torch.from_numpy(
            pd.read_csv(os.path.join(
                dir, f"{prefix}{self.ver}_{self.variable}_bias_2.csv"),
                        header=None).values).float()
        self.net.layer_1.bias = nn.Parameter(
            bias_1.to(self.device).reshape(-1))
        self.net.layer_1.weight = nn.Parameter(weights_1.to(self.device))
        self.net.layer_2.bias = nn.Parameter(
            bias_2.to(self.device).reshape(-1))
        self.net.layer_2.weight = nn.Parameter(weights_2.to(self.device))

    def set_snap_weights(self):
        """
        Set Neural Network weights and biases to BVNET's original values
        """
        self.load_weights(SNAP_WEIGHTS_PATH, prefix="weiss_")

    def forward(self, s2_data: torch.Tensor, spatial_mode=False):
        """
        Forward method of BVNET NN to predict a biophysical variable
        """
        if spatial_mode:
            if len(s2_data.size()) == 3:
                (_, size_h, size_w) = s2_data.size()
                s2_data = s2_data.permute(1, 2, 0).reshape(size_h * size_w, -1)
            else:
                raise NotImplementedError
        s2_data_norm = normalize(s2_data, self.input_min, self.input_max)
        variable_norm = self.net.forward(s2_data_norm)
        variable = denormalize(variable_norm, self.variable_min,
                               self.variable_max)
        if spatial_mode:
            variable = variable.reshape(size_h, size_w, 1).permute(2, 0, 1)
        return variable
