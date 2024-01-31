from dataclasses import dataclass

from torch import Tensor

import mt_ssl.model.projector


@dataclass
class OutUTAEForward:
    seg_map: Tensor
    attn: Tensor | None = None
    feature_maps: list[Tensor] | None = None


@dataclass
class OutDecoderTransformer:
    out_sits: Tensor
    attn: list[Tensor] | None = None
    attn_ed: list[Tensor] | None = None


@dataclass
class OutputMunTAN:
    repr: Tensor
    attn_mtan: Tensor | None = None


@dataclass
class OutBasicDecoder:
    rec: Tensor
    output_doy: Tensor


@dataclass
class OutMLPDecoder(OutBasicDecoder):
    pass


@dataclass
class OutDecoderGF(OutBasicDecoder):
    pass


@dataclass
class OutDecodHEVAEMUNTAN:
    rec: Tensor
    output_doy: Tensor
    decoder_muntan_attn: Tensor
    prob_path: Tensor | None


@dataclass
class ProjectorConfig:
    _target_: mt_ssl.model.projector.ProjectorBase
    d_int: int
    d_out: int
    l_dim: list = None
