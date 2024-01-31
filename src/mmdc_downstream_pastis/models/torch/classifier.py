import logging

import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from mt_ssl.data.classif_class import ClassifBInput
from mt_ssl.model.convolutionalblock import LastBlock
from mt_ssl.model.mtan_repr_encoder_2pe import MTANReprEncoder2Pe
from mt_ssl.model.shallow_classifier import MasterQueryDecoding
from mt_ssl.model.ubarn import UBarn
from mt_ssl.module.dataclass import OutFTForward
from mt_ssl.module.interpol_module_template import load_reference_time_points

my_logger = logging.getLogger(__name__)


class UBarnClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        repr_encoder: MTANReprEncoder2Pe | DictConfig,
        reference_time_points: torch.Tensor | ListConfig | str | None,
        d_model: int,
        load_true_reference_time_points: bool = False,
        freeze_repr_encoder=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.freeze_repr_encoder = freeze_repr_encoder
        self.d_model = d_model
        self.out_channels = out_channels
        self.input_channels = in_channels
        if reference_time_points is not None:
            self.reference_time_points = load_reference_time_points(
                reference_time_points, as_doy=False
            )
        else:
            self.reference_time_points = None

        if load_true_reference_time_points:
            my_logger.debug("Load reference time points as DOY")
            self.reference_time_points2 = load_reference_time_points(
                reference_time_points, as_doy=True
            )

        else:
            my_logger.debug("Do not load ref time points as DOY")
            self.reference_time_points2 = None
        if isinstance(repr_encoder, nn.Module):
            self.repr_encoder = repr_encoder
        elif isinstance(repr_encoder, DictConfig):
            self.repr_encoder: UBarn = instantiate(
                repr_encoder,
                input_channels=in_channels,
                reference_time_points=self.reference_time_points,
                reference_time_points2=self.reference_time_points2,
                d_model=d_model,
                _recursive_=False,
            )
        else:
            raise NotImplementedError
        if freeze_repr_encoder:
            self.repr_encoder.requires_grad_(False)
        else:
            self.repr_encoder.requires_grad_(True)


class UbarnMtanClass(UBarnClassifier):
    def __init__(
        self,
        in_channels,
        out_channels,
        repr_encoder: MTANReprEncoder2Pe | DictConfig,
        reference_time_points: torch.Tensor | ListConfig | str,
        d_model: int,
        decoder_config: nn.Module | DictConfig,
        load_true_reference_time_points: bool = False,
        freeze_repr_encoder=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            repr_encoder,
            reference_time_points,
            d_model,
            load_true_reference_time_points,
            *args,
            **kwargs,
        )

        if isinstance(decoder_config, nn.Module):
            self.shallow_classifier = decoder_config
        else:
            self.shallow_classifier: LastBlock = instantiate(
                decoder_config,
                inplanes=self.d_model
                * len(self.repr_encoder.reference_time_points),
                planes=out_channels,
            )

    def forward(self, batch: ClassifBInput) -> OutFTForward:
        out_repr = self.repr_encoder.forward_keep_input_dim(batch)
        repr = rearrange(
            out_repr.repr, " b k c h w -> b h w (k c)"
        )  # features are fetures and dates
        out_class = self.shallow_classifier(repr)  # out dim is b h w nc

        return OutFTForward(out_class, out_repr.repr)


class UbarnSC(UBarnClassifier):
    def __init__(
        self,
        in_channels,
        out_channels,
        repr_encoder: MTANReprEncoder2Pe | DictConfig,
        reference_time_points: None,
        d_model: int,
        decoder_config: nn.Module | DictConfig,
        load_true_reference_time_points: bool = False,
        freeze_repr_encoder: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            repr_encoder,
            reference_time_points,
            d_model,
            load_true_reference_time_points,
            freeze_repr_encoder,
            *args,
            **kwargs,
        )
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.reference_time_points = None
        self.reference_time_points2 = None
        if isinstance(decoder_config, MasterQueryDecoding):
            self.shallow_classifier: MasterQueryDecoding = decoder_config
        else:
            self.shallow_classifier: MasterQueryDecoding = instantiate(
                decoder_config,
                inplanes=self.d_model,
                num_classes=out_channels,
            )

    def forward(self, batch: ClassifBInput) -> OutFTForward:
        out_repr = self.repr_encoder.forward_keep_input_dim(batch)
        # features are fetures and dates

        out_class = self.shallow_classifier(
            out_repr.repr, key_padding_mask=batch.padd_index
        )  # out dim is b h w nc
        return OutFTForward(out_class, out_repr.repr)
