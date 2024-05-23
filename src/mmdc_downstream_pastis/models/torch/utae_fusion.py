# ruff: noqa
# flake8: noqa
"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import logging
from typing import Literal

import torch
import torch.nn as nn

from mmdc_downstream_pastis.models.torch.utae import UTAE, ConvBlock, DownConvBlock

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class UTAEFusion(nn.Module):
    def __init__(
        self,
        input_dim: dict[str, int] = {"S1": 12, "S2": 12},
        encoder_widths: list[int] = [64, 64, 64, 128],
        decoder_widths: list[int] = [32, 32, 64, 128],
        out_conv: list[int] = [32, 20],
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        agg_mode: str = "att_group",
        encoder_norm: str = "group",
        n_head: int = 16,
        d_model: int = 256,
        d_k: int = 4,
        pad_value: int = 0,
        padding_mode: int = "reflect",
        fusion: Literal["mean", "concat"] = "mean",
        satellites: list[str] = ("S1", "S2"),
    ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths
            of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which
            the number of channels should be given is also from top to bottom. If this argument is not specified
            the decoder will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = out_conv[-1]
        self.fusion = fusion
        if self.fusion == "mean":
            self.out_conv = ConvBlock(
                nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode
            )
        else:
            self.out_conv = ConvBlock(
                nkernels=[decoder_widths[0] * len(satellites)] + out_conv,
                padding_mode=padding_mode,
            )

        self.satellites = satellites

        self.encoders = nn.ModuleDict(
            {
                sat: UTAE(
                    input_dim[sat],
                    encoder_widths,
                    decoder_widths,
                    out_conv,
                    str_conv_k,
                    str_conv_s,
                    str_conv_p,
                    agg_mode,
                    encoder_norm,
                    n_head,
                    d_model,
                    d_k,
                    encoder=True,
                    return_maps=True,
                    pad_value=pad_value,
                    padding_mode="reflect",
                )
                for sat in satellites
            }
        )

    def forward(
        self,
        input: dict[str, torch.Tensor],
        batch_positions: dict[str, torch.Tensor] | None = None,
        return_att: bool = False,
    ):
        """Forward pass of multimodal UTAE"""

        out = [
            self.encoders[sat].forward(
                input[sat], batch_positions[sat], return_att=return_att
            )[0]
            for sat in self.satellites
        ]
        if self.fusion == "mean":
            fused = torch.sum(torch.stack(out), dim=0)
        else:  # "concat"
            fused = torch.cat(out, 1)

        out = self.out_conv(fused)
        return out
