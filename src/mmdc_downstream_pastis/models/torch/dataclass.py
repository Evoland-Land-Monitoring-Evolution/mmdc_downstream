from dataclasses import dataclass

from torch import Tensor


@dataclass
class OutUTAEForward:
    seg_map: Tensor
    attn: Tensor | None = None
    feature_maps: list[Tensor] | None = None
