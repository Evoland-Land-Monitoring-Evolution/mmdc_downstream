"""Utils"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mmdc_downstream_pastis.datamodule.pastis_oe import PastisOEDataModule


def back_to_date(
    days_int: torch.Tensor, ref_date: str = "2018-09-01"
) -> list[np.array]:
    """
    Go back from integral doy (number of days from ref date)
    to calendar doy
    """
    return [
        np.asarray(
            pd.to_datetime(
                [
                    pd.Timedelta(dd, "d") + pd.to_datetime(ref_date)
                    for dd in days_int.cpu().numpy()[d]
                ]
            )
        )
        for d in range(len(days_int))
    ]


def build_dm(
    dataset_path_oe: str | Path, dataset_path_pastis: str | Path, sats: list[str]
) -> PastisOEDataModule:
    """Builds datamodule"""
    return PastisOEDataModule(
        dataset_path_oe=dataset_path_oe,
        dataset_path_pastis=dataset_path_pastis,
        folds=None,
        sats=sats,
        task="semantic",
        batch_size=1,
        crop_size=None,
    )
