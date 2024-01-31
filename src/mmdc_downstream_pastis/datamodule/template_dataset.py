from typing import Literal

from torch.utils.data import Dataset

from mt_ssl.data.dataset.utils import randomcropindex


class TemplateDataset(Dataset):
    def __init__(
        self,
        crop_size: int = 64,
        crop_type: Literal["Center", "Random"] = "Center",
        transform: None = None,
        dataset_type: Literal["train", "val", "test"] = "test",
    ):
        self.crop_size = crop_size
        self.dataset_type = dataset_type
        self.transform = transform
        self.crop_type = crop_type

    def get_crop_idx(self, rows, cols) -> tuple[int, int]:
        """return the coordinate, width and height of the window loaded
        by the SITS depending od the value of the attribute
        self.crop_type

        Args:
            rows: int
            cols: int
        """
        if self.crop_type == "Random":
            return randomcropindex(rows, cols, self.crop_size, self.crop_size)

        return int(rows // 2 - self.crop_size // 2), int(
            cols // 2 - self.crop_size // 2
        )
