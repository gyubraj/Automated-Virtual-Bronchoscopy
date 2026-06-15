import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AirRCPatchDataset(Dataset):
    def __init__(self, split_json):
        split_json = Path(split_json)

        with open(split_json, "r") as f:
            self.items = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        ct = np.load(item["image"]).astype(np.float32)       # 1, D, H, W
        target = np.load(item["target"]).astype(np.float32)  # 2, D, H, W

        return {
            "image": torch.from_numpy(ct),
            "target": torch.from_numpy(target),
            "patch_id": item["patch_id"],
            "uid": item["uid"],
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    home = Path.home()
    root = home / "AMS_Project" / "datasets_new"

    train_json = root / "airrc_patches" / "splits" / "train.json"

    dataset = AirRCPatchDataset(train_json)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    batch = next(iter(loader))

    print("image:", batch["image"].shape)
    print("target:", batch["target"].shape)
    print("patch_id:", batch["patch_id"])