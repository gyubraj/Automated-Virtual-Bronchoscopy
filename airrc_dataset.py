import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class AirRCPatchDataset(Dataset):
    def __init__(
        self,
        split_json,
        augment=False,
        flip_prob=0.5,
        intensity_prob=0.8,
        noise_prob=0.3,
    ):
        split_json = Path(split_json)

        with open(split_json, "r") as f:
            self.items = json.load(f)
        self.augment = augment
        self.flip_prob = flip_prob
        self.intensity_prob = intensity_prob
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        ct = np.load(item["image"]).astype(np.float32)       # 1, D, H, W
        target = np.load(item["target"]).astype(np.float32)  # 2, D, H, W

        if self.augment:
            ct, target = self.apply_augmentation(ct, target)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(ct)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "patch_id": item["patch_id"],
            "uid": item["uid"],
        }

    def apply_augmentation(self, ct, target):
        # Spatial flips preserve airway topology and are cheap for large 3D patches.
        for axis in (1, 2, 3):
            if np.random.random() < self.flip_prob:
                ct = np.flip(ct, axis=axis)
                target = np.flip(target, axis=axis)

        # Mild CT-domain augmentation helps AirRC-trained models tolerate LIDC-like intensity differences.
        if np.random.random() < self.intensity_prob:
            scale = np.random.uniform(0.85, 1.15)
            shift = np.random.uniform(-0.08, 0.08)
            ct = ct * scale + shift

            gamma = np.random.uniform(0.85, 1.20)
            ct = np.clip(ct, 0.0, 1.0)
            ct = np.power(ct, gamma)

        if np.random.random() < self.noise_prob:
            sigma = np.random.uniform(0.005, 0.025)
            ct = ct + np.random.normal(0.0, sigma, size=ct.shape).astype(np.float32)

        ct = np.clip(ct, 0.0, 1.0).astype(np.float32)
        return ct, target.astype(np.float32)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parent
    train_json = root / "datasets" / "airrc_patches" / "splits" / "train.json"

    dataset = AirRCPatchDataset(train_json)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    batch = next(iter(loader))

    print("image:", batch["image"].shape)
    print("target:", batch["target"].shape)
    print("patch_id:", batch["patch_id"])
