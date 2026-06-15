from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from WingsNet import WingsNet
from airrc_dataset import AirRCPatchDataset


# Boundary-aware loss settings. These can be tuned after a short validation run.
WALL_POSITIVE_WEIGHT = 2.0
BOUNDARY_WEIGHT = 5.0
BOUNDARY_KERNEL_SIZE = 3


def dice_loss(pred, target, weight=None):
    """Soft Dice loss averaged across the lumen and wall channels."""
    smooth = 1.0
    dims = (0, 2, 3, 4)

    if weight is None:
        weight = torch.ones_like(target)

    intersection = (weight * pred * target).sum(dim=dims)
    denominator = (weight * pred).sum(dim=dims) + (weight * target).sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def channel_dice(pred, target):
    """Unweighted Dice metrics for clear lumen/wall reporting."""
    lumen_dice = 1.0 - dice_loss(pred[:, 0:1], target[:, 0:1])
    wall_dice = 1.0 - dice_loss(pred[:, 1:2], target[:, 1:2])
    return lumen_dice, wall_dice


def morphological_boundary(mask, kernel_size=BOUNDARY_KERNEL_SIZE):
    """Create a one-voxel-scale 3D boundary band from a binary target mask."""
    padding = kernel_size // 2
    dilated = F.max_pool3d(mask, kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool3d(
        1.0 - mask,
        kernel_size,
        stride=1,
        padding=padding,
    )
    return (dilated - eroded).clamp(0.0, 1.0)


def make_boundary_weights(target):
    """Build channel-specific weights around the lumen-wall interface and wall edge."""
    lumen = target[:, 0:1].clamp(0.0, 1.0)
    wall = target[:, 1:2].clamp(0.0, 1.0)

    lumen_dilated = F.max_pool3d(
        lumen,
        BOUNDARY_KERNEL_SIZE,
        stride=1,
        padding=BOUNDARY_KERNEL_SIZE // 2,
    )
    wall_dilated = F.max_pool3d(
        wall,
        BOUNDARY_KERNEL_SIZE,
        stride=1,
        padding=BOUNDARY_KERNEL_SIZE // 2,
    )

    # Include voxels on both sides of the transition between lumen and wall.
    interface = torch.maximum(lumen_dilated * wall, lumen * wall_dilated)
    wall_boundary = morphological_boundary(wall)

    weights = torch.ones_like(target)
    weights[:, 0:1] += BOUNDARY_WEIGHT * interface
    weights[:, 1:2] += WALL_POSITIVE_WEIGHT * wall
    weights[:, 1:2] += BOUNDARY_WEIGHT * torch.maximum(interface, wall_boundary)
    return weights


def boundary_aware_loss(logits, target):
    """Weighted BCE plus weighted Dice for lumen and wall prediction."""
    weights = make_boundary_weights(target)

    voxel_bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    weighted_bce = (voxel_bce * weights).sum() / weights.sum().clamp_min(1.0)

    probability = torch.sigmoid(logits)
    weighted_dice = dice_loss(probability, target, weight=weights)
    return weighted_bce + weighted_dice


def unpack_batch(batch):
    if isinstance(batch, dict):
        image = batch["image"]
        target = batch["target"]
    else:
        image = batch[0]
        target = batch[1]

    image = image.float()
    target = target.float()

    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError(f"Expected image shape [B, 1, D, H, W], got {tuple(image.shape)}")
    if target.ndim != 5 or target.shape[1] != 2:
        raise ValueError(f"Expected target shape [B, 2, D, H, W], got {tuple(target.shape)}")

    return image, target



def train():
    max_epochs = 100
    batch_size = 2
    learning_rate = 1e-4
    weight_decay = 1e-4

    train_json = "datasets/airrc_patches/splits/train.json"
    val_json = "datasets/airrc_patches/splits/val.json"
    save_dir = Path("./saved_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset = AirRCPatchDataset(train_json)
    valid_dataset = AirRCPatchDataset(val_json)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = WingsNet(in_channel=1, n_classes=2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_lumen_dice_sum = 0.0
        train_wall_dice_sum = 0.0
        train_count = 0

        for iteration, batch in enumerate(train_loader):
            image, target = unpack_batch(batch)
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_en, pred_de = model(image)

            loss_en = boundary_aware_loss(pred_en, target)
            loss_de = boundary_aware_loss(pred_de, target)
            loss = loss_de + 0.5 * loss_en

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_de_prob = torch.sigmoid(pred_de)
                lumen_dice, wall_dice = channel_dice(pred_de_prob, target)

            current_batch_size = image.shape[0]
            train_loss_sum += loss.item() * current_batch_size
            train_lumen_dice_sum += lumen_dice.item() * current_batch_size
            train_wall_dice_sum += wall_dice.item() * current_batch_size
            train_count += current_batch_size

            if iteration % 10 == 0:
                print(
                    "epoch: %d, iter: %d/%d, boundary loss: %.4f, "
                    "lumen dice: %.4f, wall dice: %.4f"
                    % (
                        epoch,
                        iteration,
                        len(train_loader),
                        train_loss_sum / max(train_count, 1),
                        train_lumen_dice_sum / max(train_count, 1),
                        train_wall_dice_sum / max(train_count, 1),
                    )
                )

        val_loss, val_lumen_dice, val_wall_dice = validation(
            model=model,
            valid_loader=valid_loader,
            device=device,
        )

        print(
            "epoch: %d, val boundary loss: %.4f, "
            "val lumen dice: %.4f, val wall dice: %.4f"
            % (epoch, val_loss, val_lumen_dice, val_wall_dice)
        )

        training_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_lumen_dice": val_lumen_dice,
            "val_wall_dice": val_wall_dice,
        }
        torch.save(model.state_dict(), save_dir / "wingsnet_latest.pth")
        torch.save(training_checkpoint, save_dir / "wingsnet_latest_checkpoint.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "wingsnet_best.pth")
            torch.save(training_checkpoint, save_dir / "wingsnet_best_checkpoint.pth")


def validation(model, valid_loader, device):
    model.eval()
    val_loss_sum = 0.0
    val_lumen_dice_sum = 0.0
    val_wall_dice_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for batch in valid_loader:
            image, target = unpack_batch(batch)
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred_en, pred_de = model(image)
            loss_en = boundary_aware_loss(pred_en, target)
            loss_de = boundary_aware_loss(pred_de, target)
            loss = loss_de + 0.5 * loss_en

            pred_de_prob = torch.sigmoid(pred_de)
            lumen_dice, wall_dice = channel_dice(pred_de_prob, target)

            current_batch_size = image.shape[0]
            val_loss_sum += loss.item() * current_batch_size
            val_lumen_dice_sum += lumen_dice.item() * current_batch_size
            val_wall_dice_sum += wall_dice.item() * current_batch_size
            val_count += current_batch_size

    return (
        val_loss_sum / max(val_count, 1),
        val_lumen_dice_sum / max(val_count, 1),
        val_wall_dice_sum / max(val_count, 1),
    )


if __name__ == "__main__":
    train()