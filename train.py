from pathlib import Path
import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from WingsNet import WingsNet
from airrc_dataset import AirRCPatchDataset


# Boundary-aware loss settings. These can be tuned after a short validation run.
WALL_POSITIVE_WEIGHT = 2.0
BOUNDARY_WEIGHT = 5.0
BOUNDARY_KERNEL_SIZE = 3
DISTAL_LUMEN_WEIGHT = 3.0
DISTAL_KERNEL_SIZE = 7
DISTAL_OCCUPANCY_THRESHOLD = 0.12
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
CLDICE_WEIGHT = 0.5
CLDICE_ITERATIONS = 8


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


def tversky_loss(pred, target, weight=None, alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA):
    smooth = 1.0
    dims = (0, 2, 3, 4)

    if weight is None:
        weight = torch.ones_like(target)

    true_positive = (weight * pred * target).sum(dim=dims)
    false_positive = (weight * pred * (1.0 - target)).sum(dim=dims)
    false_negative = (weight * (1.0 - pred) * target).sum(dim=dims)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - score.mean()


def soft_erode3d(volume):
    return -F.max_pool3d(-volume, kernel_size=3, stride=1, padding=1)


def soft_dilate3d(volume):
    return F.max_pool3d(volume, kernel_size=3, stride=1, padding=1)


def soft_open3d(volume):
    return soft_dilate3d(soft_erode3d(volume))


def soft_skeletonize3d(volume, iterations=CLDICE_ITERATIONS):
    volume = volume.clamp(0.0, 1.0)
    opened = soft_open3d(volume)
    skeleton = F.relu(volume - opened)

    eroded = volume
    for _ in range(iterations):
        eroded = soft_erode3d(eroded)
        opened = soft_open3d(eroded)
        delta = F.relu(eroded - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton.clamp(0.0, 1.0)


def cldice_loss(pred_lumen, target_lumen, iterations=CLDICE_ITERATIONS):
    smooth = 1.0
    dims = (1, 2, 3, 4)

    pred_lumen = pred_lumen.clamp(0.0, 1.0)
    target_lumen = target_lumen.clamp(0.0, 1.0)
    pred_skeleton = soft_skeletonize3d(pred_lumen, iterations=iterations)
    target_skeleton = soft_skeletonize3d(target_lumen, iterations=iterations)

    topology_precision = (
        (pred_skeleton * target_lumen).sum(dim=dims) + smooth
    ) / (pred_skeleton.sum(dim=dims) + smooth)
    topology_sensitivity = (
        (target_skeleton * pred_lumen).sum(dim=dims) + smooth
    ) / (target_skeleton.sum(dim=dims) + smooth)
    cldice = (
        2.0 * topology_precision * topology_sensitivity
    ) / (topology_precision + topology_sensitivity + 1e-6)

    return 1.0 - cldice.mean()


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


def local_occupancy(mask, kernel_size=DISTAL_KERNEL_SIZE):
    padding = kernel_size // 2
    return F.avg_pool3d(mask, kernel_size, stride=1, padding=padding)


def distal_lumen_mask(target):
    lumen = target[:, 0:1].clamp(0.0, 1.0)
    occupancy = local_occupancy(lumen)
    return lumen * (occupancy <= DISTAL_OCCUPANCY_THRESHOLD).float()


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
    distal_lumen = distal_lumen_mask(target)

    weights = torch.ones_like(target)
    weights[:, 0:1] += BOUNDARY_WEIGHT * interface
    weights[:, 0:1] += DISTAL_LUMEN_WEIGHT * distal_lumen
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
    lumen_tversky = tversky_loss(
        probability[:, 0:1],
        target[:, 0:1],
        weight=weights[:, 0:1],
    )
    topology_loss = cldice_loss(probability[:, 0:1], target[:, 0:1])
    return weighted_bce + weighted_dice + 0.5 * lumen_tversky + CLDICE_WEIGHT * topology_loss


def distal_lumen_recall(pred, target, threshold=0.5):
    pred_mask = (pred[:, 0:1] >= threshold).float()
    distal_target = distal_lumen_mask(target)
    target_count = distal_target.sum()
    if float(target_count.item()) == 0.0:
        return torch.tensor(1.0, device=pred.device)
    covered = (pred_mask * distal_target).sum()
    return covered / target_count.clamp_min(1.0)


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


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def log(message, *values):
    print(message, *values, flush=True)


def load_resume_checkpoint(model, optimizer, resume_path, device, fine_tune=False, allow_partial=False):
    if resume_path is None:
        return 0, float("inf"), 0.0

    resume_path = Path(resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    start_epoch = 0
    best_val_loss = float("inf")
    best_distal_recall = 0.0

    if isinstance(checkpoint, dict):
        state_dict = None
        for key in ("state_dict", "model_state_dict", "net", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
        else:
            if "optimizer_state_dict" in checkpoint and not fine_tune:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if not fine_tune:
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
                best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
                best_distal_recall = float(checkpoint.get("val_distal_lumen_recall", best_distal_recall))
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=not allow_partial)
    if allow_partial:
        if missing:
            print("[WARN] Missing checkpoint keys:", len(missing))
        if unexpected:
            print("[WARN] Unexpected checkpoint keys:", len(unexpected))
    mode = "fine-tune weights only" if fine_tune else "resume training"
    log("resumed checkpoint:", resume_path)
    log("resume mode:", mode)
    log("start epoch:", start_epoch)
    return start_epoch, best_val_loss, best_distal_recall


def parse_args():
    data_root = Path(os.environ.get("AVB_DATA_ROOT", Path.home() / "AMS_Project" / "datasets_new"))
    parser = argparse.ArgumentParser(description="Train WingsNet with distal-airway-aware sampling/loss.")
    parser.add_argument(
        "--train-json",
        default=str(data_root / "airrc_patches" / "splits" / "train.json"),
    )
    parser.add_argument(
        "--val-json",
        default=str(data_root / "airrc_patches" / "splits" / "val.json"),
    )
    parser.add_argument("--save-dir", default="./saved_model")
    parser.add_argument("--resume", default=None, help="Optional checkpoint/state_dict to fine-tune from")
    parser.add_argument("--fine-tune", action="store_true", help="Load model weights but reset optimizer and epoch count")
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys. Use only for deliberate architecture migration.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Epochs to run; with --fine-tune this is additional epochs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--distal-lumen-weight", type=float, default=DISTAL_LUMEN_WEIGHT)
    parser.add_argument("--distal-occupancy-threshold", type=float, default=DISTAL_OCCUPANCY_THRESHOLD)
    parser.add_argument("--cldice-weight", type=float, default=CLDICE_WEIGHT)
    parser.add_argument("--cldice-iterations", type=int, default=CLDICE_ITERATIONS)
    parser.add_argument("--augment", action="store_true", help="Use topology-preserving CT augmentations for training patches")
    return parser.parse_args()


def configure_training(args):
    global DISTAL_LUMEN_WEIGHT
    global DISTAL_OCCUPANCY_THRESHOLD
    global CLDICE_WEIGHT
    global CLDICE_ITERATIONS

    DISTAL_LUMEN_WEIGHT = args.distal_lumen_weight
    DISTAL_OCCUPANCY_THRESHOLD = args.distal_occupancy_threshold
    CLDICE_WEIGHT = args.cldice_weight
    CLDICE_ITERATIONS = args.cldice_iterations


def train():
    args = parse_args()
    configure_training(args)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    log("device:", device)
    log("train json:", args.train_json)
    log("val json:", args.val_json)
    log("distal lumen weight:", DISTAL_LUMEN_WEIGHT)
    log("distal occupancy threshold:", DISTAL_OCCUPANCY_THRESHOLD)
    log("cldice weight:", CLDICE_WEIGHT)
    log("cldice iterations:", CLDICE_ITERATIONS)
    log("train augment:", args.augment)

    train_dataset = AirRCPatchDataset(args.train_json, augment=args.augment)
    valid_dataset = AirRCPatchDataset(args.val_json)
    log("train patches:", len(train_dataset))
    log("val patches:", len(valid_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = WingsNet(in_channel=1, n_classes=2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch, best_val_loss, best_distal_recall = load_resume_checkpoint(
        model,
        optimizer,
        args.resume,
        device,
        fine_tune=args.fine_tune,
        allow_partial=args.allow_partial_checkpoint,
    )

    end_epoch = start_epoch + args.epochs if args.fine_tune else args.epochs
    log("end epoch:", end_epoch)
    if start_epoch >= end_epoch:
        log(
            "[WARN] No epochs to run. Use --fine-tune for additional epochs "
            "or set --epochs greater than the resumed start epoch."
        )
        return

    for epoch in range(start_epoch, end_epoch):
        log("starting epoch:", epoch)
        model.train()
        train_loss_sum = 0.0
        train_lumen_dice_sum = 0.0
        train_wall_dice_sum = 0.0
        train_distal_recall_sum = 0.0
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
                distal_recall = distal_lumen_recall(pred_de_prob, target)

            current_batch_size = image.shape[0]
            train_loss_sum += loss.item() * current_batch_size
            train_lumen_dice_sum += lumen_dice.item() * current_batch_size
            train_wall_dice_sum += wall_dice.item() * current_batch_size
            train_distal_recall_sum += distal_recall.item() * current_batch_size
            train_count += current_batch_size

            if iteration % 10 == 0:
                print(
                    "epoch: %d, iter: %d/%d, distal-aware loss: %.4f, "
                    "lumen dice: %.4f, wall dice: %.4f, distal recall: %.4f"
                    % (
                        epoch,
                        iteration,
                        len(train_loader),
                        train_loss_sum / max(train_count, 1),
                        train_lumen_dice_sum / max(train_count, 1),
                        train_wall_dice_sum / max(train_count, 1),
                        train_distal_recall_sum / max(train_count, 1),
                    ),
                    flush=True,
                )

        val_loss, val_lumen_dice, val_wall_dice, val_distal_recall = validation(
            model=model,
            valid_loader=valid_loader,
            device=device,
        )

        log(
            "epoch: %d, val distal-aware loss: %.4f, "
            "val lumen dice: %.4f, val wall dice: %.4f, val distal recall: %.4f"
            % (epoch, val_loss, val_lumen_dice, val_wall_dice, val_distal_recall)
        )

        training_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_lumen_dice": val_lumen_dice,
            "val_wall_dice": val_wall_dice,
            "val_distal_lumen_recall": val_distal_recall,
            "settings": {
                "distal_lumen_weight": DISTAL_LUMEN_WEIGHT,
                "distal_kernel_size": DISTAL_KERNEL_SIZE,
                "distal_occupancy_threshold": DISTAL_OCCUPANCY_THRESHOLD,
                "tversky_alpha": TVERSKY_ALPHA,
                "tversky_beta": TVERSKY_BETA,
                "cldice_weight": CLDICE_WEIGHT,
                "cldice_iterations": CLDICE_ITERATIONS,
                "augment": args.augment,
            },
        }
        torch.save(model.state_dict(), save_dir / "wingsnet_latest.pth")
        torch.save(training_checkpoint, save_dir / "wingsnet_latest_checkpoint.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "wingsnet_best.pth")
            torch.save(training_checkpoint, save_dir / "wingsnet_best_checkpoint.pth")
        if val_distal_recall > best_distal_recall:
            best_distal_recall = val_distal_recall
            torch.save(model.state_dict(), save_dir / "wingsnet_best_distal.pth")
            torch.save(training_checkpoint, save_dir / "wingsnet_best_distal_checkpoint.pth")

        history_path = save_dir / "training_history.jsonl"
        history_record = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "val_lumen_dice": float(val_lumen_dice),
            "val_wall_dice": float(val_wall_dice),
            "val_distal_lumen_recall": float(val_distal_recall),
            "best_val_loss": float(best_val_loss),
            "best_distal_lumen_recall": float(best_distal_recall),
        }
        with open(history_path, "a") as f:
            f.write(json.dumps(history_record) + "\n")


def validation(model, valid_loader, device):
    model.eval()
    val_loss_sum = 0.0
    val_lumen_dice_sum = 0.0
    val_wall_dice_sum = 0.0
    val_distal_recall_sum = 0.0
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
            distal_recall = distal_lumen_recall(pred_de_prob, target)

            current_batch_size = image.shape[0]
            val_loss_sum += loss.item() * current_batch_size
            val_lumen_dice_sum += lumen_dice.item() * current_batch_size
            val_wall_dice_sum += wall_dice.item() * current_batch_size
            val_distal_recall_sum += distal_recall.item() * current_batch_size
            val_count += current_batch_size

    return (
        val_loss_sum / max(val_count, 1),
        val_lumen_dice_sum / max(val_count, 1),
        val_wall_dice_sum / max(val_count, 1),
        val_distal_recall_sum / max(val_count, 1),
    )


if __name__ == "__main__":
    train()
