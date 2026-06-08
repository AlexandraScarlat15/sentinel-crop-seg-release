from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from time import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.config_loader import apply_overrides
from src.training.dataset import SegmentationPatchDataset
from src.training.dataset_manifest import ManifestSegmentationDataset
from src.training.losses import CrossEntropyDiceLoss
from src.training.model_unet_4levels import UNet4LevelsGN


@dataclass(frozen=True)
class Config:
    experiment_name: str

    manifest_csv: Path | None
    train_root: Path
    val_root: Path
    test_root: Path
    out_dir: Path
    resume_ckpt: Path | None

    train_limit: int
    val_limit: int
    in_channels: int
    num_classes: int
    band_indices: tuple[int, ...] | None
    normalize_mode: str
    scale_divisor: float
    sample_format: str

    base: int
    dropout_bottleneck: float
    dropout_decoder: float
    up_mode: str

    dice_weight: float
    ignore_index: int | None
    class_weights: tuple[float, ...] | None
    dice_ignore_background: bool

    augment: bool
    augment_prob: float

    loader_num_workers: int
    loader_pin_memory: bool
    loader_persistent_workers: bool
    loader_prefetch_factor: int | None

    epochs: int
    batch_size: int
    lr: float
    seed: int

    lr_patience: int
    lr_factor: float
    lr_min: float

    early_stop_patience: int
    min_delta: float


def load_config(
    config_path: str | Path,
    overrides: list[str] | tuple[str, ...] | None = None,
) -> Config:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    apply_overrides(raw, overrides)

    band_indices_raw = raw["data"].get("band_indices")
    band_indices = None if band_indices_raw is None else tuple(int(x) for x in band_indices_raw)

    in_channels = int(raw["data"]["in_channels"])
    num_classes = int(raw["data"]["num_classes"])

    if band_indices is not None and len(band_indices) != in_channels:
        raise ValueError(
            "in_channels must match the number of selected band_indices "
            f"(got in_channels={in_channels}, band_indices={band_indices})"
        )

    sample_format = str(raw["data"].get("sample_format", "npy")).lower().strip()

    loss_cfg = raw.get("loss", {})

    ignore_index_raw = loss_cfg.get("ignore_index", None)
    ignore_index = None if ignore_index_raw is None else int(ignore_index_raw)

    class_weights_raw = loss_cfg.get("class_weights", None)
    class_weights = None

    if class_weights_raw is not None:
        class_weights = tuple(float(x) for x in class_weights_raw)

        if len(class_weights) != num_classes:
            raise ValueError(
                "loss.class_weights must have exactly num_classes values "
                f"(got {len(class_weights)}, expected {num_classes})"
            )

    loader_cfg = raw.get("loader", {}) or {}
    loader_num_workers = int(loader_cfg.get("num_workers", 0))
    loader_pin_memory = bool(loader_cfg.get("pin_memory", True))
    loader_persistent_workers = bool(loader_cfg.get("persistent_workers", False))
    loader_prefetch_factor_raw = loader_cfg.get("prefetch_factor", None)
    loader_prefetch_factor = (
        None if loader_prefetch_factor_raw is None else int(loader_prefetch_factor_raw)
    )

    if loader_num_workers < 0:
        raise ValueError(f"loader.num_workers must be >= 0, got {loader_num_workers}")

    if loader_prefetch_factor is not None and loader_prefetch_factor <= 0:
        raise ValueError(
            f"loader.prefetch_factor must be positive when set, got {loader_prefetch_factor}"
        )

    if loader_num_workers == 0 and loader_persistent_workers:
        raise ValueError("loader.persistent_workers requires loader.num_workers > 0")

    if loader_num_workers == 0 and loader_prefetch_factor is not None:
        raise ValueError("loader.prefetch_factor requires loader.num_workers > 0")

    return Config(
        experiment_name=str(raw["experiment"]["name"]),

        manifest_csv=Path(raw["paths"]["manifest_csv"]) if raw["paths"].get("manifest_csv") else None,
        train_root=Path(raw["paths"]["train_root"]),
        val_root=Path(raw["paths"]["val_root"]),
        test_root=Path(raw["paths"]["test_root"]),
        out_dir=Path(raw["paths"]["out_dir"]),
        resume_ckpt=Path(raw["paths"]["resume_ckpt"]) if raw["paths"].get("resume_ckpt") else None,

        train_limit=int(raw["data"].get("train_limit", 0)),
        val_limit=int(raw["data"].get("val_limit", 0)),
        in_channels=in_channels,
        num_classes=num_classes,
        band_indices=band_indices,
        normalize_mode=str(raw["data"].get("normalize_mode", "none")),
        scale_divisor=float(raw["data"].get("scale_divisor", 10000.0)),
        sample_format=sample_format,

        base=int(raw["model"]["base"]),
        dropout_bottleneck=float(raw["model"].get("dropout_bottleneck", 0.0)),
        dropout_decoder=float(raw["model"].get("dropout_decoder", 0.0)),
        up_mode=str(raw["model"].get("up_mode", "bilinear")),

        dice_weight=float(loss_cfg["dice_weight"]),
        ignore_index=ignore_index,
        class_weights=class_weights,
        dice_ignore_background=bool(loss_cfg.get("dice_ignore_background", False)),

        augment=bool(raw["augmentation"].get("augment", False)),
        augment_prob=float(raw["augmentation"].get("augment_prob", 1.0)),

        loader_num_workers=loader_num_workers,
        loader_pin_memory=loader_pin_memory,
        loader_persistent_workers=loader_persistent_workers,
        loader_prefetch_factor=loader_prefetch_factor,

        epochs=int(raw["training"]["epochs"]),
        batch_size=int(raw["training"]["batch_size"]),
        lr=float(raw["training"]["lr"]),
        seed=int(raw["training"]["seed"]),

        lr_patience=int(raw["scheduler"]["lr_patience"]),
        lr_factor=float(raw["scheduler"]["lr_factor"]),
        lr_min=float(raw["scheduler"]["lr_min"]),

        early_stop_patience=int(raw["early_stopping"]["early_stop_patience"]),
        min_delta=float(raw["early_stopping"]["min_delta"]),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_confusion_matrix(
    cm: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> None:
    pred = torch.argmax(logits, dim=1)
    pred = pred.view(-1).to(torch.int64)
    tgt = y.view(-1).to(torch.int64)

    valid = (tgt >= 0) & (tgt < num_classes)

    if ignore_index is not None:
        valid = valid & (tgt != int(ignore_index))

    pred = pred[valid]
    tgt = tgt[valid]

    if tgt.numel() == 0:
        return

    idx = tgt * num_classes + pred

    cm += torch.bincount(
        idx,
        minlength=num_classes * num_classes,
    ).view(num_classes, num_classes)


def compute_iou_from_cm(cm: torch.Tensor) -> torch.Tensor:
    cm = cm.to(torch.float32)

    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    denom = tp + fp + fn

    return torch.where(denom > 0, tp / denom, torch.zeros_like(denom))


def compute_pixel_accuracy_from_cm(cm: torch.Tensor) -> float:
    total = cm.sum().item()

    if total <= 0:
        return 0.0

    correct = torch.diag(cm).sum().item()

    return float(correct / total)


def run_train_epoch(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()

    total_loss = 0.0
    n_batches = 0

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def run_val_epoch(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    num_classes: int,
    ignore_index: int | None = None,
) -> tuple[float, torch.Tensor, float]:
    model.eval()

    total_loss = 0.0
    n_batches = 0

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        n_batches += 1

        update_confusion_matrix(
            cm=cm,
            logits=logits,
            y=y,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    avg_loss = total_loss / max(n_batches, 1)
    iou = compute_iou_from_cm(cm).detach().cpu()
    pixel_acc = compute_pixel_accuracy_from_cm(cm)

    return avg_loss, iou, pixel_acc


def save_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    cfg: Config,
    epoch: int,
    miou: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "val_mIoU": miou,
        },
        path,
    )


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: Config,
    epoch: int,
    best_epoch: int,
    best_miou: float,
    epochs_no_improve: int,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": asdict(cfg),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_miou": best_miou,
            "epochs_no_improve": epochs_no_improve,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def _as_cpu_uint8_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device="cpu", dtype=torch.uint8).contiguous()

    return torch.tensor(x, dtype=torch.uint8)


def restore_rng_state(rng_state: dict | None) -> None:
    if rng_state is None:
        return

    random.setstate(rng_state["python"])

    np_state = rng_state["numpy"]

    if isinstance(np_state, list):
        np_state = tuple(np_state)

    if isinstance(np_state, tuple) and len(np_state) >= 2 and not isinstance(np_state[1], np.ndarray):
        np_state = (np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:])

    np.random.set_state(np_state)

    torch_state = _as_cpu_uint8_tensor(rng_state["torch"])
    torch.set_rng_state(torch_state)

    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        cuda_states = [_as_cpu_uint8_tensor(s) for s in rng_state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a U-Net model for Sentinel-2 crop segmentation."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML config, e.g. config/train_unet.yaml",
    )

    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override training configuration values using dotted keys. "
            "Example: --override training.epochs=1. "
            "Can be used multiple times."
        ),
    )

    return parser.parse_args()


def load_existing_history(metrics_path: Path) -> list[dict]:
    records: list[dict] = []

    if not metrics_path.exists():
        return records

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def save_training_metrics_csv(csv_path: Path, history_records: list[dict]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_mIoU",
                "val_pixel_acc",
                "lr",
                "time_s",
                "val_iou_per_class_json",
            ]
        )

        for rec in history_records:
            writer.writerow(
                [
                    rec["epoch"],
                    rec["train_loss"],
                    rec["val_loss"],
                    rec["val_mIoU"],
                    rec["val_pixel_acc"],
                    rec["lr"],
                    rec["time_s"],
                    json.dumps(rec["val_iou_per_class"]),
                ]
            )


def count_model_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return int(total_params), int(trainable_params)


def file_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None

    return int(path.stat().st_size)


def json_safe_config(cfg: Config) -> dict:
    result = {}

    for key, value in asdict(cfg).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value

    return result


def build_dataloader_kwargs(cfg: Config, device: torch.device) -> dict:
    kwargs = {
        "num_workers": cfg.loader_num_workers,
        "pin_memory": bool(cfg.loader_pin_memory and device.type == "cuda"),
    }

    if cfg.loader_num_workers > 0:
        kwargs["persistent_workers"] = cfg.loader_persistent_workers

        if cfg.loader_prefetch_factor is not None:
            kwargs["prefetch_factor"] = cfg.loader_prefetch_factor

    return kwargs


def main() -> None:
    args = parse_args()

    if args.override:
        print(f"[TRAIN] config_overrides={len(args.override)}")

    cfg = load_config(args.config, overrides=args.override)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg_path = cfg.out_dir / "resolved_config.json"

    with resolved_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe_config(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[TRAIN] device:", device)

    if device.type == "cuda":
        print("[TRAIN] gpu:", torch.cuda.get_device_name(0))

    set_seed(cfg.seed)

    if cfg.manifest_csv is not None:
        ds_train = ManifestSegmentationDataset(
            manifest_csv=cfg.manifest_csv,
            split="train",
            limit=cfg.train_limit,
            shuffle=True,
            seed=cfg.seed,
            augment=cfg.augment,
            augment_prob=cfg.augment_prob,
            band_indices=cfg.band_indices,
            normalize_mode=cfg.normalize_mode,
            scale_divisor=cfg.scale_divisor,
        )

        ds_val = ManifestSegmentationDataset(
            manifest_csv=cfg.manifest_csv,
            split="val",
            limit=cfg.val_limit,
            shuffle=False,
            seed=cfg.seed,
            augment=False,
            band_indices=cfg.band_indices,
            normalize_mode=cfg.normalize_mode,
            scale_divisor=cfg.scale_divisor,
        )

        print("[TRAIN] data source: manifest")
        print("[TRAIN] manifest_csv:", cfg.manifest_csv)
        print("[TRAIN] sample_format: inferred from manifest paths")
    else:
        ds_train = SegmentationPatchDataset(
            cfg.train_root,
            limit=cfg.train_limit,
            shuffle=True,
            seed=cfg.seed,
            augment=cfg.augment,
            augment_prob=cfg.augment_prob,
            band_indices=cfg.band_indices,
            normalize_mode=cfg.normalize_mode,
            scale_divisor=cfg.scale_divisor,
            sample_format=cfg.sample_format,
        )

        ds_val = SegmentationPatchDataset(
            cfg.val_root,
            limit=cfg.val_limit,
            shuffle=False,
            seed=cfg.seed,
            augment=False,
            band_indices=cfg.band_indices,
            normalize_mode=cfg.normalize_mode,
            scale_divisor=cfg.scale_divisor,
            sample_format=cfg.sample_format,
        )

        print("[TRAIN] data source: roots")
        print("[TRAIN] train_root:", cfg.train_root)
        print("[TRAIN] val_root:", cfg.val_root)
        print("[TRAIN] sample_format:", cfg.sample_format)

    print("[TRAIN] dataset train:", ds_train.describe())
    if hasattr(ds_train, "describe_storage"):
        print("[TRAIN] dataset train storage:", ds_train.describe_storage())

    print("[TRAIN] dataset val:", ds_val.describe())
    if hasattr(ds_val, "describe_storage"):
        print("[TRAIN] dataset val storage:", ds_val.describe_storage())

    dataloader_kwargs = build_dataloader_kwargs(cfg, device)
    print("[TRAIN] dataloader:", dataloader_kwargs)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    model = UNet4LevelsGN(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base=cfg.base,
        dropout_bottleneck=cfg.dropout_bottleneck,
        dropout_decoder=cfg.dropout_decoder,
        up_mode=cfg.up_mode,
    ).to(device)

    criterion = CrossEntropyDiceLoss(
        cfg.num_classes,
        dice_weight=cfg.dice_weight,
        ignore_index=cfg.ignore_index,
        class_weights=cfg.class_weights,
        dice_ignore_background=cfg.dice_ignore_background,
    )

    print("[TRAIN] loss dice_weight:", cfg.dice_weight)
    print("[TRAIN] loss ignore_index:", cfg.ignore_index)
    print("[TRAIN] loss dice_ignore_background:", cfg.dice_ignore_background)
    print("[TRAIN] loss class_weights:", cfg.class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.lr_min,
    )

    start_epoch = 1
    best_miou = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    if cfg.resume_ckpt is not None:
        if not cfg.resume_ckpt.exists():
            raise FileNotFoundError(f"resume_ckpt not found: {cfg.resume_ckpt}")

        ckpt = torch.load(cfg.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])

            if ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_epoch = int(ckpt.get("best_epoch", 0))
            best_miou = float(ckpt.get("best_miou", ckpt.get("val_mIoU", -1.0)))
            epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))

            restore_rng_state(ckpt.get("rng_state"))

            print("[TRAIN] resumed full state from:", cfg.resume_ckpt)
            print("[TRAIN] resume start_epoch:", start_epoch)
            print("[TRAIN] resumed best_miou:", best_miou)
            print("[TRAIN] resumed best_epoch:", best_epoch)
            print("[TRAIN] resumed epochs_no_improve:", epochs_no_improve)
        else:
            print("[TRAIN] warm start from model-only checkpoint:", cfg.resume_ckpt)

    print(
        f"[TRAIN] experiment={cfg.experiment_name} "
        f"train_samples={len(ds_train)} val_samples={len(ds_val)} "
        f"batch={cfg.batch_size} epochs={cfg.epochs} lr={cfg.lr} "
        f"base={cfg.base} dice_weight={cfg.dice_weight} "
        f"normalize_mode={cfg.normalize_mode} "
        f"sample_format={cfg.sample_format} "
        f"band_indices={cfg.band_indices}"
    )

    metrics_path = cfg.out_dir / "metrics.jsonl"
    metrics_mode = "a" if start_epoch > 1 and metrics_path.exists() else "w"
    history_records = load_existing_history(metrics_path) if metrics_mode == "a" else []

    last_epoch_completed = 0
    last_epoch_miou = -1.0
    early_stopped = False
    stop_reason = "max_epochs_reached"

    overall_t0 = time()

    with metrics_path.open(metrics_mode, encoding="utf-8") as f:
        for epoch in range(start_epoch, cfg.epochs + 1):
            t0 = time()

            train_loss = run_train_epoch(
                model=model,
                dl=dl_train,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
            )

            val_loss, val_iou, val_pixel_acc = run_val_epoch(
                model=model,
                dl=dl_val,
                device=device,
                criterion=criterion,
                num_classes=cfg.num_classes,
                ignore_index=cfg.ignore_index,
            )

            dt = time() - t0
            miou = float(val_iou.mean().item())
            lr_now = float(optimizer.param_groups[0]["lr"])

            last_epoch_completed = epoch
            last_epoch_miou = miou

            scheduler.step(miou)

            rec = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mIoU": miou,
                "val_pixel_acc": val_pixel_acc,
                "val_iou_per_class": [float(x) for x in val_iou.tolist()],
                "lr": lr_now,
                "time_s": dt,
            }

            history_records.append(rec)

            f.write(json.dumps(rec) + "\n")
            f.flush()

            print(
                f"[TRAIN] epoch {epoch:03d}/{cfg.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mIoU={miou:.4f} val_pixacc={val_pixel_acc:.4f} "
                f"lr={lr_now:.6g} time_s={dt:.1f}"
            )

            improved = miou > best_miou + cfg.min_delta

            if improved:
                best_miou = miou
                best_epoch = epoch
                epochs_no_improve = 0

                save_model_checkpoint(
                    cfg.out_dir / "best.pt",
                    model,
                    cfg,
                    epoch,
                    miou,
                )

                print(f"[TRAIN] best updated: epoch={epoch} val_mIoU={miou:.4f}")
            else:
                epochs_no_improve += 1

            save_training_state(
                cfg.out_dir / "last_state.pt",
                model,
                optimizer,
                scheduler,
                cfg,
                epoch,
                best_epoch,
                best_miou,
                epochs_no_improve,
            )

            if epochs_no_improve >= cfg.early_stop_patience:
                early_stopped = True
                stop_reason = "early_stopping"

                print(
                    f"[TRAIN] early stopping: no improvement for {cfg.early_stop_patience} epochs "
                    f"(best epoch={best_epoch}, best mIoU={best_miou:.4f})"
                )

                break

    if last_epoch_completed > 0:
        save_model_checkpoint(
            cfg.out_dir / "last.pt",
            model,
            cfg,
            last_epoch_completed,
            last_epoch_miou,
        )

    total_params, trainable_params = count_model_parameters(model)

    session_elapsed_seconds = time() - overall_t0
    total_recorded_training_seconds = float(sum(rec["time_s"] for rec in history_records))
    final_lr = float(optimizer.param_groups[0]["lr"])

    training_metrics_csv_path = cfg.out_dir / "training_metrics.csv"
    save_training_metrics_csv(training_metrics_csv_path, history_records)

    training_summary = {
        "experiment_name": cfg.experiment_name,
        "data_source": "manifest" if cfg.manifest_csv is not None else "roots",
        "manifest_csv": str(cfg.manifest_csv) if cfg.manifest_csv is not None else None,
        "sample_format": cfg.sample_format if cfg.manifest_csv is None else "manifest_paths",
        "train_samples": len(ds_train),
        "val_samples": len(ds_val),
        "start_epoch": start_epoch,
        "last_epoch_completed": last_epoch_completed,
        "best_epoch": best_epoch,
        "best_val_mIoU": best_miou,
        "last_val_mIoU": last_epoch_miou,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "epochs_no_improve_at_end": epochs_no_improve,
        "initial_lr": cfg.lr,
        "final_lr": final_lr,
        "batch_size": cfg.batch_size,
        "requested_epochs": cfg.epochs,
        "dice_weight": cfg.dice_weight,
        "ignore_index": cfg.ignore_index,
        "dice_ignore_background": cfg.dice_ignore_background,
        "class_weights": cfg.class_weights,
        "session_elapsed_seconds": session_elapsed_seconds,
        "total_recorded_training_seconds": total_recorded_training_seconds,
        "num_history_records": len(history_records),
        "best_checkpoint": str(cfg.out_dir / "best.pt"),
        "last_checkpoint": str(cfg.out_dir / "last.pt"),
        "last_state_checkpoint": str(cfg.out_dir / "last_state.pt"),
        "metrics_jsonl": str(metrics_path),
        "training_metrics_csv": str(training_metrics_csv_path),
        "resolved_config": str(resolved_cfg_path),
    }

    training_summary_path = cfg.out_dir / "training_summary.json"

    with training_summary_path.open("w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    model_info = {
        "experiment_name": cfg.experiment_name,
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "in_channels": cfg.in_channels,
        "num_classes": cfg.num_classes,
        "base": cfg.base,
        "dropout_bottleneck": cfg.dropout_bottleneck,
        "dropout_decoder": cfg.dropout_decoder,
        "up_mode": cfg.up_mode,
        "best_checkpoint_bytes": file_size_bytes(cfg.out_dir / "best.pt"),
        "last_checkpoint_bytes": file_size_bytes(cfg.out_dir / "last.pt"),
        "last_state_checkpoint_bytes": file_size_bytes(cfg.out_dir / "last_state.pt"),
    }

    model_info_path = cfg.out_dir / "model_info.json"

    with model_info_path.open("w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print("[TRAIN] saved last model checkpoint:", cfg.out_dir / "last.pt")
    print("[TRAIN] saved full training state:", cfg.out_dir / "last_state.pt")
    print("[TRAIN] saved training metrics CSV:", training_metrics_csv_path)
    print("[TRAIN] saved training summary:", training_summary_path)
    print("[TRAIN] saved model info:", model_info_path)
    print("[TRAIN] metrics:", metrics_path)
    print("[TRAIN] resolved config:", resolved_cfg_path)


if __name__ == "__main__":
    main()