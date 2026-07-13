import os
import argparse
import yaml
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from dataset import Dataset
from model.semseg.dpt import DPT
from util.inference import slide_inference
from util.classes import CLASSES

ROOT = Path(__file__).resolve().parent

_PATH_KEYS = (
    "train_root",
    "train_mask_root",
    "train_full_mask_root",
    "val_root",
    "val_mask_root",
    "save_root",
    "data_root",
)


def resolve_path(path):
    if not path:
        return path
    p = Path(path)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return str(p)


def resolve_cfg_paths(cfg):
    for key in _PATH_KEYS:
        value = cfg.get(key)
        if value:
            cfg[key] = resolve_path(value)
    return cfg


def load_model_from_ckpt(cfg, ckpt_path, num_classes):
    model_configs = {
        "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
        "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
        "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    backbone_type = cfg["backbone"].split("_")[-1]
    net = DPT(**{**model_configs[backbone_type], "nclass": num_classes})

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("model.", "module.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_state[nk] = v

    missing, unexpected = net.load_state_dict(new_state, strict=False)
    print(f"[INFO] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    return net


def get_sample_indices(val_set, num_samples=30, sample_ids_file=None, save_ids_file=None, seed=42):
    total_size = len(val_set)
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(val_set.ids)}

    if sample_ids_file and os.path.exists(sample_ids_file):
        print(f"[INFO] Loading image IDs from: {sample_ids_file}")
        with open(sample_ids_file, "r") as f:
            target_ids = json.load(f)

        if isinstance(target_ids, list) and target_ids and str(target_ids[0]).startswith("sample_"):
            indices = sorted({int(str(name).split("_")[-1]) for name in target_ids if 0 <= int(str(name).split("_")[-1]) < total_size})
            image_ids = [f"sample_{idx:04d}" for idx in indices]
        else:
            target_ids_set = {str(x) for x in target_ids}
            indices = [id_to_idx[sample_id] for sample_id in target_ids_set if sample_id in id_to_idx]
            image_ids = [val_set.ids[idx] for idx in indices]

        if not indices:
            print("[WARN] No matching image IDs found; falling back to random sampling")
        else:
            return indices, image_ids

    num_samples = min(num_samples, total_size)
    random.seed(seed)
    indices = sorted(random.sample(range(total_size), num_samples))
    image_ids = [val_set.ids[idx] for idx in indices]

    if save_ids_file:
        os.makedirs(os.path.dirname(save_ids_file) or ".", exist_ok=True)
        with open(save_ids_file, "w") as f:
            json.dump(image_ids, f, indent=2)
        print(f"[INFO] Saved selected image IDs to: {save_ids_file}")

    return indices, image_ids


@torch.no_grad()
def evaluate_and_visualize(
    cfg,
    ckpt_path,
    save_root,
    num_workers=2,
    samples=0,
    visualize_all=False,
    num_samples=30,
    sample_ids_file=None,
    save_ids_file=None,
    seed=42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["dataset"]
    num_classes = cfg["nclass"]

    val_set = Dataset(
        mode="val",
        img_dir=cfg["val_root"],
        mask_dir=cfg["val_mask_root"],
    )

    if visualize_all:
        indices = list(range(len(val_set)))
        image_ids = val_set.ids
        print(f"[INFO] Visualizing all images: {len(indices)}")
    else:
        indices, image_ids = get_sample_indices(
            val_set,
            num_samples=num_samples,
            sample_ids_file=sample_ids_file,
            save_ids_file=save_ids_file,
            seed=seed,
        )
        print(f"[INFO] Visualizing {len(indices)} images")

    val_loader = DataLoader(
        Subset(val_set, indices),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    net = load_model_from_ckpt(cfg, ckpt_path, num_classes).to(device)
    net.eval()

    miou_metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        average="none",
        ignore_index=255,
        dist_sync_on_step=False,
    ).to(device)

    viz_dir = os.path.join(save_root, "best_ckpt_vis")
    os.makedirs(viz_dir, exist_ok=True)
    saved = 0
    crop_size = (518, 518)
    stride = (448, 448)

    pbar = tqdm(val_loader, desc=f"Evaluating {dataset_name}", unit="img", ncols=120)
    for batch_idx, batch in enumerate(pbar):
        img, mask, sample_id = batch
        img = img.to(device)
        mask = mask.to(device).long()
        h, w = img.shape[2], img.shape[3]

        logits, _ = slide_inference(net, img, crop_size=crop_size, stride=stride)
        logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=True)
        preds = logits.argmax(dim=1)
        miou_metric.update(preds, mask)

        if samples > 0 and saved >= samples:
            continue

        vis_img = img[0].detach().cpu().numpy().transpose(1, 2, 0)
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-6)
        gt = mask[0].detach().cpu().numpy()
        pd = preds[0].detach().cpu().numpy()

        error_mask = (gt != 255) & (pd != gt)
        error_img = np.zeros_like(gt, dtype=np.float32)
        error_img[error_mask] = 1.0
        error_img[gt == 255] = 0.5

        plt.figure(figsize=(16, 4))
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(vis_img)
        ax1.set_title("Image")
        ax1.axis("off")
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(gt, vmin=0, vmax=num_classes - 1, cmap="gray")
        ax2.set_title("GT")
        ax2.axis("off")
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(pd, vmin=0, vmax=num_classes - 1, cmap="gray")
        ax3.set_title("Pred")
        ax3.axis("off")
        ax4 = plt.subplot(1, 4, 4)
        ax4.imshow(error_img, vmin=0, vmax=1, cmap="gray")
        ax4.set_title("Error")
        ax4.axis("off")

        img_id = image_ids[batch_idx] if batch_idx < len(image_ids) else sample_id[0]
        out_path = os.path.join(viz_dir, f"{img_id}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved += 1

        pbar.set_postfix(saved=saved, progress=f"{batch_idx + 1}/{len(indices)}")

    iou_class = miou_metric.compute().detach().cpu().numpy() * 100.0
    miou_metric.reset()
    mIoU = float(np.mean(iou_class))

    print("=" * 60)
    print(f"[INFO] Evaluated ckpt: {ckpt_path}")
    print(f"[INFO] Val mIoU: {mIoU:.2f}%")
    for i, iou in enumerate(iou_class):
        cname = CLASSES.get(dataset_name, [f"class_{i}"] * num_classes)[i]
        print(f"{cname:<20} {iou:>7.2f}")

    class_names = [
        CLASSES[dataset_name][i] if dataset_name in CLASSES and i < len(CLASSES[dataset_name]) else f"class_{i}"
        for i in range(len(iou_class))
    ]
    plt.figure(figsize=(max(8, len(iou_class) * 0.6), 4.5))
    x = np.arange(len(iou_class))
    plt.bar(x, iou_class)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("IoU (%)")
    plt.title(f"Per-class IoU (mIoU={mIoU:.2f}%)")
    plt.tight_layout()
    bar_path = os.path.join(save_root, "best_ckpt_per_class_iou.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()

    print(f"[PLOT] Per-class IoU saved to: {bar_path}")
    print(f"[VIZ] {saved} images saved to: {viz_dir}")
    print("=" * 60)


def find_ckpt_file(ckpt_dir):
    if os.path.isfile(ckpt_dir):
        return ckpt_dir
    if os.path.isdir(ckpt_dir):
        for fname in os.listdir(ckpt_dir):
            if fname.endswith(".ckpt"):
                return os.path.join(ckpt_dir, fname)
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize validation predictions from a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--samples", type=int, default=0, help="Max saved triplets (0 = save all selected)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--visualize_all", action="store_true")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--sample_ids_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    save_root = resolve_path(args.save_root)
    ckpt_path = resolve_path(args.ckpt_path)
    if os.path.isdir(ckpt_path):
        ckpt_path = find_ckpt_file(ckpt_path)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    sample_ids_file = resolve_path(args.sample_ids_file) if args.sample_ids_file else None
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_cfg_paths(cfg)
    cfg["save_root"] = save_root
    os.makedirs(save_root, exist_ok=True)

    save_ids_file = os.path.join(save_root, "sample_image_ids.json") if not args.visualize_all else None
    evaluate_and_visualize(
        cfg,
        ckpt_path,
        save_root,
        num_workers=args.num_workers,
        samples=args.samples,
        visualize_all=args.visualize_all,
        num_samples=args.num_samples,
        sample_ids_file=sample_ids_file,
        save_ids_file=save_ids_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
