import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"

import argparse
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import pprint
import random
import yaml

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import Dataset
from model.semseg.dpt import DPT
from model.tree_energy import MinimumSpanningTree, TreeFilter2D
from model.reco_branch import ReCoBranchDPT
from util.boundary import per_class_boundary_iou
from util.classes import CLASSES
from util.inference import slide_inference
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import init_log
from util.reco_module_list import compute_reco_loss, label_onehot


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"[INFO] Random seed set as {seed}")


def resolve_tacos_modules(cfg):
    """Normalize TACoS paper modules and keep legacy aliases.

    Paper modules:
      UWSD, TER, ARCL(= region + boundary, parallel)
    Legacy:
      consistency <-> uwsd, tel <-> ter, arcl/reco <-> region
    """
    uwsd = dict(cfg.get("uwsd") or cfg.get("consistency") or {})
    # λ_c: prefer weight; accept legacy progressive max_weight
    if "weight" not in uwsd and "max_weight" in uwsd:
        uwsd["weight"] = uwsd["max_weight"]
    cfg["uwsd"] = uwsd
    cfg["consistency"] = uwsd

    ter = dict(cfg.get("ter") or cfg.get("tel") or {})
    if "lambda_ter" not in ter and "lambda_tel" in ter:
        ter["lambda_ter"] = ter["lambda_tel"]
    if "lambda_tel" not in ter and "lambda_ter" in ter:
        ter["lambda_tel"] = ter["lambda_ter"]
    cfg["ter"] = ter
    cfg["tel"] = ter

    # ARCL.L_Region — prefer region; accept legacy arcl/reco
    region = dict(cfg.get("region") or cfg.get("arcl") or cfg.get("reco") or {})
    cfg["region"] = region
    cfg["arcl"] = region  # legacy alias
    cfg["reco"] = region

    # ARCL.L_Boundary — parallel to region
    boundary = dict(cfg.get("boundary") or {})
    if not boundary and ("use_boundary_loss" in region or "boundary_weight" in region):
        nested = dict(region.get("boundary") or {})
        boundary["enable"] = nested.get(
            "enable", region.get("use_boundary_loss", False)
        )
        boundary["weight"] = nested.get(
            "weight", region.get("boundary_weight", 0.05)
        )
    boundary.setdefault("enable", False)
    boundary.setdefault("weight", 0.05)
    cfg["boundary"] = boundary

    # L_sup
    cfg["sup"] = dict(cfg.get("sup") or {"enable": True})
    cfg["sup"].setdefault("enable", True)

    # OHEM kwargs (§4.2 mentions OHEM + ignore 255; thresh/min_kept are lib defaults)
    if "criterion" not in cfg:
        cfg["criterion"] = {
            "kwargs": {
                "ignore_index": cfg.get("ignore_index", 255),
                "thresh": 0.7,
                "min_kept": 1024,
            }
        }
    else:
        kwargs = cfg["criterion"].setdefault("kwargs", {})
        kwargs.setdefault("ignore_index", cfg.get("ignore_index", 255))
        kwargs.setdefault("thresh", 0.7)
        kwargs.setdefault("min_kept", 1024)

    return cfg


_PATH_KEYS = (
    "train_root",
    "train_mask_root",
    "train_full_mask_root",
    "val_root",
    "val_mask_root",
    "save_root",
    "data_root",
)


def resolve_cfg_paths(cfg, base_dir=None):
    """Resolve relative data/output paths against the project root (mynet/)."""
    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    for key in _PATH_KEYS:
        value = cfg.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = (root / path).resolve()
        cfg[key] = str(path)
    return cfg


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


parser = argparse.ArgumentParser(description="TACoS: Tree-based Asymmetric Contrast Segmentation")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming")


class DummyValDataset(Dataset):
    def __init__(self, n, H=512, W=512):
        self.n = n
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.zeros(3, self.H, self.W, dtype=torch.float32)
        mask = torch.full((self.H, self.W), 255, dtype=torch.long)
        return img, mask, {"dummy": True}


class DataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        gpus = cfg.get("gpus", 1)
        self.num_workers = min(2, max(1, gpus * 2))

    def setup(self, stage=None):
        self.train_set = Dataset(
            mode="train",
            size=(self.cfg["crop_height"], self.cfg["crop_width"]),
            img_dir=self.cfg["train_root"],
            mask_dir=self.cfg["train_mask_root"],
            full_mask_dir=self.cfg.get("train_full_mask_root"),
        )

        self.val_set = Dataset(
            mode="val",
            img_dir=self.cfg["val_root"],
            mask_dir=self.cfg["val_mask_root"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg["batch_size"],
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        val_len = len(self.val_set)
        gpus = self.cfg.get("gpus", 1)
        pad = (gpus - val_len % gpus) % gpus

        if pad > 0:
            dummy_set = DummyValDataset(pad)
            dataset = ConcatDataset([self.val_set, dummy_set])
        else:
            dataset = self.val_set

        return DataLoader(
            dataset,
            batch_size=1,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
        )


class TreeEnergyLoss(torch.nn.Module):
    def __init__(self, weight=1.0, sigma=0.02):
        super(TreeEnergyLoss, self).__init__()
        self.weight = weight
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=sigma)

    def forward(self, preds, low_feats, high_feats=None, unlabeled_ROIs=None):
        """
        Args:
            preds: (B, num_classes, H, W) model predictions
            low_feats: (B, C, H_low, W_low) low-level features (e.g., from early layers)
            high_feats: (B, C, H_high, W_high) high-level features (optional, from later layers)
            unlabeled_ROIs: (B, H, W) mask indicating unlabeled regions (255 for unlabeled, 0 for labeled)
        """
        batch, _, h, w = preds.size()
        target = (h, w)

        with torch.no_grad():
            low_feats = F.interpolate(low_feats, size=target, mode="bilinear", align_corners=False)
            if high_feats is not None:
                high_feats = F.interpolate(high_feats, size=target, mode="bilinear", align_corners=False)
            if unlabeled_ROIs is not None:
                roi = (
                    unlabeled_ROIs.unsqueeze(1).float()
                    if unlabeled_ROIs.dim() == 3
                    else unlabeled_ROIs.float()
                )
                unlabeled_ROIs = F.interpolate(roi, size=target, mode="nearest")
            else:
                unlabeled_ROIs = torch.ones(batch, 1, *target, device=preds.device)

        N = unlabeled_ROIs.sum()
        if N == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        prob = torch.softmax(preds, dim=1)  # (B, C, H', W')

        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)

        # high-level MST (optional)
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N
        return self.weight * tree_loss


def compute_boundary_loss(logits, labels, valid_mask=None):
    """Boundary loss using dense full masks (L_Boundary)."""
    device = logits.device
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)

    losses = []
    for b in range(logits.shape[0]):
        logits_b = logits[b : b + 1]
        labels_b = labels[b : b + 1]
        valid_mask_b = valid_mask[b : b + 1] if valid_mask is not None else None
        if len(torch.unique(labels_b).cpu().tolist()) <= 1:
            losses.append(torch.tensor(0.0, device=device, requires_grad=True))
            continue
        losses.append(_compute_boundary_loss_single(logits_b, labels_b, valid_mask_b, device))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def _compute_boundary_loss_single(logits, labels, valid_mask, device):
    _, _, h, w = logits.shape
    assert logits.shape[0] == 1

    loss_map = F.cross_entropy(logits, labels.squeeze(1), reduction="none", ignore_index=255).detach()
    edge_weight = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], device=device).view(1, 1, 3, 3).float()

    label_float = labels.float()
    ignore_edge = F.max_pool2d(label_float, kernel_size=5, stride=1, padding=2)
    edge = F.conv2d(label_float, edge_weight, padding=1)

    edge_threshold = 1e-2
    mask_tmp = (torch.abs(edge) > edge_threshold).float() * (ignore_edge < 200).float()
    if valid_mask is not None:
        mask_tmp = mask_tmp * (valid_mask != 255).float()

    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device).view(1, 1, 3, 3).float()
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device).view(1, 1, 3, 3).float()
    smooth_kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0

    x_edge = F.conv2d(F.conv2d(mask_tmp, gx, padding=1), smooth_kernel, padding=1)
    y_edge = F.conv2d(F.conv2d(mask_tmp, gy, padding=1), smooth_kernel, padding=1)
    norms = torch.sqrt(x_edge * x_edge + y_edge * y_edge) + 1e-1
    x_edge = -x_edge / norms
    y_edge = -y_edge / norms

    mask_fea = mask_tmp > edge_threshold
    if mask_fea.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    x_edge_flat = torch.masked_select(x_edge, mask_fea)
    y_edge_flat = torch.masked_select(y_edge, mask_fea)
    y_coords = torch.arange(h, device=device).view(1, 1, h, 1).expand(1, 1, h, w)
    x_coords = torch.arange(w, device=device).view(1, 1, 1, w).expand(1, 1, h, w)
    y_e = torch.masked_select(y_coords, mask_fea)
    x_e = torch.masked_select(x_coords, mask_fea)

    near_offset, far_offset = 2, 8
    x_near = (x_e + x_edge_flat * near_offset).long().clamp(0, w - 1)
    y_near = (y_e + y_edge_flat * near_offset).long().clamp(0, h - 1)
    x_far = (x_e + x_edge_flat * far_offset).long().clamp(0, w - 1)
    y_far = (y_e + y_edge_flat * far_offset).long().clamp(0, h - 1)

    fea_near = logits[0, :, y_near, x_near].transpose(0, 1)
    fea_far = logits[0, :, y_far, x_far].transpose(0, 1).detach()
    weight = torch.exp(-loss_map[0, y_far, x_far])
    corr = weight * torch.sum(F.normalize(fea_near, 2, 1) * F.normalize(fea_far, 2, 1), dim=1)
    return torch.mean(1.0 + corr)


class SegModel(L.LightningModule):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.num_classes = num_classes
        self.val_iou_metric = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            average="none",
            ignore_index=255,
            dist_sync_on_step=False,
        )
        # Boundary IoU (§4.2: dilation_ratio = 2% of image diagonal)
        self.boundary_dilation_ratio = float(cfg.get("boundary_dilation_ratio", 0.02))
        self.boundary_skip_background = bool(cfg.get("boundary_skip_background", False))

        # L_sup
        self.sup_enable = (cfg.get("sup") or {}).get("enable", True)

        # TER: Tree Energy Regularization (legacy key: tel)
        # dual-tree (low+high) as in §3.2.2; σ_aff from color_sigma
        ter_cfg = cfg.get("ter") or cfg.get("tel") or {}
        self.ter_enable = ter_cfg.get("enable", True)
        self.lambda_ter = ter_cfg.get(
            "lambda_ter", ter_cfg.get("lambda_tel", ter_cfg.get("weight", 0.0))
        )
        sigma_ter = ter_cfg.get("sigma", ter_cfg.get("color_sigma", 0.1))
        self.ter_loss = TreeEnergyLoss(weight=1.0, sigma=sigma_ter)

        # UWSD: Unlabeled Weak-Strong Distribution alignment (legacy key: consistency)
        # progressive λ_c schedule is an implementation detail; weight is λ_c peak
        uwsd_cfg = cfg.get("uwsd") or cfg.get("consistency") or {}
        self.uwsd_enable = uwsd_cfg.get("enable", True)
        self.uwsd_max_weight = uwsd_cfg.get("weight", uwsd_cfg.get("max_weight", 5.0))
        self.uwsd_warmup_epochs = 30
        self.uwsd_initial_ratio = 0.1

        # ARCL modules (parallel): L_Region + L_Boundary
        region_cfg = cfg.get("region") or cfg.get("arcl") or cfg.get("reco") or {}
        self.region_enable = region_cfg.get("enable", False)
        boundary_cfg = cfg.get("boundary") or {}
        self.boundary_enable = boundary_cfg.get(
            "enable", region_cfg.get("use_boundary_loss", False)
        )
        self.boundary_weight = boundary_cfg.get(
            "weight", region_cfg.get("boundary_weight", 0.05)
        )

        model_configs = {
            "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
            "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
            "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        backbone_type = cfg["backbone"].split("_")[-1]
        self.model = DPT(**{**model_configs[backbone_type], "nclass": num_classes})

        state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth', map_location="cpu")
        self.model.backbone.load_state_dict(state_dict)

        if cfg.get("lock_backbone", True):
            self.model.lock_backbone()

        self.criterion_l = ProbOhemCrossEntropy2d(**cfg["criterion"]["kwargs"])

        # ARCL L_Region representation branch
        if self.region_enable:
            features_dim = model_configs[backbone_type]["features"]
            self.region_branch = ReCoBranchDPT(
                in_channels=features_dim,
                output_dim=256,
            )

            self.weak_threshold = region_cfg.get("weak_threshold", 0.7)
            self.strong_threshold = region_cfg.get("strong_threshold", 0.97)
            self.temp = region_cfg.get("temp", 0.5)
            self.num_queries = region_cfg.get("num_queries", 256)
            self.num_negatives = region_cfg.get("num_negatives", 512)
            self.region_weight = region_cfg.get("weight", 0.1)

            self.intermediate_feat = None

            def hook_fn(module, input, output):
                self.intermediate_feat = output

            self.model.head.scratch.output_conv[1].register_forward_hook(hook_fn)
        else:
            self.region_branch = None
            self.region_weight = 0.0
            self.intermediate_feat = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            img_w, img_s, ignore_mask, full_mask_raw = batch
        else:
            img_w, img_s, ignore_mask = batch
            full_mask_raw = None

        B, n_crops, C, H, W = img_w.shape
        img_w = img_w.view(B * n_crops, C, H, W).to(self.device)
        img_s = img_s.view(B * n_crops, C, H, W).to(self.device)
        ignore_mask = ignore_mask.view(B * n_crops, H, W).to(self.device)

        full_mask = None
        if full_mask_raw is not None:
            full_mask = full_mask_raw.view(B * n_crops, H, W).to(self.device)

        if self.region_enable:
            self.intermediate_feat = None

        pred_w = self.model(img_w)

        if self.sup_enable and (ignore_mask != 255).sum() > 0:
            try:
                loss_sup = self.criterion_l(pred_w, ignore_mask.long())
            except Exception:
                loss_sup = F.cross_entropy(pred_w, ignore_mask.long(), ignore_index=255)
        else:
            loss_sup = torch.tensor(0.0, device=self.device, requires_grad=True)

        # TER: reuse strong-view backbone feats to avoid a third forward (OOM)
        run_ter = self.ter_enable and self.lambda_ter > 0

        if run_ter:
            pred_s, ter_feats = self.model(img_s, return_backbone_feats=True)
            low_feats, high_feats = ter_feats
        else:
            pred_s = self.model(img_s)
            low_feats, high_feats = None, None

        # UWSD: unlabeled weak-strong cosine consistency
        mask_unlabeled = (ignore_mask == 255).float().unsqueeze(1)
        soft_s = F.softmax(pred_s, dim=1)
        soft_w = F.softmax(pred_w, dim=1).detach()
        loss_uwsd = (1 - F.cosine_similarity(soft_s, soft_w, dim=1)) * mask_unlabeled.squeeze(1)
        loss_uwsd = loss_uwsd.sum() / (mask_unlabeled.sum() + 1e-6)

        del img_w

        if self.uwsd_enable:
            lambda_uwsd = min(
                self.uwsd_max_weight,
                (
                    self.uwsd_initial_ratio
                    + self.current_epoch
                    / self.uwsd_warmup_epochs
                    * (1 - self.uwsd_initial_ratio)
                )
                * self.uwsd_max_weight,
            )
        else:
            lambda_uwsd = 0.0

        if run_ter:
            unlabeled_ROIs = (ignore_mask == 255).float()
            loss_ter = self.ter_loss(
                preds=pred_s,
                low_feats=low_feats,
                high_feats=high_feats,
                unlabeled_ROIs=unlabeled_ROIs,
            )
        else:
            loss_ter = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = loss_sup + lambda_uwsd * loss_uwsd + self.lambda_ter * loss_ter

        enhanced_labels = None
        valid_mask = None

        if self.region_enable and self.region_branch is not None and self.intermediate_feat is not None:
            feat_h, feat_w = self.intermediate_feat.shape[2:]
            pred_h_s, pred_w_s = pred_s.shape[2:]

            if feat_h != pred_h_s or feat_w != pred_w_s:
                feat_interpolated = F.interpolate(
                    self.intermediate_feat,
                    size=(pred_h_s, pred_w_s),
                    mode='bilinear',
                    align_corners=True
                )
            else:
                feat_interpolated = self.intermediate_feat

            rep = self.region_branch(feat_interpolated)

            with torch.no_grad():
                probs_w = F.softmax(pred_w, dim=1)
                conf_w, pseudo_labels = probs_w.max(dim=1)

                enhanced_labels = ignore_mask.clone()
                high_conf_mask = (conf_w >= self.weak_threshold) & (ignore_mask == 255)
                enhanced_labels[high_conf_mask] = pseudo_labels[high_conf_mask]

                valid_mask = ((ignore_mask != 255) | high_conf_mask).float()

                labels_for_onehot = enhanced_labels.clone()
                labels_for_onehot[enhanced_labels == 255] = 0
                label_onehot_tensor = label_onehot(labels_for_onehot, self.num_classes)

                ignore_mask_expanded = (enhanced_labels == 255).unsqueeze(1).expand_as(label_onehot_tensor)
                label_onehot_tensor[ignore_mask_expanded] = 0.0

                label_onehot_resized = F.interpolate(
                    label_onehot_tensor,
                    size=rep.shape[2:],
                    mode='nearest'
                )
                valid_mask_resized = F.interpolate(
                    valid_mask.unsqueeze(1),
                    size=rep.shape[2:],
                    mode='nearest'
                )

                probs_s = F.softmax(pred_s, dim=1)
                probs_resized = F.interpolate(
                    probs_s,
                    size=rep.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            loss_region = compute_reco_loss(
                rep=rep,
                label=label_onehot_resized,
                mask=valid_mask_resized,
                prob=probs_resized,
                strong_threshold=self.strong_threshold,
                temp=self.temp,
                num_queries=self.num_queries,
                num_negatives=self.num_negatives
            )

            if not isinstance(loss_region, torch.Tensor):
                loss_region = torch.tensor(0.0, device=self.device, requires_grad=True)
            elif not loss_region.requires_grad:
                if loss_region.item() == 0.0:
                    loss_region = rep.sum() * 0.0

            lambda_region = min(
                self.region_weight,
                (0.1 + self.current_epoch / 30 * 0.9) * self.region_weight,
            )
            loss = loss + lambda_region * loss_region
            self.log("train/loss_region", loss_region, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/lambda_region", lambda_region, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/loss_region_weighted", lambda_region * loss_region, on_step=True, on_epoch=True, sync_dist=True)

        if self.boundary_enable:
            if full_mask is not None:
                loss_boundary = compute_boundary_loss(
                    logits=pred_s,
                    labels=full_mask,
                    valid_mask=None
                )
            elif self.region_enable and enhanced_labels is not None and valid_mask is not None:
                valid_mask_boundary = valid_mask.clone()
                valid_mask_boundary[valid_mask_boundary < 0.5] = 255.0

                loss_boundary = compute_boundary_loss(
                    logits=pred_s,
                    labels=enhanced_labels,
                    valid_mask=valid_mask_boundary
                )
            else:
                valid_mask_boundary = torch.ones_like(ignore_mask, dtype=torch.float32)
                valid_mask_boundary[ignore_mask == 255] = 255.0

                loss_boundary = compute_boundary_loss(
                    logits=pred_s,
                    labels=ignore_mask,
                    valid_mask=valid_mask_boundary
                )

            if not isinstance(loss_boundary, torch.Tensor):
                loss_boundary = torch.tensor(0.0, device=self.device, requires_grad=True)
            elif not loss_boundary.requires_grad:
                if loss_boundary.item() == 0.0:
                    loss_boundary = pred_s.sum() * 0.0

            loss = loss + self.boundary_weight * loss_boundary
            self.log("train/loss_boundary", loss_boundary, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/loss_boundary_weighted", self.boundary_weight * loss_boundary, on_step=True, on_epoch=True, sync_dist=True)

        if torch.isnan(loss):
            self.log("train/loss", 0.0, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_sup", loss_sup, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_uwsd", loss_uwsd, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/lambda_uwsd", lambda_uwsd, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_uwsd_weighted", lambda_uwsd * loss_uwsd, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss_ter", loss_ter, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self, outputs=None):
        avg_loss = self.trainer.callback_metrics.get("train/loss_epoch")

        if avg_loss is not None and self.global_rank == 0:
            print("=" * 40)
            print(f"Epoch {self.current_epoch} Training Loss: {avg_loss:.4f}")
            print("=" * 40)

        self.log("train/loss_epoch", avg_loss, on_epoch=True, sync_dist=True)

    def _is_dummy_val_batch(self, meta):
        if isinstance(meta, (list, tuple)):
            meta = meta[0]
        return isinstance(meta, dict) and meta.get("dummy", False)

    def _reset_boundary_iou_stats(self):
        device = self.device
        self._biou_sum = torch.zeros(self.num_classes, dtype=torch.float64, device=device)
        self._biou_cnt = torch.zeros(self.num_classes, dtype=torch.int64, device=device)

    def _sync_boundary_iou_stats(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self._biou_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self._biou_cnt, op=torch.distributed.ReduceOp.SUM)

    def _compute_boundary_miou(self):
        classes = range(self.num_classes)
        if self.boundary_skip_background:
            classes = [c for c in classes if c != 0]

        per_class = {}
        for cls in classes:
            if self._biou_cnt[cls].item() > 0:
                per_class[cls] = 100.0 * self._biou_sum[cls].item() / self._biou_cnt[cls].item()
            else:
                per_class[cls] = float("nan")

        vals = [v for v in per_class.values() if not np.isnan(v)]
        mean_v = float(np.mean(vals)) if vals else float("nan")
        return mean_v, per_class

    def validation_step(self, batch, batch_idx):
        img, mask, meta = batch

        if img.shape[0] == 0 or self._is_dummy_val_batch(meta):
            return

        ori_h, ori_w = img.shape[2], img.shape[3]
        crop_size = (518, 518)
        stride = (448, 448)

        pred_logits_student, _ = slide_inference(self.model, img, crop_size=crop_size, stride=stride)
        if pred_logits_student.shape[0] == 0:
            return

        pred_logits_student = F.interpolate(pred_logits_student, (ori_h, ori_w), mode="bilinear", align_corners=True)
        preds_student = pred_logits_student.argmax(dim=1)

        if preds_student.shape != mask.shape:
            preds_student = (
                F.interpolate(preds_student.unsqueeze(1).float(), mask.shape[1:], mode="nearest")
                .squeeze(1)
                .long()
            )

        try:
            self.val_iou_metric.update(preds_student, mask)
        except Exception:
            pass

        mask_np = mask[0].cpu().numpy().astype(np.int64)
        pred_np = preds_student[0].cpu().numpy().astype(np.int64)
        biou_cls = per_class_boundary_iou(
            mask_np,
            pred_np,
            self.num_classes,
            dilation_ratio=self.boundary_dilation_ratio,
            skip_background=self.boundary_skip_background,
        )
        for cls, v in biou_cls.items():
            if not np.isnan(v):
                self._biou_sum[cls] += v
                self._biou_cnt[cls] += 1

    def on_validation_epoch_start(self):
        self.val_iou_metric.reset()
        self._reset_boundary_iou_stats()

    def on_validation_epoch_end(self):
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*compute.*update.*")
            iou_class = self.val_iou_metric.compute().cpu().numpy() * 100.0
            mIoU = float(np.mean(iou_class))
        self.val_iou_metric.reset()

        self._sync_boundary_iou_stats()
        boundary_mIoU, boundary_iou_class = self._compute_boundary_miou()

        self.log("val/mIoU", mIoU, on_epoch=True, sync_dist=True)
        if not np.isnan(boundary_mIoU):
            self.log("val/boundary_mIoU", boundary_mIoU, on_epoch=True, sync_dist=True)

        if self.global_rank == 0:
            row = {
                "epoch": self.current_epoch,
                "loss": float(self.trainer.callback_metrics.get("train/loss_epoch", 0)),
                "mIoU": float(mIoU),
                "boundary_mIoU": float(boundary_mIoU) if not np.isnan(boundary_mIoU) else None,
            }
            for i, iou in enumerate(iou_class):
                class_name = CLASSES[self.cfg["dataset"]][i]
                row[class_name] = float(iou)
            for cls, biou in boundary_iou_class.items():
                class_name = CLASSES[self.cfg["dataset"]][cls]
                row[f"{class_name}_boundary_IoU"] = float(biou) if not np.isnan(biou) else None

            print("\n" + "=" * 40)
            print(f"Epoch {self.current_epoch} Validation Results:")
            print(f"{'Class':<15} | {'IoU (%)':>7} | {'Bd IoU (%)':>10}")
            print("-" * 40)
            for i, iou in enumerate(iou_class):
                class_name = CLASSES[self.cfg["dataset"]][i]
                biou = boundary_iou_class.get(i, float("nan"))
                biou_str = f"{biou:>10.2f}" if not np.isnan(biou) else f"{'N/A':>10}"
                print(f"{class_name:<15} | {iou:>7.2f} | {biou_str}")
            print("-" * 40)
            print(f"{'mIoU':<15} | {mIoU:>7.2f}")
            if not np.isnan(boundary_mIoU):
                print(f"{'boundary mIoU':<15} | {boundary_mIoU:>7.2f}")
            print("=" * 40 + "\n")

            save_path = os.path.join(self.cfg["save_root"], "results.csv")
            if os.path.exists(save_path):
                df = pd.read_csv(save_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df = df.groupby("epoch", as_index=False).mean()
            else:
                df = pd.DataFrame([row])
            df.to_csv(save_path, index=False)
            print(f"[INFO] Results saved to {save_path}")

    def configure_optimizers(self):
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = [param for name, param in self.model.named_parameters() if "backbone" not in name]

        # ARCL branch params
        params_list = [
            {"params": backbone_params, "lr": self.cfg["lr"]},
            {"params": head_params, "lr": self.cfg["lr"] * self.cfg["lr_multi"]},
        ]

        if self.region_enable and self.region_branch is not None:
            params_list.append({
                "params": self.region_branch.parameters(),
                "lr": self.cfg["lr"] * self.cfg["lr_multi"]
            })

        optimizer = AdamW(
            params_list,
            lr=self.cfg["lr"],
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        total_steps = len(self.trainer.datamodule.train_dataloader()) * self.cfg["epochs"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (1 - step / total_steps) ** 0.9,
        )

        return [optimizer], [scheduler]


def main():
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = yaml.safe_load(open(args.config, "r"))
    cfg = resolve_tacos_modules(cfg)
    cfg = resolve_cfg_paths(cfg)

    logger = init_log("global", logging.INFO)
    logger.info(
        "TACoS losses — L_sup=%s UWSD=%s TER=%s ARCL.region=%s ARCL.boundary=%s",
        cfg["sup"].get("enable", True),
        cfg["uwsd"].get("enable", True),
        cfg["ter"].get("enable", True),
        cfg["region"].get("enable", False),
        cfg["boundary"].get("enable", False),
    )
    logger.info(pprint.pformat({**cfg, **vars(args)}))
    os.makedirs(cfg["save_root"], exist_ok=True)

    datamodule = DataModule(cfg=cfg)

    model = SegModel(
        cfg=cfg,
        num_classes=cfg["nclass"],
    )

    tb_logger = TensorBoardLogger(
        save_dir=cfg["save_root"],
        name="lightning_logs",
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=cfg["save_root"],
            filename="best-{epoch}-{val/mIoU:.2f}",
            monitor="val/mIoU",
            mode="max",
            save_top_k=1,
        ),
    ]

    trainer = L.Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp_find_unused_parameters_true",
        num_sanity_val_steps=0,
        default_root_dir=cfg["save_root"],
        log_every_n_steps=10,
        precision=16,  # AMP as in §4.2
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path", None) or args.ckpt_path)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

