from copy import deepcopy

import numpy as np
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from transform import RandomCrop, RandomResize, blur, normalize


class Dataset(Dataset):
    def __init__(
        self,
        mode,
        size=None,
        img_dir=None,
        mask_dir=None,
        n_crops=1,
        full_mask_dir=None,
    ):
        self.mode = mode
        self.size = size
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.n_crops = n_crops
        self.full_mask_dir = full_mask_dir

        if mode not in ("train", "val"):
            raise ValueError(f"Unsupported mode: {mode}. Only 'train' and 'val' are allowed.")
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith(".jpg")]
        if len(self.ids) == 0:
            raise ValueError(f"No image files found in {self.img_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample_id = self.ids[index]
        img_path = os.path.join(self.img_dir, f"{sample_id}.jpg")
        if self.mode == "val":
            mask_path = os.path.join(self.mask_dir, f"{sample_id}.png")
        else:
            mask_path = os.path.join(self.mask_dir, f"{sample_id}_scribble.png")
            if not os.path.isfile(mask_path):
                mask_path = os.path.join(self.mask_dir, f"{sample_id}.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.mode == "val":
            img, mask = normalize(img, mask)
            return img, mask, sample_id

        img_w_list = []
        img_s_list = []
        ignore_mask_list = []
        full_mask_list = []

        full_mask = None
        if self.full_mask_dir is not None:
            full_mask_path = os.path.join(self.full_mask_dir, f"{sample_id}.png")
            if os.path.exists(full_mask_path):
                full_mask = Image.open(full_mask_path)

        for _ in range(self.n_crops):
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            results = {
                "img": img_np,
                "gt_seg_map": np.array(mask),
                "seg_fields": ["gt_seg_map"],
            }

            random_resize = RandomResize(
                scale=((h, w), (h, w)),
                ratio_range=(0.8, 1.2),
                keep_ratio=True,
            )
            results = random_resize.transform(results)
            results = RandomCrop(self.size, cat_max_ratio=1.0, ignore_index=255).transform(results)

            img_crop = Image.fromarray(results["img"])
            mask_crop = Image.fromarray(results["gt_seg_map"])

            full_mask_crop = None
            if full_mask is not None:
                results_full = {
                    "img": img_np,
                    "gt_seg_map": np.array(full_mask),
                    "seg_fields": ["gt_seg_map"],
                }
                results_full = random_resize.transform(results_full)
                results_full = RandomCrop(self.size, cat_max_ratio=1.0, ignore_index=255).transform(results_full)
                full_mask_crop = Image.fromarray(results_full["gt_seg_map"])

            if random.random() < 0.5:
                img_crop = img_crop.transpose(Image.FLIP_LEFT_RIGHT)
                mask_crop = mask_crop.transpose(Image.FLIP_LEFT_RIGHT)
                if full_mask_crop is not None:
                    full_mask_crop = full_mask_crop.transpose(Image.FLIP_LEFT_RIGHT)

            img_w_img = deepcopy(img_crop)
            img_s_img = deepcopy(img_crop)
            if random.random() < 0.8:
                img_s_img = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)(img_s_img)
            img_s_img = transforms.RandomGrayscale(p=0.05)(img_s_img)
            img_s_img = blur(img_s_img, p=0.2)

            ignore_mask = torch.from_numpy(np.array(mask_crop)).long()
            img_s_t, ignore_mask_t = normalize(img_s_img, ignore_mask)
            img_w_t = normalize(img_w_img)
            if ignore_mask_t.ndim == 3:
                ignore_mask_t = ignore_mask_t.squeeze(0)

            img_w_list.append(img_w_t)
            img_s_list.append(img_s_t)
            ignore_mask_list.append(ignore_mask_t)
            if full_mask_crop is not None:
                full_mask_list.append(torch.from_numpy(np.array(full_mask_crop)).long())

        if full_mask_list:
            return (
                torch.stack(img_w_list),
                torch.stack(img_s_list),
                torch.stack(ignore_mask_list),
                torch.stack(full_mask_list),
            )
        return torch.stack(img_w_list), torch.stack(img_s_list), torch.stack(ignore_mask_list)
