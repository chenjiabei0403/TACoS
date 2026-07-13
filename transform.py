import random

import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from base import BaseTransform, Resize
from typing import Optional, Sequence, Tuple, Union


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


class RandomCrop(BaseTransform):
    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def crop_bbox(self, results: dict) -> tuple:
        img = results['img']

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            margin_h = max(img.shape[0] - self.crop_size[0], 0)
            margin_w = max(img.shape[1] - self.crop_size[1], 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

            return crop_y1, crop_y2, crop_x1, crop_x2

        crop_bbox = generate_crop_bbox(img)

        if self.cat_max_ratio < 1.:
            for _ in range(50):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                valid_count = np.sum(seg_temp != self.ignore_index)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if (valid_count > 100
                    and len(cnt) >= 1
                    and np.max(cnt) / (np.sum(cnt) + 1e-6) < self.cat_max_ratio):
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        img = results['img']
        crop_bbox = self.crop_bbox(results)
        img = self.crop(img, crop_bbox)
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


class RandomResize(BaseTransform):
    def __init__(
        self,
        scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
        ratio_range: Optional[Tuple[float, float]] = None,
        keep_ratio: bool = True,
        interpolation: str = 'bilinear',
    ) -> None:
        self.scale = scale
        self.ratio_range = ratio_range
        self.resize = Resize(scale=0, keep_ratio=keep_ratio, interpolation=interpolation)

    def _random_sample_ratio(self, scale: tuple, ratio_range: Tuple[float, float]) -> tuple:
        if isinstance(scale[0], (tuple, list)):
            scale0 = int(scale[0][0])
        else:
            scale0 = int(scale[0])

        if isinstance(scale[1], (tuple, list)):
            scale1 = int(scale[1][0])
        else:
            scale1 = int(scale[1])

        min_ratio, max_ratio = ratio_range
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        return int(scale0 * ratio), int(scale1 * ratio)

    def _random_scale(self, img_shape: tuple = None) -> tuple:
        if self.ratio_range is not None and len(self.ratio_range) == 2:
            if img_shape is not None:
                scale = self._random_sample_ratio(img_shape, self.ratio_range)
            else:
                scale = self._random_sample_ratio(self.scale, self.ratio_range)
        else:
            scale = img_shape if img_shape is not None else self.scale

        return scale

    def transform(self, results: dict) -> dict:
        results['scale'] = self._random_scale()
        self.resize.scale = results['scale']
        results = self.resize(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range})'
        return repr_str
