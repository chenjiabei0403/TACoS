from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import mmcv


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self,
                 results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @abstractmethod
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """


class Resize:
    """
    A simple MMCV-based Resize wrapper compatible with MMCV 1.x and 2.x.

    Args:
        scale (int or tuple): When keep_ratio is False, passed to imresize as (w, h).
            When keep_ratio is True and scale is a 2-tuple (sh, sw), use a uniform
            factor sh / current_height (consistent with RandomResize H*r), instead of
            mmcv's default "fit inside bounding box" behavior for 2-tuples.
        keep_ratio (bool): Whether to keep aspect ratio. Defaults to False.
        interpolation (str): Interpolation method. Defaults to 'bilinear'.
        backend (str): 'cv2' or 'pillow'. Defaults to 'cv2'.
    """
    def __init__(self, scale=None, keep_ratio=False, interpolation='bilinear', backend='cv2'):
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, results):
        """
        Args:
            results (dict): {
                'img': ndarray [H, W, C],
                'gt_seg_map': ndarray [H, W] (optional),
                'gt_depth_map': ndarray [H, W] (optional),
            }
        Returns:
            dict: results with resized images/maps
        """
        # Resize image
        if self.keep_ratio:
            img = results['img']
            h, w = img.shape[:2]
            s = self.scale
            # Avoid mmcv's 2-tuple "fit inside box" resize: landscape images shrink
            # along the short edge and produce abnormally small training patches.
            # RandomResize passes (sh, sw) as (H*r, H*r); use uniform factor sh/h instead.
            if isinstance(s, (tuple, list)) and len(s) == 2:
                sh = float(s[0])
                factor = sh / float(h)
                results['img'] = mmcv.imrescale(
                    img,
                    factor,
                    interpolation=self.interpolation,
                    backend=self.backend
                )
            else:
                results['img'] = mmcv.imrescale(
                    img,
                    s,
                    interpolation=self.interpolation,
                    backend=self.backend
                )
        else:
            results['img'] = mmcv.imresize(
                results['img'],
                self.scale,
                interpolation=self.interpolation,
                backend=self.backend
            )

        # Resize segmentation map (nearest interpolation)
        for key in ['gt_seg_map', 'gt_depth_map']:
            if key in results and results[key] is not None:
                results[key] = mmcv.imresize(
                    results[key],
                    results['img'].shape[:2][::-1],  # width, height
                    interpolation='nearest',
                    backend=self.backend
                )
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self):
        return f"Resize(scale={self.scale}, keep_ratio={self.keep_ratio}, " \
               f"interpolation={self.interpolation}, backend={self.backend})"
