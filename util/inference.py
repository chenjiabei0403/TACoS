import torch
import torch.nn.functional as F
from torch import Tensor


def slide_inference(
    model,
    inputs: Tensor,
    crop_size: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[Tensor, list]:
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = inputs.size()

    if batch_size == 0:
        return inputs.new_zeros((0, 2, h_img, w_img)), []

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    preds = None
    count_mat = None
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = inputs[:, :, y1:y2, x1:x2]
            if crop_img.shape[0] == 0:
                continue

            outs = model(crop_img)
            if isinstance(outs, Tensor):
                outs = [outs]
            crop_seg_logit = outs[0]
            if crop_seg_logit.shape[-2:] != crop_img.shape[-2:]:
                crop_seg_logit = F.interpolate(
                    crop_seg_logit, crop_img.shape[2:], align_corners=False, mode="bilinear"
                )

            if preds is None:
                actual_batch_size = batch_size if crop_seg_logit.shape[0] == 0 else max(crop_seg_logit.shape[0], batch_size)
                num_classes = 2 if crop_seg_logit.shape[0] == 0 else crop_seg_logit.shape[1]
                preds = inputs.new_zeros((actual_batch_size, num_classes, h_img, w_img))
                count_mat = inputs.new_zeros((actual_batch_size, 1, h_img, w_img))

            if crop_seg_logit.shape[0] == 0:
                crop_seg_logit = inputs.new_zeros(
                    (preds.shape[0], preds.shape[1], crop_img.shape[2], crop_img.shape[3])
                )
            elif crop_seg_logit.shape[0] != preds.shape[0]:
                crop_seg_logit = crop_seg_logit[: preds.shape[0]]

            preds += F.pad(
                crop_seg_logit,
                (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
            )
            count_mat[:, :, y1:y2, x1:x2] += 1

    if preds is None:
        raise ValueError(
            f"No predictions generated. batch_size={batch_size}, h_grids={h_grids}, w_grids={w_grids}."
        )
    assert (count_mat == 0).sum() == 0
    return preds / count_mat, []
