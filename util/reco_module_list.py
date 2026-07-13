import numpy as np
import torch
import torch.nn.functional as F


def compute_reco_loss(
    rep,
    label,
    mask,
    prob,
    strong_threshold=1.0,
    temp=0.5,
    num_queries=256,
    num_negatives=256,
):
    batch_size, num_feat, im_w_, im_h = rep.shape
    num_segments = label.shape[1]
    device = rep.device

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.clamp(0, 1)

    valid_pixel = label * mask
    rep = rep.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]
        if valid_pixel_seg.sum().item() == 0:
            continue

        prob_seg = prob[:, i, :, :]
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()
        proto = torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True)
        if torch.isnan(proto).any() or torch.isinf(proto).any():
            continue

        seg_proto_list.append(proto)
        seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    if len(seg_num_list) <= 1 or len(seg_proto_list) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    seg_proto = torch.cat(seg_proto_list)
    if torch.isnan(seg_proto).any() or torch.isinf(seg_proto).any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    valid_seg = len(seg_num_list)
    seg_len = torch.arange(valid_seg, device=device)
    reco_loss = torch.tensor(0.0, device=device, requires_grad=True)
    processed_classes = 0

    for i in range(valid_seg):
        if len(seg_feat_hard_list[i]) == 0:
            continue

        seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
        anchor_feat = seg_feat_hard_list[i][seg_hard_idx]

        with torch.no_grad():
            seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))
            if len(seg_mask) <= 1:
                continue

            proto_sim = torch.cosine_similarity(
                seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1
            )
            if (
                proto_sim.numel() == 0
                or torch.isnan(proto_sim).any()
                or torch.isinf(proto_sim).any()
                or temp <= 0
            ):
                continue

            proto_prob = torch.softmax(proto_sim / temp, dim=0)
            if torch.isnan(proto_prob).any() or torch.isinf(proto_prob).any() or proto_prob.sum() <= 0:
                continue

            processed_classes += 1
            negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
            samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
            samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

            negative_num_list = seg_num_list[i + 1 :] + seg_num_list[:i]
            negative_index = negative_index_sampler(samp_num, negative_num_list)
            negative_feat_all = torch.cat(seg_feat_all_list[i + 1 :] + seg_feat_all_list[:i])
            negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)
            positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
            all_feat = torch.cat((positive_feat, negative_feat), dim=1)

        seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
        reco_loss = reco_loss + F.cross_entropy(
            seg_logits / temp, torch.zeros(num_queries, dtype=torch.long, device=device)
        )

    if processed_classes == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return reco_loss / processed_classes


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(
                low=sum(seg_num_list[:j]),
                high=sum(seg_num_list[: j + 1]),
                size=int(samp_num[i, j]),
            ).tolist()
    return negative_index


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w], device=inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)
