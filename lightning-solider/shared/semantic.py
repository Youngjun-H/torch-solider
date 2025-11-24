"""Semantic clustering utilities for SOLIDER."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def build_onehot_semantic_weight(dim, setVal=-1, device=None):
    """
    Semantic weight를 생성합니다.

    Args:
        dim: 배치 크기
        setVal: -1 (랜덤), 1 (모두 1), 0 (모두 0)
        device: 텐서가 위치할 device (None이면 자동으로 cuda 사용)

    Returns:
        weight: [dim, 2] 형태의 one-hot 벡터
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if setVal == -1:
        w = torch.randint(2, (dim, 1), device=device).float()
    elif setVal == 1:
        w = torch.ones(dim, 1, device=device)
    elif setVal == 0:
        w = torch.zeros(dim, 1, device=device)
    weight = torch.cat([w, 1 - w], axis=-1)
    return weight


def get_mask(features, clsnum):
    """
    Teacher features로부터 semantic mask를 생성합니다.

    Args:
        features: [N, C, H, W] 형태의 feature map
        clsnum: 클러스터 개수 (partnum)

    Returns:
        masks: [N', H, W] 형태의 mask (N'는 foreground가 충분한 이미지 수)
        mask_idxs: mask가 생성된 이미지 인덱스
    """
    n, c, h, w = features.shape
    masks = []
    mask_idxs = []

    for i in range(n):
        x = features[i].detach().cpu().numpy()
        x = x.transpose(1, 2, 0)
        x = x.reshape(-1, c)

        # foreground/background cluster
        _x = np.linalg.norm(x, axis=1, keepdims=True)
        km = KMeans(n_clusters=2, random_state=0).fit(_x)
        bg_mask = km.labels_
        ctrs = km.cluster_centers_
        if ctrs[0][0] > ctrs[1][0]:
            bg_mask = 1 - bg_mask
        idx = np.where(bg_mask == 1)[0]
        if len(idx) <= 0.5 * w * h:
            continue
        mask_idxs.append(i)

        # pixel cluster
        _x = x[idx]
        cluster = KMeans(n_clusters=clsnum, random_state=0).fit(_x)
        _res = cluster.labels_
        res = np.zeros(h * w)
        res[idx] = _res + 1

        # align
        res = res.reshape(h, w)
        ys = []
        for k in range(1, clsnum + 1):
            y = np.where(res == k)[0].mean()
            ys.append(y)
        ys = np.hstack(ys)
        y_idxs = np.argsort(ys) + 1
        heatmap = np.zeros_like(res)
        for k in range(1, clsnum + 1):
            heatmap[res == y_idxs[k - 1]] = k
        masks.append(heatmap)

    masks = np.stack(masks) if len(mask_idxs) > 0 else np.zeros(0)
    mask_idxs = np.hstack(mask_idxs) if len(mask_idxs) > 0 else np.zeros(0)
    return masks, mask_idxs
