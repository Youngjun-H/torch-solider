"""
ReID Evaluation Metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def euclidean_distance(qf: torch.Tensor, gf: torch.Tensor) -> np.ndarray:
    """
    Compute Euclidean distance between query and gallery features

    Args:
        qf: Query features [m, d]
        gf: Gallery features [n, d]

    Returns:
        Distance matrix [m, n]
    """
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_distance(qf: torch.Tensor, gf: torch.Tensor) -> np.ndarray:
    """
    Compute cosine distance between query and gallery features

    Args:
        qf: Query features [m, d]
        gf: Gallery features [n, d]

    Returns:
        Distance matrix [m, n]
    """
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    dist_mat = 1 - torch.mm(qf, gf.t())
    return dist_mat.cpu().numpy()


def eval_func(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50
) -> Tuple[np.ndarray, float]:
    """
    Evaluation with market1501 metric

    Key: for each query identity, its gallery images from
    the same camera view are discarded.

    Args:
        distmat: Distance matrix [num_query, num_gallery]
        q_pids: Query person IDs
        g_pids: Gallery person IDs
        q_camids: Query camera IDs
        g_camids: Gallery camera IDs
        max_rank: Maximum rank for CMC

    Returns:
        cmc: CMC curve
        mAP: mean Average Precision
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # Compute CMC for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        # Get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # Compute CMC
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # Query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # Compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    feat_norm: bool = True,
    max_rank: int = 50
) -> dict:
    """
    Full evaluation pipeline

    Args:
        query_features: Query features [num_query, feat_dim]
        gallery_features: Gallery features [num_gallery, feat_dim]
        query_pids: Query person IDs
        gallery_pids: Gallery person IDs
        query_camids: Query camera IDs
        gallery_camids: Gallery camera IDs
        feat_norm: Whether to normalize features
        max_rank: Maximum rank for CMC

    Returns:
        Dictionary with mAP, rank1, rank5, rank10
    """
    # Normalize features
    if feat_norm:
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)

    # Compute distance matrix
    distmat = cosine_distance(query_features, gallery_features)

    # Evaluate
    cmc, mAP = eval_func(
        distmat, query_pids, gallery_pids,
        query_camids, gallery_camids, max_rank
    )

    return {
        'mAP': mAP,
        'rank1': cmc[0],
        'rank5': cmc[4] if len(cmc) > 4 else cmc[-1],
        'rank10': cmc[9] if len(cmc) > 9 else cmc[-1],
        'cmc': cmc
    }
