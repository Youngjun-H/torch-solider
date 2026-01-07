import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics import Metric
from typing import List, Tuple


def euclidean_distance(qf, gf):
    """Compute Euclidean distance between query and gallery features"""
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_distance(qf, gf):
    """Compute cosine distance between query and gallery features"""
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    dist_mat = 1 - torch.mm(qf, gf.t())
    return dist_mat.cpu().numpy()


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
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
    num_valid_q = 0.0  # number of valid query

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
            # This condition is true when query identity does not appear in gallery
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


class ReIDMetrics(Metric):
    """
    TorchMetrics-compatible metric for Person Re-Identification

    Computes:
    - mAP (mean Average Precision)
    - CMC curve (Cumulative Matching Characteristics)
    """

    def __init__(
        self,
        num_query: int,
        max_rank: int = 50,
        feat_norm: str = 'yes',
        dist_type: str = 'euclidean',
        compute_on_step: bool = False,
        **kwargs
    ):
        super().__init__(compute_on_step=compute_on_step, **kwargs)

        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.dist_type = dist_type

        # State variables for accumulating predictions
        self.add_state("feats", default=[], dist_reduce_fx="cat")
        self.add_state("pids", default=[], dist_reduce_fx="cat")
        self.add_state("camids", default=[], dist_reduce_fx="cat")

    def update(self, feat: torch.Tensor, pid: List[int], camid: List[int]):
        """Update state with batch of predictions"""
        self.feats.append(feat.cpu())
        self.pids.append(torch.tensor(pid))
        self.camids.append(torch.tensor(camid))

    def compute(self):
        """Compute mAP and CMC scores"""
        # Concatenate all features
        feats = torch.cat(self.feats, dim=0)
        pids = torch.cat(self.pids, dim=0).numpy()
        camids = torch.cat(self.camids, dim=0).numpy()

        # Split into query and gallery
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        q_pids = pids[:self.num_query]
        g_pids = pids[self.num_query:]
        q_camids = camids[:self.num_query]
        g_camids = camids[self.num_query:]

        # Normalize features if needed
        if self.feat_norm == 'yes':
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        # Compute distance matrix
        if self.dist_type == 'cosine' or self.feat_norm == 'yes':
            distmat = cosine_distance(qf, gf)
        else:
            distmat = euclidean_distance(qf, gf)

        # Evaluate
        cmc, mAP = eval_func(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            max_rank=self.max_rank
        )

        return {
            'mAP': torch.tensor(mAP),
            'rank1': torch.tensor(cmc[0]),
            'rank5': torch.tensor(cmc[4]),
            'rank10': torch.tensor(cmc[9]),
            'cmc': torch.tensor(cmc)
        }

    def reset(self):
        """Reset internal state"""
        self.feats = []
        self.pids = []
        self.camids = []
