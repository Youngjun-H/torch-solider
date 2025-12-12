"""Evaluation metrics for ReID."""
import numpy as np
import torch
import torch.nn.functional as F


def euclidean_distance(qf, gf):
    """Compute euclidean distance matrix."""
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric."""
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0
    
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0
        
        # Compute AP
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP


class R1_mAP_eval:
    """R1 and mAP evaluator."""
    
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()
    
    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
    
    def update(self, output):
        """Update with batch output."""
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
    
    def compute(self):
        """Compute metrics."""
        feats = torch.cat(self.feats, dim=0)
        
        if self.feat_norm == 'yes':
            feats = F.normalize(feats, p=2, dim=1)
        
        # Query and gallery
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        
        # Compute distance matrix
        distmat = euclidean_distance(qf, gf)
        
        # Evaluate
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, self.max_rank)
        
        return cmc, mAP



