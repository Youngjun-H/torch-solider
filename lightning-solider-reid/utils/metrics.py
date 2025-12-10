import os

import numpy as np
import torch

from .reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    # Updated to new PyTorch API: addmm_(mat1, mat2, *, beta=1, alpha=1)
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    import logging

    logger = logging.getLogger("transreid.check")

    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # Query와 Gallery 간 PID 불일치 확인
    unique_q_pids = np.unique(q_pids)
    unique_g_pids = np.unique(g_pids)
    common_pids = np.intersect1d(unique_q_pids, unique_g_pids)
    query_only_pids = np.setdiff1d(unique_q_pids, unique_g_pids)

    # DDP 환경 확인: torch.distributed가 초기화되어 있는지 확인
    import torch.distributed as dist

    is_ddp = dist.is_initialized() if hasattr(dist, "is_initialized") else False
    is_rank0 = not is_ddp or (dist.get_rank() == 0) if is_ddp else True

    if is_rank0:
        logger.info("=" * 60)
        logger.info("Evaluation PID Overlap Check")
        logger.info("=" * 60)
        logger.info(f"Query unique PIDs: {len(unique_q_pids)}")
        logger.info(f"Gallery unique PIDs: {len(unique_g_pids)}")
        logger.info(f"Common PIDs: {len(common_pids)}")
        logger.info(f"Query-only PIDs (will be skipped): {len(query_only_pids)}")

    if len(common_pids) == 0:
        if is_rank0:
            logger.error(
                "❌ ERROR: No common PIDs between query and gallery! "
                "Evaluation will fail."
            )
    elif len(common_pids) < len(unique_q_pids):
        # DDP 환경에서는 각 프로세스가 데이터의 일부만 받기 때문에
        # PID overlap이 적을 수 있습니다. 이는 정상적인 현상입니다.
        # Rank 0에서만 경고를 출력합니다.
        if is_rank0:
            logger.warning(
                f"⚠️  WARNING: Only {len(common_pids)}/{len(unique_q_pids)} "
                f"query PIDs appear in gallery."
            )
            if is_ddp:
                logger.warning(
                    "Note: In DDP mode, this warning may appear because each process "
                    "only receives a subset of the data. This is expected behavior."
                )
            if len(query_only_pids) <= 20:
                logger.warning(f"Query-only PIDs: {query_only_pids.tolist()}")
            else:
                logger.warning(
                    f"Query-only PIDs (first 20): {query_only_pids[:20].tolist()} ..."
                )

    # 각 query PID가 gallery에 있는지 확인
    invalid_query_indices = []
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        if q_pid not in unique_g_pids:
            invalid_query_indices.append(q_idx)

    if invalid_query_indices and is_rank0:
        logger.warning(
            f"⚠️  {len(invalid_query_indices)}/{num_q} query samples "
            f"have PIDs that don't appear in gallery"
        )
        if is_ddp:
            logger.warning(
                "Note: In DDP mode, this is expected as each process only sees "
                "a subset of the full dataset."
            )

    logger.info("=" * 60)

    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            logger.debug(
                f"Query {q_idx}: PID {q_pid}, camid {q_camid} - "
                f"no matching identity in gallery (after removing same camid)"
            )
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        logger.error("=" * 60)
        logger.error("EVALUATION FAILED: No valid queries found!")
        logger.error(f"Total queries: {num_q}")
        logger.error(f"Valid queries: {num_valid_q}")
        logger.error(
            "Possible causes: "
            "1. Query and gallery have no common PIDs, "
            "2. All query PIDs have same camid as gallery (removed during evaluation)"
        )
        logger.error("=" * 60)
        raise AssertionError(
            "Error: all query identities do not appear in gallery. "
            f"Total queries: {num_q}, Valid queries: {num_valid_q}. "
            "Check logs above for details."
        )

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval:
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        # Initialize attributes
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        # 데이터가 없는 경우 (DDP 환경에서 Rank 0이 아닌 프로세스에서 호출된 경우)
        if len(self.feats) == 0:
            # 더미 값 반환 (실제로는 호출되지 않아야 함)
            # torch는 파일 상단에서 이미 import되어 있음
            dummy_cmc = np.zeros(50, dtype=np.float32)
            dummy_mAP = 0.0
            dummy_distmat = np.zeros((1, 1), dtype=np.float32)
            return (
                dummy_cmc,
                dummy_mAP,
                dummy_distmat,
                [],
                [],
                torch.zeros(1, 1),
                torch.zeros(1, 1),
            )

        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        # 데이터가 num_query보다 적은 경우 (DDP 환경에서 부분 데이터만 받은 경우)
        if len(feats) < self.num_query:
            import logging

            logger = logging.getLogger("transreid.check")
            logger.warning(
                f"⚠️  WARNING: Received only {len(feats)} samples, but num_query={self.num_query}. "
                "This should not happen in Rank 0. Skipping evaluation."
            )
            # 더미 값 반환
            dummy_cmc = np.zeros(50, dtype=np.float32)
            dummy_mAP = 0.0
            dummy_distmat = np.zeros((1, 1), dtype=np.float32)
            return (
                dummy_cmc,
                dummy_mAP,
                dummy_distmat,
                [],
                [],
                torch.zeros(1, 1),
                torch.zeros(1, 1),
            )

        # query
        qf = feats[: self.num_query]
        q_pids = np.asarray(self.pids[: self.num_query])
        q_camids = np.asarray(self.camids[: self.num_query])
        # gallery
        gf = feats[self.num_query :]
        g_pids = np.asarray(self.pids[self.num_query :])

        g_camids = np.asarray(self.camids[self.num_query :])
        if self.reranking:
            print("=> Enter reranking")
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            print("=> Computing DistMat with euclidean_distance")
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf
