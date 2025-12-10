import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection / (x_norm * y_norm)
    dist = (1.0 - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True
    )

    # `dist_an` means distance(anchor, negative)
    # DDP 환경에서 배치가 작아지면 negative example이 없는 경우가 발생할 수 있음
    # negative distance 행렬 생성 (negative가 아닌 위치는 매우 큰 값으로 채움)
    neg_dist_mat = dist_mat.clone()
    neg_dist_mat[~is_neg] = 1e6  # negative가 아닌 위치는 매우 큰 값

    # 각 행에서 최소값 찾기 (negative가 없는 행은 1e6가 선택됨)
    dist_an, absolute_n_inds = torch.min(neg_dist_mat, 1, keepdim=True)

    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (
            labels.new()
            .resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze(0)
            .expand(N, N)
        )
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data
        )
        # absolute_n_inds는 이미 전체 행렬에서의 인덱스이므로 직접 사용
        n_inds = absolute_n_inds.squeeze(1)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        #  dist_ap *= (1.0 + self.hard_factor)
        #  dist_an *= (1.0 - self.hard_factor)

        # negative example이 없는 anchor는 매우 큰 dist_an 값을 가짐 (1e6)
        # DDP에서 모든 파라미터가 loss에 기여하도록 하기 위해,
        # 모든 anchor에 대해 loss를 계산하되, invalid한 경우는 매우 작은 loss로 처리
        # invalid한 anchor는 dist_an이 매우 크므로 loss에 거의 기여하지 않음
        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            # MarginRankingLoss: dist_an이 매우 크면 loss가 거의 0이 됨
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            # SoftMarginLoss: dist_an - dist_ap이 매우 크면 loss가 거의 0이 됨
            loss = self.ranking_loss(dist_an - dist_ap, y)

        # valid_mask가 있는 경우에만 실제로 의미있는 loss가 계산됨
        # invalid한 anchor는 이미 dist_an이 매우 크므로 loss에 거의 기여하지 않음

        return loss, dist_ap, dist_an
