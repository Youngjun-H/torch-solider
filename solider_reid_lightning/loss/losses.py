"""Loss functions for ReID."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalize to unit length."""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """Compute euclidean distance."""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def hard_example_mining(dist_mat, labels):
    """Hard example mining for triplet loss."""
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    # Find positive and negative pairs
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    
    # Hard positive: farthest positive
    dist_ap, _ = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    
    # Hard negative: closest negative
    dist_an, _ = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_an = dist_an.squeeze(1)
    
    return dist_ap, dist_an


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy."""
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class TripletLoss(nn.Module):
    """Triplet loss with hard example mining."""
    
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss


class ReIDLoss(nn.Module):
    """Combined loss for ReID: ID loss + Triplet loss."""
    
    def __init__(self, num_classes, id_loss_weight=1.0, triplet_loss_weight=1.0,
                 triplet_margin=0.3, label_smooth=True, smoothing=0.1):
        super().__init__()
        self.id_loss_weight = id_loss_weight
        self.triplet_loss_weight = triplet_loss_weight
        
        if label_smooth:
            self.id_loss = LabelSmoothingCrossEntropy(num_classes, smoothing)
        else:
            self.id_loss = nn.CrossEntropyLoss()
        
        self.triplet_loss = TripletLoss(triplet_margin)
    
    def forward(self, score, feat, target):
        """
        Args:
            score: Classification scores (B, num_classes)
            feat: Global features (B, feat_dim)
            target: Person IDs (B,)
        """
        # ID loss
        id_loss = self.id_loss(score, target)
        
        # Triplet loss
        triplet_loss = self.triplet_loss(feat, target)
        
        # Combined loss
        loss = self.id_loss_weight * id_loss + self.triplet_loss_weight * triplet_loss
        
        return loss, id_loss, triplet_loss



