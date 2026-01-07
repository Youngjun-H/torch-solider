"""
PyTorch Lightning 2.6+ Callback for Person Re-Identification Evaluation

Updated to follow Lightning 2.6+ best practices:
- import lightning as L
- Use Lightning 2.6+ Callback API
"""

import lightning as L  # ✅ Lightning 2.6+ import
import torch
from typing import Any
import sys
from pathlib import Path

# Add metrics to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics.reid_metrics import ReIDMetrics


class ReIDEvaluationCallback(L.Callback):  # ✅ L.Callback (Lightning 2.6+)
    """
    Callback for evaluating Person Re-Identification performance

    Computes mAP and CMC metrics during validation
    """

    def __init__(
        self,
        num_query: int,
        eval_period: int = 10,
        max_rank: int = 50,
        feat_norm: str = 'yes',
        dist_type: str = 'euclidean',
    ):
        super().__init__()
        self.num_query = num_query
        self.eval_period = eval_period
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.dist_type = dist_type

        # Initialize metric
        self.metric = None

    def on_validation_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule
    ) -> None:
        """Initialize metrics at the start of validation"""
        # Only evaluate every N epochs
        if (trainer.current_epoch + 1) % self.eval_period != 0:
            return

        # Initialize metric
        self.metric = ReIDMetrics(
            num_query=self.num_query,
            max_rank=self.max_rank,
            feat_norm=self.feat_norm,
            dist_type=self.dist_type,
        ).to(pl_module.device)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate features from each validation batch"""
        # Only evaluate every N epochs
        if (trainer.current_epoch + 1) % self.eval_period != 0:
            return

        if self.metric is not None and outputs is not None:
            feat = outputs['feat']
            pids = outputs['pids']
            cam_ids = outputs['cam_ids']

            # Update metric
            self.metric.update(feat, pids, cam_ids)

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule
    ) -> None:
        """Compute and log metrics at the end of validation"""
        # Only evaluate every N epochs
        if (trainer.current_epoch + 1) % self.eval_period != 0:
            return

        if self.metric is not None:
            # Compute metrics
            results = self.metric.compute()

            # Log metrics
            pl_module.log('val/mAP', results['mAP'], prog_bar=True, sync_dist=True)
            pl_module.log('val/rank1', results['rank1'], prog_bar=True, sync_dist=True)
            pl_module.log('val/rank5', results['rank5'], sync_dist=True)
            pl_module.log('val/rank10', results['rank10'], sync_dist=True)

            # Print results
            if trainer.is_global_zero:
                print(f"\n{'='*80}")
                print(f"Validation Results - Epoch {trainer.current_epoch + 1}:")
                print(f"  mAP: {results['mAP']:.1%}")
                print(f"  Rank-1: {results['rank1']:.1%}")
                print(f"  Rank-5: {results['rank5']:.1%}")
                print(f"  Rank-10: {results['rank10']:.1%}")
                print(f"{'='*80}\n")

            # Reset metric for next epoch
            self.metric.reset()
