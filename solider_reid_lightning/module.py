"""Lightning Module for SOLIDER-REID."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model.reid_model import ReIDModel
from loss.losses import ReIDLoss
from utils.metrics import R1_mAP_eval


class ReIDLightningModule(LightningModule):
    """Lightning Module for ReID training."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['config'])
        
        # Build model
        self.model = ReIDModel(config)
        
        # Build loss
        self.criterion = ReIDLoss(
            num_classes=config.num_classes,
            id_loss_weight=config.id_loss_weight,
            triplet_loss_weight=config.triplet_loss_weight,
            triplet_margin=config.triplet_margin,
            label_smooth=config.label_smooth,
        )
        
        # Metrics
        self.train_acc = []
        self.val_evaluator = None
        self.num_query = None  # Will be set in on_validation_start
        
    def forward(self, x, cam_label=None, view_label=None):
        """Forward pass for inference."""
        return self.model(x, cam_label=cam_label, view_label=view_label)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        imgs, pids, camids, viewids = batch
        
        # Forward
        score, feat, _ = self.model(imgs, label=pids, cam_label=camids, view_label=viewids)
        
        # Loss
        loss, id_loss, triplet_loss = self.criterion(score, feat, pids)
        
        # Accuracy
        acc = (score.max(1)[1] == pids).float().mean()
        
        # Logging
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/id_loss', id_loss, on_step=True, on_epoch=True)
        self.log('train/triplet_loss', triplet_loss, on_step=True, on_epoch=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], on_step=True)
        
        return loss
    
    def on_validation_start(self):
        """Initialize evaluator at validation start."""
        # Get num_query from datamodule
        if self.num_query is None:
            self.num_query = len(self.trainer.datamodule.dataset.query)
        self.val_evaluator = R1_mAP_eval(self.num_query, max_rank=50, feat_norm='yes')
        self.val_evaluator.reset()
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        imgs, pids, camids, viewids, img_paths = batch
        
        # Forward
        feat, _ = self.model(imgs, cam_label=camids, view_label=viewids)
        
        # Update evaluator
        if self.val_evaluator is None:
            # Fallback: initialize if not done in on_validation_start
            if self.num_query is None:
                self.num_query = len(self.trainer.datamodule.dataset.query)
            self.val_evaluator = R1_mAP_eval(self.num_query, max_rank=50, feat_norm='yes')
            self.val_evaluator.reset()
        
        self.val_evaluator.update((feat, pids, camids))
        
        return None
    
    def on_validation_epoch_end(self):
        """Validation epoch end."""
        if self.val_evaluator is not None:
            cmc, mAP = self.val_evaluator.compute()
            
            # Log metrics
            self.log('val/mAP', mAP, prog_bar=True, sync_dist=True)
            self.log('val/Rank1', cmc[0], prog_bar=True, sync_dist=True)
            self.log('val/Rank5', cmc[4], sync_dist=True)
            self.log('val/Rank10', cmc[9], sync_dist=True)
            
            # Reset evaluator
            self.val_evaluator.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Optimizer
        if self.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.base_lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config.base_lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.base_lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Scheduler
        if self.config.warmup_method == "cosine":
            # Cosine annealing with warmup
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=self.config.warmup_factor,
                end_factor=1.0,
                total_iters=self.config.warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.max_epochs - self.config.warmup_epochs,
                eta_min=self.config.base_lr * 0.002,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.config.warmup_epochs],
            )
        else:
            # Linear warmup only
            scheduler = LinearLR(
                optimizer,
                start_factor=self.config.warmup_factor,
                end_factor=1.0,
                total_iters=self.config.warmup_epochs,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

