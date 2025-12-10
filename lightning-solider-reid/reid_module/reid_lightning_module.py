"""ReID LightningModule."""

import lightning as L

# Local imports
from loss import make_loss
from model import make_model
from solver import WarmupMultiStepLR, make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.metrics import R1_mAP_eval


class ReIDLightningModule(L.LightningModule):
    """ReID Lightning Module."""

    def __init__(self, args, datamodule=None):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self.datamodule = datamodule  # Store datamodule reference

        # Will be initialized in setup
        self.model = None
        self.loss_func = None
        self.center_criterion = None
        self.evaluator = None

    def setup(self, stage=None):
        """Setup model and loss."""
        # Get dataset info from datamodule
        # Try multiple ways to get datamodule info
        num_classes = 1000  # placeholder
        camera_num = 0
        view_num = 0

        # Method 1: From stored datamodule reference
        if self.datamodule is not None:
            num_classes = self.datamodule.num_classes
            camera_num = self.datamodule.camera_num
            view_num = self.datamodule.view_num
        # Method 2: From trainer (when attached)
        elif hasattr(self, "trainer") and self.trainer is not None:
            try:
                if (
                    hasattr(self.trainer, "datamodule")
                    and self.trainer.datamodule is not None
                ):
                    num_classes = self.trainer.datamodule.num_classes
                    camera_num = self.trainer.datamodule.camera_num
                    view_num = self.trainer.datamodule.view_num
            except RuntimeError:
                # Trainer not attached yet, use placeholder
                pass

        # Create model config
        class Config:
            MODEL = type("obj", (object,), {})()
            MODEL.NAME = "transformer"
            MODEL.TRANSFORMER_TYPE = self.args.transformer_type
            MODEL.PRETRAIN_PATH = self.args.pretrain_path
            MODEL.PRETRAIN_CHOICE = self.args.pretrain_choice
            MODEL.SEMANTIC_WEIGHT = self.args.semantic_weight
            MODEL.ID_LOSS_TYPE = self.args.id_loss_type
            MODEL.NECK = "bnneck"
            MODEL.NECK_FEAT = self.args.neck_feat
            MODEL.REDUCE_FEAT_DIM = False
            MODEL.FEAT_DIM = 512
            MODEL.DROPOUT_RATE = 0.0
            MODEL.DROP_PATH = 0.1
            MODEL.DROP_OUT = 0.0
            MODEL.ATT_DROP_RATE = 0.0
            MODEL.SIE_CAMERA = False
            MODEL.SIE_VIEW = False
            MODEL.JPM = False
            MODEL.LAST_STRIDE = 1
            MODEL.COS_LAYER = False
            MODEL.STRIDE_SIZE = [16, 16]
            TEST = type("obj", (object,), {})()
            TEST.NECK_FEAT = self.args.neck_feat
            TEST.FEAT_NORM = self.args.feat_norm
            INPUT = type("obj", (object,), {})()
            INPUT.SIZE_TRAIN = self.args.size_train
            SOLVER = type("obj", (object,), {})()
            SOLVER.COSINE_SCALE = self.args.cosine_scale
            SOLVER.COSINE_MARGIN = self.args.cosine_margin

        cfg = Config()

        self.model = make_model(
            cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
            semantic_weight=self.args.semantic_weight,
        )

        # Create loss config
        class LossConfig:
            MODEL = type("obj", (object,), {})()
            MODEL.METRIC_LOSS_TYPE = self.args.metric_loss_type
            MODEL.NO_MARGIN = self.args.no_margin
            MODEL.IF_LABELSMOOTH = self.args.if_labelsmooth
            MODEL.IF_WITH_CENTER = "no"
            MODEL.ID_LOSS_WEIGHT = self.args.id_loss_weight
            MODEL.TRIPLET_LOSS_WEIGHT = self.args.triplet_loss_weight
            DATALOADER = type("obj", (object,), {})()
            DATALOADER.SAMPLER = self.args.sampler
            SOLVER = type("obj", (object,), {})()
            SOLVER.MARGIN = self.args.margin
            SOLVER.TRP_L2 = False

        loss_cfg = LossConfig()
        self.loss_func, self.center_criterion = make_loss(
            loss_cfg, num_classes=num_classes
        )

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""

        # Create optimizer config
        class OptimizerConfig:
            SOLVER = type("obj", (object,), {})()
            SOLVER.OPTIMIZER_NAME = self.args.optimizer_name
            SOLVER.BASE_LR = self.args.base_lr
            SOLVER.WEIGHT_DECAY = self.args.weight_decay
            SOLVER.WEIGHT_DECAY_BIAS = self.args.weight_decay_bias
            SOLVER.BIAS_LR_FACTOR = self.args.bias_lr_factor
            SOLVER.MOMENTUM = self.args.momentum
            SOLVER.LARGE_FC_LR = self.args.large_fc_lr
            SOLVER.CENTER_LR = 0.5

        opt_cfg = OptimizerConfig()
        optimizer, optimizer_center = make_optimizer(
            opt_cfg, self.model, self.center_criterion
        )

        # Configure scheduler
        if self.args.warmup_method == "cosine":

            class SchedulerConfig:
                SOLVER = type("obj", (object,), {})()
                SOLVER.MAX_EPOCHS = self.args.max_epochs
                SOLVER.BASE_LR = self.args.base_lr
                SOLVER.WARMUP_EPOCHS = self.args.warmup_epochs

            sched_cfg = SchedulerConfig()
            scheduler = create_scheduler(sched_cfg, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            # Step scheduler
            scheduler = WarmupMultiStepLR(
                optimizer,
                milestones=self.args.steps,
                gamma=self.args.gamma,
                warmup_factor=self.args.warmup_factor,
                warmup_iters=self.args.warmup_epochs,
                warmup_method=self.args.warmup_method,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

    def on_fit_start(self):
        """Initialize evaluator."""
        num_query = 100  # placeholder
        feat_norm = self.args.feat_norm == "yes"

        # Try to get num_query from datamodule
        if self.datamodule is not None:
            num_query = self.datamodule.num_query
        elif (
            hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None
        ):
            num_query = self.trainer.datamodule.num_query

        self.evaluator = R1_mAP_eval(
            num_query=num_query,
            max_rank=50,
            feat_norm=feat_norm,
            reranking=False,
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        img, vid, target_cam, target_view = batch

        # Forward
        score, feat, featmaps = self.model(
            img, label=vid, cam_label=target_cam, view_label=target_view
        )

        # Loss
        loss = self.loss_func(score, feat, vid, target_cam)

        # Accuracy
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == vid).float().mean()
        else:
            acc = (score.max(1)[1] == vid).float().mean()

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)

        # Log learning rate
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        img, vid, camid, camids, target_view, _ = batch

        # Feature extraction
        feat, _ = self.model(img, cam_label=camids, view_label=target_view)

        # Update evaluator
        self.evaluator.update((feat, vid, camid))

    def on_validation_epoch_end(self):
        """Compute validation metrics."""
        if self.evaluator is None:
            return

        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()

        self.log("val_mAP", mAP, prog_bar=True)
        self.log("val_Rank1", cmc[0], prog_bar=True)
        self.log("val_Rank5", cmc[4])
        self.log("val_Rank10", cmc[9])

        self.evaluator.reset()
