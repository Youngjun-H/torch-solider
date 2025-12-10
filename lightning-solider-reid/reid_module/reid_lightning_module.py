"""ReID LightningModule."""

import lightning as L
import torch

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
        self.camera_num = 0  # Will be set in setup
        self.view_num = 0  # Will be set in setup

    def setup(self, stage=None):
        """Setup model and loss.

        Lightning 표준: 이 메서드는 Lightning이 자동으로 호출합니다.
        stage는 "fit", "validate", "test", "predict" 중 하나입니다.
        DDP 환경에서는 각 프로세스에서 호출되며, Lightning이 자동으로 동기화를 처리합니다.
        """
        # Get dataset info from datamodule
        # Lightning 표준: setup()이 호출될 때는 datamodule이 이미 setup되어 있어야 함
        num_classes = 1000  # placeholder
        camera_num = 0
        view_num = 0

        # Method 1: From stored datamodule reference (가장 안정적)
        if self.datamodule is not None:
            num_classes = self.datamodule.num_classes
            camera_num = self.datamodule.camera_num
            view_num = self.datamodule.view_num
        # Method 2: From trainer (fallback)
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

        # Store camera_num, view_num, and num_classes for later use
        self.camera_num = camera_num
        self.view_num = view_num
        self.num_classes = num_classes

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
        """Configure optimizer and scheduler.

        Lightning 표준: 이 메서드는 Lightning이 자동으로 호출합니다.
        trainer가 이미 attach되어 있으므로 self.trainer를 통해 DDP 정보를 가져올 수 있습니다.
        """
        # Lightning 표준: setup()이 호출되었는지 확인
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. setup() must be called before configure_optimizers()."
            )

        # Lightning 표준: trainer를 통해 DDP 환경 확인
        # configure_optimizers()가 호출될 때는 trainer가 이미 attach되어 있음
        if (
            self.trainer is not None
            and hasattr(self.trainer, "world_size")
            and self.trainer.world_size > 1
        ):
            world_size = self.trainer.world_size
            # Gradient accumulation 횟수 가져오기
            accumulate_grad_batches = getattr(
                self.trainer, "accumulate_grad_batches", 1
            )
            if isinstance(accumulate_grad_batches, dict):
                # dict 형태인 경우 첫 번째 값 사용
                accumulate_grad_batches = (
                    list(accumulate_grad_batches.values())[0]
                    if accumulate_grad_batches
                    else 1
                )
            elif isinstance(accumulate_grad_batches, (list, tuple)):
                # list/tuple 형태인 경우 첫 번째 값 사용
                accumulate_grad_batches = (
                    accumulate_grad_batches[0] if accumulate_grad_batches else 1
                )

            # Effective batch size 계산
            # Linear scaling rule: lr = base_lr * (world_size * accumulate_grad_batches * mini_batch_size)
            # train.py와 동일한 계산식 사용: world_size * accumulate * mini_batch_size
            mini_batch_size = (
                self.args.ims_per_batch // world_size
                if world_size > 0
                else self.args.ims_per_batch
            )
            effective_batch_size = (
                world_size * accumulate_grad_batches * mini_batch_size
            )
            scaled_base_lr = self.args.base_lr * effective_batch_size
            if self.trainer.global_rank == 0:
                print(
                    f"DDP detected: scaling base_lr from {self.args.base_lr} to {scaled_base_lr} "
                    f"(world_size={world_size}, accumulate_grad_batches={accumulate_grad_batches}, "
                    f"mini_batch_size={mini_batch_size}, effective_batch_size={effective_batch_size})"
                )
        else:
            # 단일 GPU 환경
            scaled_base_lr = self.args.base_lr

        # Create optimizer config
        class OptimizerConfig:
            SOLVER = type("obj", (object,), {})()
            SOLVER.OPTIMIZER_NAME = self.args.optimizer_name
            SOLVER.BASE_LR = scaled_base_lr  # 스케일링된 learning rate 사용
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
                SOLVER.BASE_LR = scaled_base_lr  # 스케일링된 learning rate 사용
                SOLVER.WARMUP_EPOCHS = self.args.warmup_epochs

            sched_cfg = SchedulerConfig()
            scheduler = create_scheduler(sched_cfg, optimizer)
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

        # Lightning 표준: center_criterion이 실제로 사용되는 경우에만 optimizer_center 반환
        # IF_WITH_CENTER가 "no"이면 center loss가 사용되지 않으므로 optimizer_center 불필요
        # 다중 optimizer를 사용하려면 automatic_optimization=False가 필요하므로,
        # center loss가 실제로 사용되지 않는 경우에는 optimizer_center를 반환하지 않음
        #
        # 참고: make_loss()에서 center_criterion은 항상 생성되지만,
        # IF_WITH_CENTER="no"일 때는 loss 계산에 사용되지 않음
        #
        # 현재 설정에서는 IF_WITH_CENTER="no"이므로 optimizer_center를 반환하지 않음
        # center loss를 사용하려면 IF_WITH_CENTER="yes"로 설정하고
        # automatic_optimization=False로 설정한 후 training_step에서 수동으로 optimizer 호출 필요
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_fit_start(self):
        """Initialize evaluator."""
        # 디버깅: on_fit_start 호출 확인
        if self.trainer is not None and self.trainer.global_rank == 0:
            print(f"[Rank 0] on_fit_start() called")

        num_query = 100  # placeholder
        feat_norm = self.args.feat_norm == "yes"

        # Try to get num_query from datamodule
        if self.datamodule is not None:
            num_query = self.datamodule.num_query
        elif (
            hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None
        ):
            num_query = self.trainer.datamodule.num_query

        if self.trainer is not None and self.trainer.global_rank == 0:
            print(f"[Rank 0] Initializing evaluator with num_query={num_query}")

        self.evaluator = R1_mAP_eval(
            num_query=num_query,
            max_rank=50,
            feat_norm=feat_norm,
            reranking=False,
        )

        if self.trainer is not None and self.trainer.global_rank == 0:
            print(f"[Rank 0] Evaluator initialized")

    def training_step(self, batch, batch_idx):
        """Training step."""
        # batch는 collate_fn을 거친 후 (img, vid, target_cam, target_view) 튜플
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            img, vid, target_cam, target_view = batch
        else:
            # 예상치 못한 형태인 경우
            if self.trainer.global_rank == 0:
                print(
                    f"WARNING: Unexpected batch type: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}"
                )
            # 기본 unpacking 시도
            img, vid, target_cam, target_view = batch

        # 디버깅: 실제 batch size 확인 (첫 번째 step에서만)
        if batch_idx == 0 and self.trainer.global_rank == 0:
            actual_batch_size = (
                img.shape[0]
                if hasattr(img, "shape")
                else len(img) if isinstance(img, (list, tuple)) else 1
            )
            expected_batch_size = (
                self.args.ims_per_batch // self.trainer.world_size
                if hasattr(self.trainer, "world_size")
                else self.args.ims_per_batch
            )
            print(f"\n{'='*60}")
            print(
                f"Batch Debug Info (Rank {self.trainer.global_rank}, Step {batch_idx})"
            )
            print(f"{'='*60}")
            print(f"Actual batch size: {actual_batch_size}")
            print(f"Expected batch size per GPU: {expected_batch_size}")
            print(f"Image shape: {img.shape if hasattr(img, 'shape') else type(img)}")
            print(f"Batch type: {type(batch)}")
            print(
                f"Batch structure: {[type(x) for x in batch] if isinstance(batch, (list, tuple)) else 'N/A'}"
            )
            if actual_batch_size != expected_batch_size:
                print(
                    f"⚠️  WARNING: Batch size mismatch! Expected {expected_batch_size}, got {actual_batch_size}"
                )
                print(
                    f"   This may be normal for the first batch if sampler initialization is delayed."
                )
                print(
                    f"   If this persists, check sampler and collate_fn configuration."
                )
            print(f"{'='*60}\n")

        # PID 범위 검증 (디버깅용)
        if batch_idx == 0 and self.trainer.global_rank == 0:
            max_vid = vid.max().item()
            min_vid = vid.min().item()
            num_classes = self.num_classes if hasattr(self, "num_classes") else None
            if num_classes is not None:
                if max_vid >= num_classes or min_vid < 0:
                    raise ValueError(
                        f"Invalid PID range in batch: min={min_vid}, max={max_vid}, "
                        f"but num_classes={num_classes}. PIDs must be in range [0, {num_classes-1}]"
                    )

        # Camera ID와 View를 사용하지 않는 경우 None 전달
        # (모델이 camera_num=0, view_num=0이면 자동으로 무시)
        cam_label = None if self.camera_num == 0 else target_cam
        view_label = None if self.view_num == 0 else target_view

        # Forward
        score, feat, featmaps = self.model(
            img, label=vid, cam_label=cam_label, view_label=view_label
        )

        # Loss
        loss = self.loss_func(score, feat, vid, target_cam)

        # Accuracy
        if isinstance(score, list):
            acc = (score[0].max(1)[1] == vid).float().mean()
        else:
            acc = (score.max(1)[1] == vid).float().mean()

        # Get actual batch size for logging
        actual_batch_size = img.shape[0]

        # Get gradient accumulation info
        accumulate_grad_batches = 1
        if self.trainer is not None:
            accumulate_grad_batches = getattr(
                self.trainer, "accumulate_grad_batches", 1
            )
            if isinstance(accumulate_grad_batches, dict):
                accumulate_grad_batches = (
                    list(accumulate_grad_batches.values())[0]
                    if accumulate_grad_batches
                    else 1
                )
            elif isinstance(accumulate_grad_batches, (list, tuple)):
                accumulate_grad_batches = (
                    accumulate_grad_batches[0] if accumulate_grad_batches else 1
                )

        # Logging
        # Step 기반 로깅 (라인그래프로 표시)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=actual_batch_size,  # 실제 batch size 로깅
        )
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=actual_batch_size,
        )

        # Log step/batch ratio and effective batch size
        if self.trainer is not None and hasattr(self.trainer, "world_size"):
            world_size = self.trainer.world_size
            effective_batch_size = (
                world_size * accumulate_grad_batches * actual_batch_size
            )
            self.log(
                "step_batch_ratio",
                1.0 / accumulate_grad_batches,  # step/batch ratio
                on_step=True,
                logger=True,
            )
            self.log(
                "effective_batch_size",
                effective_batch_size,
                on_step=True,
                logger=True,
            )

        # Log learning rate
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        """Log epoch-level metrics to wandb as line graphs."""
        # wandb에서 epoch 메트릭을 라인그래프로 표시하기 위해 step 기반으로 로깅
        if self.logger is not None and hasattr(self.logger, "experiment"):
            current_step = self.trainer.global_step

            # Lightning이 계산한 epoch 평균 메트릭을 wandb에 step 기반으로 로깅
            # train_loss_epoch와 train_acc_epoch는 Lightning이 자동으로 계산하지만
            # wandb에서 직접 접근하기 어려우므로, trainer의 logged_metrics에서 가져옴
            logged_metrics = (
                self.trainer.logged_metrics
                if hasattr(self.trainer, "logged_metrics")
                else {}
            )

            # epoch 메트릭을 step 기반으로 로깅 (라인그래프로 표시)
            epoch_metrics = {}
            if "train_loss_epoch" in logged_metrics:
                epoch_metrics["train_loss_epoch"] = (
                    logged_metrics["train_loss_epoch"].item()
                    if hasattr(logged_metrics["train_loss_epoch"], "item")
                    else logged_metrics["train_loss_epoch"]
                )
            if "train_acc_epoch" in logged_metrics:
                epoch_metrics["train_acc_epoch"] = (
                    logged_metrics["train_acc_epoch"].item()
                    if hasattr(logged_metrics["train_acc_epoch"], "item")
                    else logged_metrics["train_acc_epoch"]
                )

            if epoch_metrics:
                self.logger.experiment.log(epoch_metrics, step=current_step)

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Standard Lightning implementation:
        - All ranks process batches for DDP synchronization
        - Only Rank 0 updates evaluator (ReID eval needs full dataset)
        - Returns None (Lightning standard for validation without loss)
        """
        img, vid, camid, camids, target_view, _ = batch

        # Camera ID와 View를 사용하지 않는 경우 None 전달
        cam_label = None if self.camera_num == 0 else camids
        view_label = None if self.view_num == 0 else target_view

        # Feature extraction (all ranks do this for DDP synchronization)
        with torch.no_grad():
            feat, _ = self.model(img, cam_label=cam_label, view_label=view_label)

        # Only Rank 0 updates evaluator (ReID eval needs full dataset)
        if self.trainer is not None and self.trainer.global_rank == 0:
            if self.evaluator is not None:
                self.evaluator.update((feat, vid, camid))
            elif batch_idx == 0:
                # Only print warning once per epoch
                print("Warning: evaluator is None in validation_step")

        # Lightning standard: return None for validation without loss
        # Lightning will automatically handle DDP synchronization
        return None

    def on_validation_epoch_end(self):
        """Compute validation metrics.

        Standard Lightning implementation:
        - Only Rank 0 computes metrics (ReID eval needs full dataset)
        - Use sync_dist=True to broadcast metrics to all ranks
        - All ranks log the same metrics for consistency
        """
        # Check if evaluator is available
        if self.evaluator is None:
            if self.trainer is not None and self.trainer.global_rank == 0:
                print("Warning: evaluator is None in on_validation_epoch_end")
            # Log zero values for synchronization
            self.log("val_mAP", 0.0, sync_dist=True, on_epoch=True, logger=True)
            self.log("val_Rank1", 0.0, sync_dist=True, on_epoch=True, logger=True)
            self.log("val_Rank5", 0.0, sync_dist=True, on_epoch=True, logger=True)
            self.log("val_Rank10", 0.0, sync_dist=True, on_epoch=True, logger=True)
            return

        # Only Rank 0 computes metrics (ReID eval needs full dataset)
        if self.trainer is None or self.trainer.global_rank != 0:
            # Other ranks wait for Rank 0 to compute and broadcast
            # sync_dist=True will handle synchronization
            return

        # Check if evaluator has data
        if len(self.evaluator.feats) == 0:
            print(
                f"Warning: evaluator has no data (epoch {self.trainer.current_epoch})"
            )
            # Log zero values for synchronization
            self.log("val_mAP", 0.0, sync_dist=True, on_epoch=True, logger=True)
            self.log("val_Rank1", 0.0, sync_dist=True, on_epoch=True, logger=True)
            self.log("val_Rank5", 0.0, sync_dist=True, on_epoch=True, logger=True)
            self.log("val_Rank10", 0.0, sync_dist=True, on_epoch=True, logger=True)
            return

        # Compute metrics (only on Rank 0)
        cmc, mAP, _, _, _, _, _ = self.evaluator.compute()

        # Print validation results to console (only on Rank 0)
        print(f"\n{'='*60}")
        print(f"Validation Results - Epoch {self.trainer.current_epoch}")
        print(f"{'='*60}")
        print(f"mAP: {mAP:.4f} ({mAP*100:.2f}%)")
        print(f"Rank-1:  {cmc[0]:.4f} ({cmc[0]*100:.2f}%)")
        print(f"Rank-5:  {cmc[4]:.4f} ({cmc[4]*100:.2f}%)")
        print(f"Rank-10: {cmc[9]:.4f} ({cmc[9]*100:.2f}%)")
        print(f"{'='*60}\n")

        # Log metrics with sync_dist=True to broadcast to all ranks
        # This is the Lightning standard way to handle DDP metrics
        self.log(
            "val_mAP",
            mAP,
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_Rank1",
            cmc[0],
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_Rank5",
            cmc[4],
            sync_dist=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "val_Rank10",
            cmc[9],
            sync_dist=True,
            on_epoch=True,
            logger=True,
        )

        # Additional wandb logging for line graph visualization
        # Log with step-based indexing for line graph display
        if self.logger is not None and hasattr(self.logger, "experiment"):
            current_step = self.trainer.global_step
            self.logger.experiment.log(
                {
                    "val_mAP": mAP,
                    "val_Rank1": cmc[0],
                    "val_Rank5": cmc[4],
                    "val_Rank10": cmc[9],
                },
                step=current_step,
            )

        # Reset evaluator for next epoch
        self.evaluator.reset()
