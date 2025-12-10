"""ReID DataModule for Lightning."""

import lightning as L
import torch.distributed as dist
import torchvision.transforms as T
from datasets.bases import ImageDataset
from datasets.custom_dataset import CustomDataset

# Local imports
from datasets.make_dataloader import (
    RandomIdentitySampler,
    RandomIdentitySampler_IdUniform,
    StratifiedIdentitySampler,
    train_collate_fn,
    val_collate_fn,
)
from datasets.market1501 import Market1501
from datasets.mm import MM
from datasets.msmt17 import MSMT17
from datasets.sampler_ddp import (
    RandomIdentitySampler_DDP,
    StratifiedIdentitySampler_DDP,
)
from torch.utils.data import DataLoader

_factory = {
    "market1501": Market1501,
    "msmt17": MSMT17,
    "mm": MM,
    "custom": CustomDataset,
}


class ReIDDataModule(L.LightningDataModule):
    """ReID DataModule for PyTorch Lightning."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset = None
        self.val_dataset = None
        self.num_classes = None
        self.num_query = None
        self.camera_num = None
        self.view_num = None

    def setup(self, stage=None):
        """Setup datasets.

        Lightning 표준: 이 메서드는 Lightning이 자동으로 호출합니다.
        DDP 환경에서는 각 프로세스에서 호출되며, Lightning이 자동으로 동기화를 처리합니다.
        """
        # Create transforms
        train_transforms = T.Compose(
            [
                T.Resize(
                    self.args.size_train, interpolation=T.InterpolationMode.BICUBIC
                ),
                T.RandomHorizontalFlip(p=self.args.prob),
                T.Pad(self.args.padding),
                T.RandomCrop(self.args.size_train),
                T.ToTensor(),
                T.Normalize(mean=self.args.pixel_mean, std=self.args.pixel_std),
            ]
        )

        # Add RandomErasing
        from datasets.transforms import RandomErasing

        train_transforms.transforms.append(
            RandomErasing(
                probability=self.args.re_prob,
                mode="pixel",
                max_count=1,
                device="cpu",
            )
        )

        val_transforms = T.Compose(
            [
                T.Resize(self.args.size_test),
                T.ToTensor(),
                T.Normalize(mean=self.args.pixel_mean, std=self.args.pixel_std),
            ]
        )

        # Load dataset
        # Lightning 표준: 각 프로세스가 독립적으로 데이터셋을 로드합니다.
        # Lightning이 자동으로 동기화를 처리하므로 수동 barrier 불필요
        if self.args.dataset_name == "custom":
            # CustomDataset: train_dir, query_dir, gallery_dir이 모두 제공되면 별도 디렉토리 방식 사용
            if self.args.train_dir and self.args.query_dir and self.args.gallery_dir:
                dataset = _factory[self.args.dataset_name](
                    train_dir=self.args.train_dir,
                    query_dir=self.args.query_dir,
                    gallery_dir=self.args.gallery_dir,
                )
            else:
                # 기존 방식: root에서 자동 분리
                dataset = _factory[self.args.dataset_name](root=self.args.root_dir)
        else:
            dataset = _factory[self.args.dataset_name](root=self.args.root_dir)

        self.train_dataset = ImageDataset(dataset.train, train_transforms)
        self.val_dataset = ImageDataset(dataset.query + dataset.gallery, val_transforms)

        self.num_classes = dataset.num_train_pids
        self.num_query = len(dataset.query)
        self.camera_num = dataset.num_train_cams
        self.view_num = dataset.num_train_vids

    def train_dataloader(self):
        """Create training dataloader.

        Lightning 표준: 이 메서드는 Lightning이 필요할 때 호출합니다.
        DDP 환경에서는 각 프로세스에서 호출되며, trainer가 이미 attach되어 있을 수 있습니다.
        """
        # 디버깅: train_dataloader 호출 확인
        rank = 0
        if hasattr(self, "trainer") and self.trainer is not None:
            rank = getattr(self.trainer, "global_rank", 0)
        elif dist.is_initialized():
            rank = dist.get_rank()

        if rank == 0:
            print(f"[Rank 0] train_dataloader() called")
            print(f"  - sampler: {self.args.sampler}")
            print(f"  - dataset_name: {self.args.dataset_name}")

        if self.args.sampler in ["softmax_triplet", "img_triplet"]:
            # Lightning 표준: trainer를 통해 DDP 환경 확인
            # train_dataloader()가 호출될 때는 이미 trainer가 attach되어 있을 수 있음
            is_ddp = False
            world_size = 1
            rank = 0

            # Method 1: trainer를 통해 확인 (Lightning 표준)
            if hasattr(self, "trainer") and self.trainer is not None:
                is_ddp = (
                    self.trainer.world_size > 1
                    if hasattr(self.trainer, "world_size")
                    else False
                )
                world_size = (
                    self.trainer.world_size
                    if hasattr(self.trainer, "world_size")
                    else 1
                )
                rank = (
                    self.trainer.global_rank
                    if hasattr(self.trainer, "global_rank")
                    else 0
                )
            # Method 2: dist를 통해 확인 (fallback)
            elif dist.is_initialized():
                is_ddp = True
                world_size = dist.get_world_size()
                rank = dist.get_rank()

            if is_ddp:
                # DDP 환경에서는 DDP용 sampler 사용
                if rank == 0:
                    print(
                        f"[Rank 0] Creating DDP sampler (type: {self.args.sampler})..."
                    )
                    print(f"  - world_size: {world_size}")
                    print(f"  - rank: {rank}")

                if self.args.dataset_name == "custom":
                    data_sampler = StratifiedIdentitySampler_DDP(
                        self.train_dataset.dataset,  # ImageDataset의 dataset 속성 접근
                        self.args.ims_per_batch,
                        self.args.num_instance,
                    )
                else:
                    data_sampler = RandomIdentitySampler_DDP(
                        self.train_dataset.dataset,
                        self.args.ims_per_batch,
                        self.args.num_instance,
                    )

                # DDP에서는 mini_batch_size 계산
                mini_batch_size = self.args.ims_per_batch // world_size

                # sampler_ddp에서 이미 mini_batch_size를 조정했을 수 있으므로
                # sampler의 실제 mini_batch_size를 사용
                actual_mini_batch_size = (
                    data_sampler.mini_batch_size
                    if hasattr(data_sampler, "mini_batch_size")
                    else mini_batch_size
                )

                # 디버깅: 실제 batch size 확인 (Rank 0만)
                if rank == 0:
                    print(f"Batch size configuration:")
                    print(f"  - ims_per_batch: {self.args.ims_per_batch}")
                    print(f"  - world_size: {world_size}")
                    print(f"  - calculated mini_batch_size: {mini_batch_size}")
                    print(
                        f"  - actual mini_batch_size (from sampler): {actual_mini_batch_size}"
                    )
                    print(f"  - num_instance: {self.args.num_instance}")
                    print(
                        f"  - Expected samples per GPU per batch: {actual_mini_batch_size}"
                    )

                # Lightning 표준: Sampler는 인덱스를 하나씩 반환하고,
                # BatchSampler가 mini_batch_size 개씩 묶어서 배치로 만듦
                from torch.utils.data.sampler import BatchSampler

                batch_sampler = BatchSampler(
                    data_sampler, actual_mini_batch_size, drop_last=False
                )

                # 디버깅: BatchSampler 생성 확인 (Rank 0만)
                # 주의: len() 호출은 sampler의 __len__ 메서드를 호출하므로
                # DDP 동기화가 필요할 수 있습니다. 실제 학습 시작 전에는
                # sampler를 초기화하지 않도록 주의합니다.
                if rank == 0:
                    print(f"DEBUG: BatchSampler created:")
                    print(f"  - batch_size: {actual_mini_batch_size}")
                    print(f"  - num_instance: {self.args.num_instance}")
                    # len() 호출은 실제 학습 시작 시 Lightning이 자동으로 수행하므로
                    # 여기서는 호출하지 않음 (DDP 동기화 문제 방지)

                # DDP 환경에서는 num_workers를 줄여서 메모리 및 프로세스 문제 방지
                # 32개 GPU * 8 workers = 256개 worker 프로세스는 너무 많음
                # DDP에서는 보통 0-2가 적절
                num_workers = min(self.args.num_workers, 2)

                return DataLoader(
                    self.train_dataset,
                    batch_sampler=batch_sampler,  # Lightning 표준: BatchSampler 사용
                    num_workers=num_workers,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,  # Worker 재사용으로 성능 향상
                )
            else:
                # 단일 GPU 환경에서는 일반 sampler 사용
                if self.args.dataset_name == "custom":
                    sampler = StratifiedIdentitySampler(
                        self.train_dataset.dataset,  # ImageDataset의 dataset 속성 접근
                        self.args.ims_per_batch,
                        self.args.num_instance,
                    )
                else:
                    sampler = RandomIdentitySampler(
                        self.train_dataset.dataset,
                        self.args.ims_per_batch,
                        self.args.num_instance,
                    )
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.args.ims_per_batch,
                    sampler=sampler,
                    num_workers=self.args.num_workers,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                    persistent_workers=self.args.num_workers > 0,
                )
        elif self.args.sampler == "softmax":
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.ims_per_batch,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=train_collate_fn,
            )
        elif self.args.sampler in ["id_triplet", "id"]:
            sampler = RandomIdentitySampler_IdUniform(
                self.train_dataset, self.args.num_instance
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.ims_per_batch,
                sampler=sampler,
                num_workers=self.args.num_workers,
                collate_fn=train_collate_fn,
                drop_last=True,
            )
        else:
            raise ValueError(f"Unsupported sampler: {self.args.sampler}")

    def val_dataloader(self):
        """Create validation dataloader.

        Lightning 표준: 이 메서드는 Lightning이 필요할 때 호출합니다.
        For ReID evaluation, we need all data on all ranks for synchronization,
        but only Rank 0 will use it for metrics computation.
        Use SequentialSampler to ensure all ranks get the same data.
        """
        from torch.utils.data import SequentialSampler

        # Lightning 표준: trainer를 통해 DDP 환경 확인
        is_ddp = False
        if hasattr(self, "trainer") and self.trainer is not None:
            is_ddp = (
                self.trainer.world_size > 1
                if hasattr(self.trainer, "world_size")
                else False
            )
        elif dist.is_initialized():
            is_ddp = True

        # Use SequentialSampler to ensure all ranks get the same data
        # Lightning will not add DistributedSampler when a sampler is provided
        sampler = SequentialSampler(self.val_dataset)

        # DDP 환경에서는 num_workers를 줄여서 메모리 및 프로세스 문제 방지
        num_workers = min(self.args.num_workers, 2) if is_ddp else self.args.num_workers

        return DataLoader(
            self.val_dataset,
            batch_size=self.args.test_ims_per_batch,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
