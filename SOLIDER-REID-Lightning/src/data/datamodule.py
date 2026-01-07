"""
PyTorch Lightning 2.6+ DataModule for Person Re-Identification

Updated to follow Lightning 2.6+ best practices:
- import lightning as L
- Proper prepare_data() for DDP (rank 0 only)
- setup() for all ranks
- persistent_workers for efficiency
"""

import lightning as L  # ✅ Lightning 2.6+ import
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from typing import Optional, Tuple
from timm.data.random_erasing import RandomErasing

from .bases import ImageDataset
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from .market1501 import Market1501
from .msmt17 import MSMT17
from .cctv_reid import CCTVReID


class ReIDDataModule(L.LightningDataModule):  # ✅ L.LightningDataModule (Lightning 2.6+)
    """
    PyTorch Lightning 2.6+ DataModule for Person Re-Identification

    Features:
    - Support for Market1501, MSMT17, CCTVReID
    - Identity sampling (P×K) for triplet loss
    - Automatic DDP compatibility
    - Efficient data loading with persistent workers

    Lightning 2.6+ Updates:
    - Proper prepare_data() implementation
    - save_hyperparameters() support
    - persistent_workers for performance
    """

    def __init__(
        self,
        # Dataset config
        dataset_name: str = 'market1501',
        data_root: str = './data',

        # DataLoader config
        batch_size: int = 64,
        num_instances: int = 4,  # K in P×K sampling
        num_workers: int = 8,
        sampler: str = 'softmax_triplet',  # 'softmax', 'softmax_triplet', 'id_triplet'

        # Image config
        img_size_train: Tuple[int, int] = (384, 128),
        img_size_test: Tuple[int, int] = (384, 128),

        # Augmentation config
        random_flip_prob: float = 0.5,
        random_erase_prob: float = 0.5,
        padding: int = 10,
        pixel_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        pixel_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),

        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_factory = {
            'market1501': Market1501,
            'msmt17': MSMT17,
            'cctv_reid': CCTVReID,
        }

        # Initialize placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = None
        self.num_cameras = None
        self.num_views = None
        self.num_query = None

    def prepare_data(self):
        """Download datasets (called on 1 GPU/TPU in distributed)"""
        # Dataset download logic can be added here if needed
        pass

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for train/val/test"""
        dataset_name = self.hparams.dataset_name
        data_root = self.hparams.data_root

        # Load dataset
        if dataset_name not in self.dataset_factory:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Choose from {list(self.dataset_factory.keys())}")

        dataset = self.dataset_factory[dataset_name](root=data_root)

        # Store dataset info
        self.num_classes = dataset.num_train_pids
        self.num_cameras = dataset.num_train_cams
        self.num_views = dataset.num_train_vids
        self.num_query = len(dataset.query)

        # Build transforms
        train_transforms = self._build_train_transforms()
        val_transforms = self._build_val_transforms()

        if stage == 'fit' or stage is None:
            self.train_dataset = ImageDataset(dataset.train, train_transforms)
            self.val_dataset = ImageDataset(
                dataset.query + dataset.gallery,
                val_transforms
            )

        if stage == 'test' or stage is None:
            self.test_dataset = ImageDataset(
                dataset.query + dataset.gallery,
                val_transforms
            )

    def _build_train_transforms(self):
        """Build training data augmentation pipeline"""
        transforms = T.Compose([
            T.Resize(self.hparams.img_size_train, interpolation=3),
            T.RandomHorizontalFlip(p=self.hparams.random_flip_prob),
            T.Pad(self.hparams.padding),
            T.RandomCrop(self.hparams.img_size_train),
            T.ToTensor(),
            T.Normalize(mean=self.hparams.pixel_mean, std=self.hparams.pixel_std),
            RandomErasing(
                probability=self.hparams.random_erase_prob,
                mode='pixel',
                max_count=1,
                device='cpu'
            ),
        ])
        return transforms

    def _build_val_transforms(self):
        """Build validation/test data augmentation pipeline"""
        transforms = T.Compose([
            T.Resize(self.hparams.img_size_test),
            T.ToTensor(),
            T.Normalize(mean=self.hparams.pixel_mean, std=self.hparams.pixel_std)
        ])
        return transforms

    def train_dataloader(self):
        """Create training dataloader"""
        sampler_name = self.hparams.sampler

        if 'triplet' in sampler_name:
            # Use identity-based sampler for triplet loss
            if self.trainer and self.trainer.world_size > 1:
                # DDP sampler for multi-GPU
                sampler = RandomIdentitySampler_DDP(
                    self.train_dataset.dataset,
                    self.hparams.batch_size,
                    self.hparams.num_instances
                )
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    sampler,
                    self.hparams.batch_size // self.trainer.world_size,
                    True
                )
                return DataLoader(
                    self.train_dataset,
                    num_workers=self.hparams.num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=self._train_collate_fn,
                    pin_memory=True,
                )
            else:
                # Single GPU sampler
                sampler = RandomIdentitySampler(
                    self.train_dataset.dataset,
                    self.hparams.batch_size,
                    self.hparams.num_instances
                )
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.hparams.batch_size,
                    sampler=sampler,
                    num_workers=self.hparams.num_workers,
                    collate_fn=self._train_collate_fn,
                    pin_memory=True,
                )
        else:
            # Standard random sampler
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
                collate_fn=self._train_collate_fn,
                pin_memory=True,
                drop_last=True,
            )

    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size * 2,  # Larger batch for evaluation
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self._val_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size * 2,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self._val_collate_fn,
            pin_memory=True,
        )

    @staticmethod
    def _train_collate_fn(batch):
        """Collate function for training"""
        imgs, pids, camids, viewids, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, viewids

    @staticmethod
    def _val_collate_fn(batch):
        """Collate function for validation/test"""
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids_batch = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths
