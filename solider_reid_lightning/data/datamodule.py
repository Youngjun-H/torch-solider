"""Lightning DataModule for ReID."""

import torch
from lightning import LightningDataModule
from timm.data.random_erasing import RandomErasing
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import MSMT17, Market1501, ReIDDataset
from .sampler import RandomIdentitySampler


def train_collate_fn(batch):
    """Collate function for training."""
    imgs, pids, camids, viewids, _ = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return imgs, pids, camids, viewids


def val_collate_fn(batch):
    """Collate function for validation."""
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    return imgs, pids, camids, viewids, img_paths


class ReIDDataModule(LightningDataModule):
    """Lightning DataModule for ReID datasets."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Load dataset."""
        # Load dataset
        if self.config.dataset_name == "market1501":
            self.dataset = Market1501(root=self.config.data_root, verbose=True)
        elif self.config.dataset_name == "msmt17":
            self.dataset = MSMT17(root=self.config.data_root, verbose=True)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset_name}")

        # Create transforms
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.config.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Pad(10),
                transforms.RandomCrop(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                RandomErasing(probability=0.5, mode="pixel", max_count=1, device="cpu"),
            ]
        )

        val_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.config.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create datasets
        self.train_dataset = ReIDDataset(self.dataset.train, transform=train_transforms)
        self.val_dataset = ReIDDataset(
            self.dataset.query + self.dataset.gallery, transform=val_transforms
        )

        # Update num_classes in config
        if self.config.num_classes is None:
            self.config.num_classes = self.dataset.num_train_pids

    def train_dataloader(self):
        """Create training dataloader."""
        sampler = RandomIdentitySampler(
            self.dataset.train,
            batch_size=self.config.batch_size,
            num_instances=self.config.num_instances,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=torch.utils.data.BatchSampler(
                sampler, self.config.batch_size, False
            ),
            num_workers=self.config.num_workers,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=val_collate_fn,
            pin_memory=True,
        )
