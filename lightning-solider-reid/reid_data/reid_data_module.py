"""ReID DataModule for Lightning."""

import lightning as L
import torchvision.transforms as T
from datasets.bases import ImageDataset
from datasets.custom_dataset import CustomDataset

# Local imports
from datasets.make_dataloader import (
    RandomIdentitySampler,
    RandomIdentitySampler_IdUniform,
    train_collate_fn,
    val_collate_fn,
)
from datasets.market1501 import Market1501
from datasets.mm import MM
from datasets.msmt17 import MSMT17
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
        """Setup datasets."""
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
        dataset = _factory[self.args.dataset_name](root=self.args.root_dir)

        self.train_dataset = ImageDataset(dataset.train, train_transforms)
        self.val_dataset = ImageDataset(dataset.query + dataset.gallery, val_transforms)

        self.num_classes = dataset.num_train_pids
        self.num_query = len(dataset.query)
        self.camera_num = dataset.num_train_cams
        self.view_num = dataset.num_train_vids

    def train_dataloader(self):
        """Create training dataloader."""
        if self.args.sampler in ["softmax_triplet", "img_triplet"]:
            sampler = RandomIdentitySampler(
                self.train_dataset,
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
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.test_ims_per_batch,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=val_collate_fn,
        )
