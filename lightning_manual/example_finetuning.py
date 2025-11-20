import os

import lightning as L
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


# Define the Pytorch nn.Modules
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


# Define a LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


class CIFAR10Classifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = LitAutoEncoder.load_from_checkpoint(
            "lightning_logs/version_0/checkpoints/epoch=9-step=1000.ckpt"
        ).encoder
        self.feature_extractor.freeze()

        self.classifier = nn.Linear(3, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        test_loss = F.mse_loss(y_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


# Define the training dataset
train_dataset = MNIST(
    os.getcwd(), download=True, train=True, transform=transforms.ToTensor()
)

train_set_size = int(len(train_dataset) * 0.8)
valid_set_size = len(train_dataset) - train_set_size
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(
    train_dataset, [train_set_size, valid_set_size], generator=seed
)
test_set = MNIST(
    os.getcwd(), download=True, train=False, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_set, batch_size=32)
valid_loader = DataLoader(valid_set, batch_size=32)
test_loader = DataLoader(test_set)


model = CIFAR10Classifier()

trainer = L.Trainer(
    limit_train_batches=100, max_epochs=10, devices=4, accelerator="gpu"
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
trainer.test(model, dataloaders=test_loader)
