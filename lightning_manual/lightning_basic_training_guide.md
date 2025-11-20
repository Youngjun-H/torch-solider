# 모델 훈련하기 (기본)

**대상**: 자체 훈련 루프를 코딩하지 않고 모델을 훈련해야 하는 사용자

---

## Import 추가하기

파일 상단에 관련 import를 추가합니다.

```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
```

---

## PyTorch nn.Modules 정의하기

```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
```

---

## LightningModule 정의하기

LightningModule은 **nn.Modules**가 어떻게 상호작용하는지 정의하는 완전한 **레시피**입니다.

- **training_step**은 *nn.Modules*가 어떻게 함께 상호작용하는지 정의합니다.
- **configure_optimizers**에서 모델의 옵티마이저를 정의합니다.

```python
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step은 훈련 루프를 정의합니다.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

---

## 훈련 데이터셋 정의하기

훈련 데이터셋을 포함하는 PyTorch `DataLoader`를 정의합니다.

```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)
```

---

## 모델 훈련하기

모델을 훈련하려면 Lightning `Trainer`를 사용합니다. 이는 모든 엔지니어링을 처리하고 스케일링에 필요한 모든 복잡성을 추상화합니다.

```python
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = L.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```

---

## 훈련 루프 제거하기

내부적으로 Lightning Trainer는 다음과 같은 훈련 루프를 대신 실행합니다:

```python
autoencoder = LitAutoEncoder(Encoder(), Decoder())
optimizer = autoencoder.configure_optimizers()

for batch_idx, batch in enumerate(train_loader):
    loss = autoencoder.training_step(batch, batch_idx)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Lightning의 강력함은 검증/테스트 분할, 스케줄러, 분산 훈련 및 모든 최신 SOTA 기법을 추가할 때 훈련 루프가 복잡해질 때 나타납니다.

Lightning을 사용하면 매번 새로운 루프를 다시 작성할 필요 없이 이러한 모든 기법을 함께 혼합할 수 있습니다.

