# PyTorch Lightning 전이 학습(Fine-tuning) 실무 매뉴얼

이 매뉴얼은 PyTorch Lightning을 사용하여 기존에 학습된 모델(Pretrained Model)을 새로운 데이터셋에 맞춰 재학습(Fine-tuning)할 때 참조하기 위한 가이드입니다.

## 목차

1. [기본 원칙](#1-기본-원칙-core-principles)
2. [구현 패턴](#2-구현-패턴-implementation-patterns)
3. [실행 및 활용](#3-실행-및-활용-execution)
4. [주의사항 및 체크포인트](#4-주의사항-및-체크포인트-checkpoints)

---

## 1. 기본 원칙 (Core Principles)

### 완벽한 호환성
- PyTorch의 모든 `nn.Module`은 Lightning에서 즉시 사용 가능합니다.
- 기존 PyTorch 코드를 Lightning으로 마이그레이션할 때 모델 구조는 그대로 유지할 수 있습니다.

### 표준 워크플로우
전이 학습의 일반적인 단계는 다음과 같습니다:

```
사전 학습 모델 로드 
  → 특징 추출기 고정(Freeze) 
  → 새 분류기 부착 
  → 학습(Fine-tuning)
```

---

## 2. 구현 패턴 (Implementation Patterns)

상황에 맞는 패턴을 선택하여 코드를 작성하세요.

### 🅰️ 패턴 A: 기존 Lightning 체크포인트 활용

**상황**: `.ckpt` 파일로 저장된 Lightning 모델의 일부를 가져와 사용할 때

```python
import torch.nn as nn
from pytorch_lightning import LightningModule

class MyTransferModel(LightningModule):
    def __init__(self, checkpoint_path, num_classes=10):
        super().__init__()
        
        # 1. 체크포인트에서 모델 로드
        # (주의: 전체 모델이 로드되므로 필요한 부분만 추출해서 할당해야 함)
        pretrained_model = AutoEncoder.load_from_checkpoint(checkpoint_path)
        self.feature_extractor = pretrained_model.encoder 
        
        # 2. 가중치 고정 (Freeze)
        # 기존 모델의 지식이 파괴되지 않도록 업데이트를 막습니다.
        self.feature_extractor.freeze()
        
        # 3. 새로운 작업에 맞는 분류기(Head) 부착
        # 예: Encoder 출력(100 dim) -> 내 데이터 클래스(10개)
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x):
        # 4. 특징 추출 (Gradient 계산 안 됨 - freeze 효과)
        representations = self.feature_extractor(x)
        
        # 5. 분류 (이 부분만 학습됨)
        x = self.classifier(representations)
        return x
```

**핵심 포인트**:
- `load_from_checkpoint()`로 전체 모델을 로드한 후 필요한 부분만 추출
- `freeze()`로 특징 추출기의 가중치 고정
- 새로운 분류기만 학습 가능하도록 설정

---

### 🅱️ 패턴 B: 외부 Vision 모델 활용 (Torchvision)

**상황**: ResNet, EfficientNet 등 ImageNet으로 학습된 모델을 Feature Extractor로 사용할 때

```python
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule

class VisionFinetuner(LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        # 1. Backbone 로드 (weights="DEFAULT"는 최신 가중치 사용)
        backbone = models.resnet50(weights="DEFAULT")
        
        # 2. 모델 수술 (마지막 FC 레이어 제거)
        # ResNet은 마지막이 'fc'이므로 이를 제외한 모든 레이어를 사용
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        
        # 3. 가중치 고정 및 평가 모드 설정 (중요!)
        # BatchNorm 등이 학습 모드로 동작하여 통계가 틀어지는 것을 방지하기 위해 eval() 필수
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # 4. 새 분류기 부착
        num_filters = backbone.fc.in_features  # ResNet50 = 2048
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # 5. 추론 모드로 특징 추출 (no_grad 필수)
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
            
        # 6. 분류
        x = self.classifier(representations)
        return x
```

**핵심 포인트**:
- `weights="DEFAULT"`로 사전 학습된 가중치 자동 다운로드
- 마지막 분류 레이어 제거 후 새로운 분류기 부착
- **반드시 `eval()` 모드 설정** - BatchNorm, Dropout 비활성화
- `torch.no_grad()`로 메모리 효율성 향상

---

### ©️ 패턴 C: 외부 NLP 모델 활용 (Hugging Face)

**상황**: BERT, GPT 등 트랜스포머 모델을 활용할 때

```python
import torch.nn as nn
from transformers import BertModel
from pytorch_lightning import LightningModule

class BertFinetuner(LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        # 1. Hugging Face 모델 로드
        self.bert = BertModel.from_pretrained("bert-base-cased")
        
        # 2. 모드 설정
        # 전체를 미세조정(Fine-tuning) 하려면 train() 모드 유지
        # Feature로만 쓴다면 eval() 및 파라미터 freeze() 필요
        self.bert.train() 
        
        # 3. 분류기 부착
        self.W = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 4. BERT 통과
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
        
        # 5. CLS 토큰(문장 전체 의미) 추출
        h_cls = outputs.last_hidden_state[:, 0]
        
        # 6. 분류
        logits = self.W(h_cls)
        return logits
```

**핵심 포인트**:
- `from_pretrained()`로 사전 학습된 모델 자동 다운로드
- Fine-tuning vs Feature Extraction에 따라 모드 선택
- CLS 토큰을 활용한 문장 분류

---

## 3. 실행 및 활용 (Execution)

### 학습 (Training)

```python
from pytorch_lightning import Trainer

# 모델 인스턴스 생성
model = VisionFinetuner(num_classes=10)

# Trainer 설정 및 학습
trainer = Trainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    precision=16,  # Mixed precision training (선택사항)
)

trainer.fit(
    model, 
    train_dataloaders=train_loader, 
    val_dataloaders=val_loader
)
```

### 추론 (Inference)

```python
# 저장된 모델 로드
model = VisionFinetuner.load_from_checkpoint("best_model.ckpt")
model.freeze()  # 추론 시에는 반드시 freeze (메모리 절약)
model.eval()

# 데이터 예측
with torch.no_grad():
    predictions = model(input_data)
```

---

## 4. 주의사항 및 체크포인트 (Checkpoints)

### 💡 Freeze vs Train 전략

#### 데이터가 적을 때 (Small Dataset)
- **전략**: Feature Extractor를 `freeze()` 하고 Classifier만 학습
- **이유**: Overfitting 방지
- **학습률**: Classifier에 대해 상대적으로 높은 학습률 사용 (예: 1e-3)

```python
# Feature extractor 고정
for param in self.feature_extractor.parameters():
    param.requires_grad = False

# Classifier만 학습
optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)
```

#### 데이터가 많을 때 (Large Dataset)
- **전략**: 전체 모델을 `unfreeze` 하되, Learning Rate를 매우 낮게 설정
- **이유**: 사전 학습된 가중치를 천천히 조정
- **학습률**: 
  - Feature Extractor: 매우 낮은 학습률 (1e-4 ~ 1e-5)
  - Classifier: 상대적으로 높은 학습률 (1e-3)

```python
# 차별화된 학습률 설정
optimizer = torch.optim.Adam([
    {'params': self.feature_extractor.parameters(), 'lr': 1e-5},
    {'params': self.classifier.parameters(), 'lr': 1e-3}
])
```

### ⚠️ Eval Mode의 중요성 (Vision)

**문제**: Feature Extractor를 고정해서 사용할 때 `eval()` 모드를 설정하지 않으면:
- BatchNorm이 학습 모드로 동작하여 통계가 계속 업데이트됨
- Dropout이 활성화되어 일관성 없는 특징 추출
- **결과**: 성능 저하 및 불안정한 학습

**해결책**:
```python
# 반드시 eval() 모드 설정
self.feature_extractor.eval()

# 또는 freeze()와 함께 사용
self.feature_extractor.freeze()  # Lightning의 freeze()는 자동으로 eval()도 설정
```

### 📐 입력 크기 (Input Size)

**중요**: 사전 학습 모델이 기대하는 입력 크기와 정규화 방식을 정확히 맞춰야 합니다.

#### Vision 모델 예시:
```python
from torchvision import transforms

# ImageNet 사전 학습 모델의 경우
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # ResNet, EfficientNet 등은 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 통계
        std=[0.229, 0.224, 0.225]
    )
])
```

#### NLP 모델 예시:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# 모델이 기대하는 max_length 확인 (일반적으로 512)
encoded = tokenizer(text, max_length=512, padding=True, truncation=True)
```

### 🔧 추가 최적화 팁

1. **Learning Rate Scheduler 사용**
   ```python
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
   ```

2. **Early Stopping**
   ```python
   from pytorch_lightning.callbacks import EarlyStopping
   
   early_stop = EarlyStopping(monitor="val_loss", patience=3)
   trainer = Trainer(callbacks=[early_stop])
   ```

3. **Gradient Clipping**
   ```python
   trainer = Trainer(gradient_clip_val=1.0)
   ```

4. **Mixed Precision Training**
   ```python
   trainer = Trainer(precision=16)  # FP16 사용
   ```

---

## 요약 (Summary)

| 항목 | 권장 사항 |
|------|----------|
| **소규모 데이터** | Feature Extractor 고정, Classifier만 학습 |
| **대규모 데이터** | 전체 모델 학습, 차별화된 학습률 적용 |
| **Vision 모델** | 반드시 `eval()` 모드 설정 |
| **입력 크기** | 사전 학습 모델의 요구사항 정확히 준수 |
| **정규화** | 사전 학습 시 사용된 통계값 사용 |

---

## 참고 자료

- [PyTorch Lightning 공식 문서](https://lightning.ai/docs/pytorch/stable/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

