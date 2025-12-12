# SOLIDER-REID Lightning Implementation

Lightning 기반의 SOLIDER-REID 구현입니다. PyTorch Lightning 표준 코드로 깔끔하게 재구현했습니다.

## 특징

- ✅ **Lightning 표준 코드**: PyTorch Lightning의 표준 패턴을 따릅니다
- ✅ **심플한 구조**: 최소한의 코드로 핵심 기능 구현
- ✅ **SOLIDER Semantic Weight**: SOLIDER의 핵심 기능인 semantic weight 지원
- ✅ **DDP 지원**: 자동 분산 학습 지원
- ✅ **WandB 통합**: 학습 로깅 및 모니터링

## 프로젝트 구조

```
solider_reid_lightning/
├── config.py              # 설정 관리
├── train.py               # 메인 학습 스크립트
├── module.py              # LightningModule
├── data/
│   ├── dataset.py         # 데이터셋 클래스
│   ├── sampler.py         # 샘플러
│   └── datamodule.py      # LightningDataModule
├── model/
│   ├── backbone.py        # 백본 (Swin Transformer)
│   └── reid_model.py      # ReID 모델
├── loss/
│   └── losses.py          # 손실 함수
└── utils/
    └── metrics.py         # 평가 메트릭
```

## 설치

```bash
# 필요한 패키지
pip install lightning torch torchvision timm wandb
```

## 사용법

### 학습

```bash
python train.py \
    --dataset_name market1501 \
    --data_root /path/to/data \
    --model_name swin_tiny \
    --pretrain_path /path/to/pretrained.pth \
    --semantic_weight 1.0 \
    --batch_size 64 \
    --num_instances 16 \
    --max_epochs 100 \
    --base_lr 3e-4 \
    --output_dir ./outputs
```

### 주요 파라미터

- `--dataset_name`: 데이터셋 이름 (market1501, msmt17)
- `--model_name`: 모델 이름 (swin_tiny, swin_small, swin_base)
- `--semantic_weight`: SOLIDER semantic weight (기본값: 1.0)
- `--batch_size`: 배치 크기
- `--num_instances`: Identity당 인스턴스 수
- `--max_epochs`: 최대 에폭 수
- `--base_lr`: 기본 학습률
- `--devices`: GPU 개수 (None이면 자동)
- `--num_nodes`: 노드 개수

### DDP 학습

```bash
# 4개 GPU 사용
python train.py \
    --dataset_name market1501 \
    --data_root /path/to/data \
    --devices 4 \
    --batch_size 64
```

## 설정

모든 설정은 `config.py`에서 관리되며, 명령줄 인자로 오버라이드할 수 있습니다.

## 모델 구조

- **Backbone**: Swin Transformer (SOLIDER pretrained)
- **Neck**: BNNeck
- **Head**: Linear classifier
- **Loss**: ID Loss (Cross-Entropy) + Triplet Loss

## SOLIDER Semantic Weight

SOLIDER의 핵심 기능인 semantic weight는 백본의 각 stage에 적용됩니다:

```python
# semantic_weight는 appearance와 semantic의 균형을 조절
# 0.0: appearance만 사용
# 1.0: semantic만 사용
# 0.2: 일반적으로 사용되는 값
```

## 평가 메트릭

- **mAP**: Mean Average Precision
- **Rank-1, Rank-5, Rank-10**: Cumulative Matching Characteristics

## 체크포인트

체크포인트는 `output_dir/checkpoints/`에 저장되며:
- `reid-{epoch}-{val_mAP}.ckpt`: 최고 성능 모델
- `last.ckpt`: 마지막 에폭 모델

## 참고

- 원본 SOLIDER-REID 코드: `SOLIDER-REID/`
- SOLIDER pretrained 모델은 별도로 다운로드 필요
- 데이터셋은 Market1501 또는 MSMT17 형식이어야 함



