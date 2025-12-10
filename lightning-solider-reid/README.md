# SOLIDER-REID Lightning

PyTorch Lightning 기반의 SOLIDER-REID 학습 코드입니다.

## 특징

- **PyTorch Lightning 기반**: 간단하고 확장 가능한 학습 코드
- **mmcv 의존성 제거**: 순수 PyTorch 기반 구현
- **독립 실행**: SOLIDER-REID 디렉토리 없이도 동작 (필요한 코드 모두 포함)
- **기존 코드 재사용**: SOLIDER-REID의 모델, loss, 데이터셋 코드를 그대로 활용

## 설치

```bash
# Lightning 설치
pip install lightning

# 기타 의존성
pip install torch torchvision
```

## 사용법

### 기본 학습

```bash
python train_reid.py \
    --transformer_type swin_base_patch4_window7_224 \
    --pretrain_path path/to/checkpoint_tea.pth \
    --pretrain_choice self \
    --semantic_weight 0.2 \
    --dataset_name msmt17 \
    --root_dir path/to/msmt17 \
    --base_lr 0.0002 \
    --optimizer_name SGD \
    --max_epochs 120 \
    --warmup_epochs 20 \
    --warmup_method cosine \
    --output_dir ./log/reid \
    --devices 1
```

### 주요 파라미터

#### 모델 파라미터
- `--transformer_type`: Transformer 백본 타입 (swin_base/small/tiny)
- `--pretrain_path`: Pretrained model 경로 (필수)
- `--pretrain_choice`: 'self' (SOLIDER) 또는 'imagenet'
- `--semantic_weight`: Semantic weight (기본: 0.2)
- `--id_loss_type`: ID loss 타입 (softmax/arcface/cosface/amsoftmax/circle)

#### 데이터 파라미터
- `--dataset_name`: 데이터셋 이름 (msmt17/market1501/mm/custom)
- `--root_dir`: 데이터셋 루트 디렉토리
- `--sampler`: 샘플러 타입 (softmax_triplet/softmax/id_triplet/id)
- `--num_instance`: 배치당 identity당 인스턴스 수

#### 학습 파라미터
- `--base_lr`: 기본 학습률
- `--optimizer_name`: Optimizer (SGD/AdamW/Adam)
- `--max_epochs`: 최대 epoch 수
- `--warmup_epochs`: Warmup epoch 수
- `--warmup_method`: Warmup 방법 (cosine/linear/constant)
- `--weight_decay`: Weight decay

#### Loss 파라미터
- `--metric_loss_type`: Metric loss 타입 (triplet)
- `--no_margin`: Soft triplet loss 사용 (margin 없음)
- `--id_loss_weight`: ID loss 가중치
- `--triplet_loss_weight`: Triplet loss 가중치
- `--cosine_scale`: Cosine-based loss scale
- `--cosine_margin`: Cosine-based loss margin

#### Lightning 파라미터
- `--devices`: GPU 개수
- `--num_nodes`: 노드 개수
- `--precision`: 정밀도 (32/16-mixed/bf16-mixed)
- `--resume`: 체크포인트에서 재개

## 디렉토리 구조

```
lightning-solider-reid/
├── train_reid.py              # 메인 학습 스크립트
├── config/
│   ├── __init__.py
│   └── reid_args.py           # Argument parser
├── reid_module/
│   ├── __init__.py
│   └── reid_lightning_module.py  # LightningModule
├── reid_data/
│   ├── __init__.py
│   └── reid_data_module.py    # LightningDataModule
└── utils/
    ├── __init__.py
    └── swin_utils.py          # Utility functions
```

## 기존 코드와의 차이점

1. **Config 관리**: YAML 대신 argparse 사용
2. **학습 루프**: Lightning의 `training_step` 사용
3. **Validation**: Lightning의 `validation_step` 사용
4. **Mixed Precision**: Lightning의 `precision` 파라미터 사용
5. **Distributed Training**: Lightning이 자동 처리

## 체크포인트

체크포인트는 `{output_dir}/checkpoints/`에 저장됩니다.
- Lightning checkpoint: `reid-{epoch:02d}-{val_mAP:.4f}.ckpt` (상위 3개 모델만 저장, val_mAP 기준)
- PyTorch state_dict: `{transformer_type}_{epoch}.pth` (주기적으로 저장, checkpoint_period마다)

## 로그

TensorBoard 로그는 `{output_dir}/logs/`에 저장됩니다.

```bash
tensorboard --logdir ./log/reid/logs
```

## 커스텀 데이터셋 사용법

커스텀 데이터셋을 사용하려면 다음과 같은 디렉토리 구조를 준비하세요:

```
root_dir/
    ID1/
        image1.jpg
        image2.jpg
        ...
    ID2/
        image1.jpg
        image2.jpg
        ...
    ID3/
        ...
```

각 ID 디렉토리 이름이 person ID가 되며, 해당 디렉토리 내의 이미지들이 해당 ID의 이미지입니다.

사용 예시:

```bash
python train_reid.py \
    --transformer_type swin_base_patch4_window7_224 \
    --pretrain_path path/to/checkpoint_tea.pth \
    --pretrain_choice self \
    --semantic_weight 0.2 \
    --dataset_name custom \
    --root_dir path/to/your/custom/dataset \
    --base_lr 0.0002 \
    --optimizer_name SGD \
    --max_epochs 120 \
    --warmup_epochs 20 \
    --warmup_method cosine \
    --output_dir ./log/custom \
    --devices 1
```

**참고**: 커스텀 데이터셋은 자동으로 train/query/gallery를 분리합니다.
- Training: 모든 이미지 사용 (기본값)
- Query: 각 ID의 20% 이미지 (기본값)
- Gallery: 각 ID의 80% 이미지 (기본값)

## 주의사항

1. **Pretrained Model 필수**: `--pretrain_path`는 반드시 지정해야 합니다.
   - 일반 PyTorch `.pth` 파일을 사용합니다.
   - `state_dict` 키가 있으면 자동으로 추출합니다.
   - `teacher` 키가 있으면 자동으로 사용합니다 (SOLIDER/DINO 모델).
2. **데이터셋 경로**: `--root_dir`에 올바른 데이터셋 경로를 지정하세요.
3. **GPU 메모리**: 배치 크기와 이미지 크기에 따라 GPU 메모리를 조정하세요.

## Checkpoint 형식

- **학습 중 저장**: `.pth` 파일로 모델의 `state_dict`를 저장합니다.
- **Pretrained 모델**: `.pth` 파일 형식의 state_dict를 로드합니다.
  - 직접 state_dict 또는 `state_dict` 키 포함
  - `teacher` 키 지원 (SOLIDER/DINO 모델)
  - `model` 키 지원

