# Dataset Scripts

데이터셋 구축 및 관리 스크립트 모음

## 스크립트 목록

### 1. `build_reid_dataset.py`
ReID 데이터셋 구축 스크립트

**기능:**
- Training 데이터: 지정된 디렉토리의 모든 ID 이미지를 train으로 사용
- Validation 데이터: 각 ID의 이미지를 층별로 분류하여 Query/Gallery로 분할
  - Query: 각 층별로 지정된 수만큼 선택 (기본 2장)
  - Gallery: 나머지 모든 이미지

**사용법:**
```bash
python build_reid_dataset.py \
  --train_dir "/path/to/train/images" \
  --val_dir "/path/to/val/images" \
  --output_dir "/path/to/output" \
  --query_per_floor 2 \
  --seed 42
```

**출력 구조:**
```
output_dir/
  train/
    ID1/
      image1.jpg
      ...
  query/
    ID1/
      image1.jpg  (각 층별 2장씩)
      ...
  gallery/
    ID1/
      image1.jpg  (나머지 모든 이미지)
      ...
```

**특징:**
- 파일명에서 층 정보 자동 추출 (예: `1F`, `2F-out`)
- Query-Gallery ID 겹침 확인
- 층별 통계 정보 출력

---

### 2. `merge_datasets.py`
기존 데이터셋과 새로운 데이터 병합 스크립트

**기능:**
- 기존 데이터셋을 출력 디렉토리로 복사
- 새로운 데이터에서 이미지 파일을 찾아 병합
- 클래스별 폴더 구조 유지

**사용법:**
```bash
python merge_datasets.py \
  --existing_path "/path/to/existing/dataset" \
  --new_data_path "/path/to/new/data" \
  --output_path "/path/to/output"
```

**특징:**
- ImageFolder 형식 지원 (클래스별 서브디렉토리)
- 파일명 중복 자동 처리
- SLURM 환경 지원

---

### 3. `copy_images.py`
이미지 파일 복사 스크립트

**기능:**
- 지정된 디렉토리에서 모든 이미지 파일을 재귀적으로 탐색
- 찾은 이미지 파일을 출력 디렉토리로 복사

**사용법:**
```bash
python copy_images.py \
  --input_path "/path/to/input" \
  --output_path "/path/to/output"
```

**특징:**
- 최하위 디렉토리까지 재귀 탐색
- 파일명 중복 자동 처리
- 진행 상황 표시 (tqdm)

---

## 공통 사항

### 지원 이미지 형식
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.ppm`, `.tif`, `.tiff`
- 대소문자 구분 없음

### SLURM 실행
각 스크립트에 대응하는 `.sh` 파일이 있습니다:
- `build_reid_dataset.sh`
- `merge_datasets.sh` (merge_datasets.py용)
- `copy.sh` (copy_images.py용)

```bash
sbatch build_reid_dataset.sh
```

### 출력 버퍼링
SLURM 환경에서 출력 버퍼링이 자동으로 비활성화되어 실시간 로그 확인이 가능합니다.

---

## 예제

### ReID 데이터셋 구축
```bash
cd dataset
python build_reid_dataset.py \
  --train_dir "/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-15" \
  --val_dir "/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-16" \
  --output_dir "/purestorage/AILAB/AI_2/datasets/PersonReID/datasets/cctv_reid_dataset" \
  --query_per_floor 2 \
  --seed 42
```

### 데이터셋 병합
```bash
python merge_datasets.py \
  --existing_path "/path/to/existing" \
  --new_data_path "/path/to/new" \
  --output_path "/path/to/merged"
```

### 이미지 복사
```bash
python copy_images.py \
  --input_path "/path/to/source" \
  --output_path "/path/to/destination"
```

