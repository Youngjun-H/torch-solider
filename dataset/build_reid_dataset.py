#!/usr/bin/env python3
"""
ReID 데이터셋 구축 스크립트

Training: 2025-10-15의 각 ID당 최대 500장을 층별로 균일하게 선택
Validation: 2025-10-16의 각 ID를 query/gallery로 분할
  - Query: 각 층별로 2장씩 선택
  - Gallery: 각 ID당 최대 80장을 층별로 균일하게 선택 (query 제외 후)

모든 선택은 층별로 균일하게 분배되며, 각 층 내에서는 랜덤하게 샘플링됩니다.
"""
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# 이미지 확장자 정의
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".ppm",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".PPM",
)


def is_image_file(filename: str) -> bool:
    """파일이 이미지 파일인지 확인"""
    return filename.lower().endswith(IMG_EXTENSIONS)


def extract_floor_from_filename(filename: str) -> str:
    """
    파일명에서 층 정보 추출

    예: 2025-10-15_1F_07-30-00_1_id356_frame_135757.jpg -> "1F"
    예: 2025-10-16_2F-out_07-30-00_1_id79_frame_051330.jpg -> "2F-out"
    """
    # 파일명에서 날짜 다음의 층 정보 추출
    # 패턴: 날짜_층정보_...
    match = re.match(r"\d{4}-\d{2}-\d{2}_([^_]+)_", filename)
    if match:
        return match.group(1)
    return "unknown"


def find_all_images(root_path: str) -> List[str]:
    """주어진 경로에서 모든 이미지 파일을 찾습니다."""
    image_files = []
    root = Path(root_path)

    if not root.exists():
        raise ValueError(f"경로가 존재하지 않습니다: {root_path}")

    print(f"디렉토리 탐색 시작: {root_path}")
    sys.stdout.flush()

    try:
        for file_path in tqdm(
            root.rglob("*"), desc="이미지 파일 탐색", unit="항목", file=sys.stdout
        ):
            try:
                if file_path.is_file() and is_image_file(file_path.name):
                    image_files.append(str(file_path))
            except (OSError, PermissionError) as err:
                continue
            except Exception as err:
                tqdm.write(f"경고: {file_path} 처리 중 오류: {err}")
                continue
    except Exception as e:
        print(f"경고: 탐색 중 오류 발생: {e}")
        sys.stdout.flush()
        # 재시도
        for file_path in root.rglob("*"):
            try:
                if file_path.is_file() and is_image_file(file_path.name):
                    image_files.append(str(file_path))
            except (OSError, PermissionError) as err:
                continue
            except Exception as err:
                continue

    print(f"발견된 이미지 파일: {len(image_files)}개")
    sys.stdout.flush()
    return sorted(image_files)


def organize_by_id_and_floor(image_paths: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    이미지를 ID별, 층별로 정리

    Returns:
        {id_dir_name: {floor: [image_paths]}}
    """
    organized = defaultdict(lambda: defaultdict(list))

    for img_path in tqdm(image_paths, desc="ID 및 층별 정리", leave=False):
        # ID 디렉토리 이름 추출
        path_obj = Path(img_path)
        id_dir = path_obj.parent.name

        # 파일명에서 층 정보 추출
        floor = extract_floor_from_filename(path_obj.name)

        organized[id_dir][floor].append(img_path)

    return organized


def sample_images_uniformly_by_floor(
    floor_dict: Dict[str, List[str]],
    max_images: int,
    seed: int = 42,
) -> List[str]:
    """
    층별로 균일하게 이미지를 샘플링

    Args:
        floor_dict: {floor: [image_paths]} 형식의 딕셔너리
        max_images: 최대 선택할 이미지 수
        seed: 랜덤 시드

    Returns:
        선택된 이미지 경로 리스트
    """
    random.seed(seed)

    # 각 층의 이미지를 랜덤하게 섞기
    shuffled_floors = {}
    for floor, img_paths in floor_dict.items():
        shuffled = img_paths.copy()
        random.shuffle(shuffled)
        shuffled_floors[floor] = shuffled

    # 층별로 균등하게 분배
    num_floors = len(shuffled_floors)
    if num_floors == 0:
        return []

    # 각 층당 기본 할당량 계산
    per_floor_base = max_images // num_floors
    remainder = max_images % num_floors

    selected_images = []
    floor_indices = {}  # 각 층에서 현재 선택한 인덱스

    # 각 층에 기본 할당량 분배
    for floor_idx, (floor, img_paths) in enumerate(shuffled_floors.items()):
        floor_indices[floor] = 0
        # 나머지가 있으면 처음 몇 개 층에 1개씩 추가
        num_to_take = per_floor_base + (1 if floor_idx < remainder else 0)
        num_to_take = min(num_to_take, len(img_paths))
        selected_images.extend(img_paths[:num_to_take])
        floor_indices[floor] = num_to_take

    # 아직 부족하면 층별로 순환하며 추가 선택
    while len(selected_images) < max_images:
        added = False
        for floor, img_paths in shuffled_floors.items():
            if len(selected_images) >= max_images:
                break
            if floor_indices[floor] < len(img_paths):
                selected_images.append(img_paths[floor_indices[floor]])
                floor_indices[floor] += 1
                added = True
        if not added:
            break  # 더 이상 추가할 이미지가 없음

    # 최종적으로 랜덤하게 섞기 (층별 순서를 완전히 섞기)
    random.shuffle(selected_images)

    return selected_images[:max_images]


def split_validation_data(
    organized_data: Dict[str, Dict[str, List[str]]],
    query_per_floor: int = 2,
    gallery_max_per_id: int = 80,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Validation 데이터를 query/gallery로 분할

    Args:
        organized_data: {id_dir: {floor: [image_paths]}}
        query_per_floor: 각 층별로 선택할 query 이미지 수
        gallery_max_per_id: 각 ID당 최대 gallery 이미지 수
        seed: 랜덤 시드

    Returns:
        (query_dict, gallery_dict): {id_dir: [image_paths]}
    """
    random.seed(seed)

    query_dict = defaultdict(list)
    gallery_dict = defaultdict(list)

    for id_dir, floor_dict in tqdm(
        organized_data.items(), desc="Query/Gallery 분할", leave=False
    ):
        query_paths = []
        remaining_floor_dict = defaultdict(list)

        # 각 층별로 query 선택
        for floor, img_paths in floor_dict.items():
            if len(img_paths) < query_per_floor:
                # 이미지가 부족하면 모두 query로
                query_paths.extend(img_paths)
            else:
                # 랜덤하게 query_per_floor개 선택
                shuffled = img_paths.copy()
                random.shuffle(shuffled)
                query_paths.extend(shuffled[:query_per_floor])
                # 나머지는 gallery 후보로 저장
                remaining_floor_dict[floor] = shuffled[query_per_floor:]

        # Query가 있으면 저장
        if len(query_paths) > 0:
            query_dict[id_dir] = query_paths

        # Gallery는 층별로 균일하게 최대 gallery_max_per_id개 선택
        if len(remaining_floor_dict) > 0:
            gallery_paths = sample_images_uniformly_by_floor(
                remaining_floor_dict, gallery_max_per_id, seed=seed
            )
            if len(gallery_paths) > 0:
                gallery_dict[id_dir] = gallery_paths

    return dict(query_dict), dict(gallery_dict)


def copy_dataset_structure(
    source_data: Dict[str, List[str]], output_dir: str, dataset_type: str
):
    """
    데이터셋을 CustomDataset 형식으로 복사

    Args:
        source_data: {id_dir: [image_paths]}
        output_dir: 출력 디렉토리
        dataset_type: 'train', 'query', 'gallery'
    """
    output_path = Path(output_dir) / dataset_type
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{dataset_type.upper()}] 데이터셋 복사 중...")
    sys.stdout.flush()

    total_copied = 0
    for id_dir, img_paths in tqdm(
        source_data.items(), desc=f"복사 중 ({dataset_type})", file=sys.stdout
    ):
        id_output_dir = output_path / id_dir
        id_output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            src_path = Path(img_path)
            dst_path = id_output_dir / src_path.name

            # 파일명 중복 처리
            if dst_path.exists():
                # 파일명에 인덱스 추가
                stem = src_path.stem
                suffix = src_path.suffix
                counter = 1
                while dst_path.exists():
                    new_name = f"{stem}_{counter}{suffix}"
                    dst_path = id_output_dir / new_name
                    counter += 1

            try:
                shutil.copy2(src_path, dst_path)
                total_copied += 1
            except Exception as e:
                tqdm.write(f"경고: 파일 복사 실패 ({img_path}): {e}")
                continue

    print(
        f"[{dataset_type.upper()}] 완료: {len(source_data)}개 ID, 총 {total_copied}개 이미지"
    )
    sys.stdout.flush()


def print_statistics(
    train_data: Dict[str, List[str]],
    query_data: Dict[str, List[str]],
    gallery_data: Dict[str, List[str]],
):
    """데이터셋 통계 출력"""
    print("\n" + "=" * 60)
    print("데이터셋 통계")
    print("=" * 60)

    # Training 통계
    train_ids = len(train_data)
    train_images = sum(len(paths) for paths in train_data.values())
    print(f"\n[Training]")
    print(f"  ID 수: {train_ids}")
    print(f"  이미지 수: {train_images}")
    if train_ids > 0:
        print(f"  ID당 평균 이미지: {train_images/train_ids:.1f}")

    # Query 통계
    query_ids = len(query_data)
    query_images = sum(len(paths) for paths in query_data.values())
    print(f"\n[Query]")
    print(f"  ID 수: {query_ids}")
    print(f"  이미지 수: {query_images}")
    if query_ids > 0:
        print(f"  ID당 평균 이미지: {query_images/query_ids:.1f}")

    # Gallery 통계
    gallery_ids = len(gallery_data)
    gallery_images = sum(len(paths) for paths in gallery_data.values())
    print(f"\n[Gallery]")
    print(f"  ID 수: {gallery_ids}")
    print(f"  이미지 수: {gallery_images}")
    if gallery_ids > 0:
        print(f"  ID당 평균 이미지: {gallery_images/gallery_ids:.1f}")

    # Query-Gallery 겹침 확인
    query_id_set = set(query_data.keys())
    gallery_id_set = set(gallery_data.keys())
    overlap = query_id_set & gallery_id_set
    print(f"\n[Query-Gallery 겹침]")
    print(f"  Query에만 있는 ID: {len(query_id_set - gallery_id_set)}")
    print(f"  Gallery에만 있는 ID: {len(gallery_id_set - query_id_set)}")
    print(f"  겹치는 ID: {len(overlap)}")

    if len(query_id_set - gallery_id_set) > 0:
        print(f"  ⚠️  경고: Query에만 있는 ID가 있습니다! (ReID 평가 불가)")
        print(f"      ID 목록: {sorted(query_id_set - gallery_id_set)[:10]}...")

    # 층별 통계 (Query)
    if query_data:
        print(f"\n[Query 층별 통계]")
        floor_stats = defaultdict(int)
        for id_dir, img_paths in query_data.items():
            for img_path in img_paths:
                floor = extract_floor_from_filename(Path(img_path).name)
                floor_stats[floor] += 1
        for floor, count in sorted(floor_stats.items()):
            print(f"  {floor}: {count}개 이미지")

    print("=" * 60)
    sys.stdout.flush()


def build_reid_dataset(
    train_dir: str,
    val_dir: str,
    output_dir: str,
    train_max_per_id: int = 500,
    query_per_floor: int = 2,
    gallery_max_per_id: int = 80,
    seed: int = 42,
):
    """
    ReID 데이터셋 구축

    Args:
        train_dir: Training 이미지 디렉토리 (2025-10-15)
        val_dir: Validation 이미지 디렉토리 (2025-10-16)
        output_dir: 출력 디렉토리
        train_max_per_id: 각 ID당 최대 training 이미지 수 (기본값: 500)
        query_per_floor: 각 층별로 선택할 query 이미지 수 (기본값: 2)
        gallery_max_per_id: 각 ID당 최대 gallery 이미지 수 (기본값: 80)
        seed: 랜덤 시드
    """
    print("=" * 60)
    print("ReID 데이터셋 구축 시작")
    print("=" * 60)
    sys.stdout.flush()

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"출력 디렉토리가 이미 존재합니다: {output_dir}")
        print("기존 디렉토리를 유지하고 계속 진행합니다...")
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Training 데이터 수집
    print("\n[1단계] Training 데이터 수집...")
    sys.stdout.flush()
    train_images = find_all_images(train_dir)
    train_organized = organize_by_id_and_floor(train_images)

    # Training은 각 ID당 train_max_per_id장, 층별로 균일하게 선택
    train_data = {}
    for id_dir, floor_dict in tqdm(
        train_organized.items(), desc="Training 데이터 샘플링", leave=False
    ):
        selected_images = sample_images_uniformly_by_floor(
            floor_dict, train_max_per_id, seed=seed
        )
        if len(selected_images) > 0:
            train_data[id_dir] = selected_images

    print(
        f"Training: {len(train_data)}개 ID, 총 {sum(len(imgs) for imgs in train_data.values())}개 이미지"
    )
    sys.stdout.flush()

    # 2. Validation 데이터 수집 및 분할
    print("\n[2단계] Validation 데이터 수집 및 분할...")
    sys.stdout.flush()
    val_images = find_all_images(val_dir)
    val_organized = organize_by_id_and_floor(val_images)

    # Query/Gallery 분할 (층별로 query_per_floor개씩, gallery는 ID당 gallery_max_per_id장)
    query_data, gallery_data = split_validation_data(
        val_organized,
        query_per_floor=query_per_floor,
        gallery_max_per_id=gallery_max_per_id,
        seed=seed,
    )

    print(
        f"Query: {len(query_data)}개 ID, 총 {sum(len(imgs) for imgs in query_data.values())}개 이미지"
    )
    print(
        f"Gallery: {len(gallery_data)}개 ID, 총 {sum(len(imgs) for imgs in gallery_data.values())}개 이미지"
    )
    sys.stdout.flush()

    # 3. 통계 출력
    print_statistics(train_data, query_data, gallery_data)

    # 4. 데이터셋 복사
    print("\n[3단계] 데이터셋 복사...")
    sys.stdout.flush()

    copy_dataset_structure(train_data, output_dir, "train")
    copy_dataset_structure(query_data, output_dir, "query")
    copy_dataset_structure(gallery_data, output_dir, "gallery")

    print("\n" + "=" * 60)
    print("데이터셋 구축 완료!")
    print("=" * 60)
    print(f"출력 디렉토리: {output_dir}")
    print("=" * 60)
    sys.stdout.flush()

    # 최종 구조 안내
    print("\n데이터셋 구조:")
    print(f"  {output_dir}/")
    print(f"    train/")
    print(f"      ID1/")
    print(f"        image1.jpg")
    print(f"        ...")
    print(f"    query/")
    print(f"      ID1/")
    print(f"        image1.jpg")
    print(f"        ...")
    print(f"    gallery/")
    print(f"      ID1/")
    print(f"        image1.jpg")
    print(f"        ...")
    sys.stdout.flush()


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="ReID 데이터셋 구축")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-15",
        help="Training 이미지 디렉토리",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-16",
        help="Validation 이미지 디렉토리",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="출력 디렉토리")
    parser.add_argument(
        "--train_max_per_id",
        type=int,
        default=500,
        help="각 ID당 최대 training 이미지 수 (기본값: 500)",
    )
    parser.add_argument(
        "--query_per_floor",
        type=int,
        default=2,
        help="각 층별로 선택할 query 이미지 수 (기본값: 2)",
    )
    parser.add_argument(
        "--gallery_max_per_id",
        type=int,
        default=80,
        help="각 ID당 최대 gallery 이미지 수 (기본값: 80)",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본값: 42)")

    args = parser.parse_args()

    # SLURM 환경에서 출력 버퍼링 비활성화
    (
        sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stdout, "reconfigure")
        else None
    )

    try:
        build_reid_dataset(
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            output_dir=args.output_dir,
            train_max_per_id=args.train_max_per_id,
            query_per_floor=args.query_per_floor,
            gallery_max_per_id=args.gallery_max_per_id,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("\n\n작업이 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        import traceback

        print(f"\n\n치명적 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
