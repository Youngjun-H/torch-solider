#!/usr/bin/env python3
"""
데이터셋 병합 스크립트

기존 데이터셋과 새로운 데이터를 병합하여 하나의 새로운 데이터셋을 생성합니다.
"""
import shutil
import sys
from pathlib import Path
from typing import List, Set

import torch
from torch import nn
from tqdm import tqdm

model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000),
)

model.to("cuda")

# 이미지 확장자 정의
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".TIF",
    ".TIFF",
)


def is_image_file(filename: str) -> bool:
    """파일이 이미지 파일인지 확인"""
    return filename.lower().endswith(IMG_EXTENSIONS)


def find_all_images(root_path: str) -> List[str]:
    """
    주어진 경로에서 최하위까지 탐색하여 모든 이미지 파일을 찾습니다.

    Args:
        root_path: 탐색할 루트 경로

    Returns:
        찾은 모든 이미지 파일의 절대 경로 리스트
    """
    image_files = []
    root = Path(root_path)

    if not root.exists():
        raise ValueError(f"경로가 존재하지 않습니다: {root_path}")

    # 최하위까지 재귀적으로 탐색 (메모리 효율적으로)
    # rglob를 제너레이터로 사용하여 메모리에 모든 경로를 저장하지 않음
    print(f"디렉토리 탐색 시작: {root_path}")
    sys.stdout.flush()

    try:
        # rglob를 제너레이터로 사용하여 메모리 효율적으로 처리
        all_paths = root.rglob("*")
        for file_path in tqdm(
            all_paths, desc="이미지 파일 탐색 중", unit="항목", file=sys.stdout
        ):
            try:
                if file_path.is_file() and is_image_file(file_path.name):
                    image_files.append(str(file_path))
            except (OSError, PermissionError) as err:
                # 개별 파일 접근 오류는 건너뛰고 계속 진행
                continue
    except Exception as e:
        print(f"경고: 탐색 중 오류 발생: {e}")
        sys.stdout.flush()
        # 재시도: 더 안전한 방법으로
        for file_path in root.rglob("*"):
            try:
                if file_path.is_file() and is_image_file(file_path.name):
                    image_files.append(str(file_path))
            except (OSError, PermissionError) as err:
                continue
            except Exception as err:
                print(f"경고: {file_path} 처리 중 오류: {err}")
                continue

    print(f"발견된 이미지 파일: {len(image_files)}개")
    sys.stdout.flush()
    return sorted(image_files)


def copy_existing_dataset(existing_path: str, output_path: str) -> int:
    """
    기존 데이터셋을 출력 디렉토리로 복사합니다.

    Args:
        existing_path: 기존 데이터셋 경로
        output_path: 출력 디렉토리 경로

    Returns:
        복사된 파일 개수
    """
    existing = Path(existing_path)
    output = Path(output_path)

    if not existing.exists():
        raise ValueError(f"기존 데이터셋 경로가 존재하지 않습니다: {existing_path}")

    copied_count = 0

    # 기존 데이터셋이 ImageFolder 형식인지 확인 (클래스별 서브디렉토리)
    if existing.is_dir():
        # ImageFolder 형식인 경우 (클래스별 폴더 구조)
        subdirs = [d for d in existing.iterdir() if d.is_dir()]
        if subdirs:
            # 클래스별 폴더 구조가 있는 경우
            for class_dir in tqdm(subdirs, desc="클래스별 복사 중", unit="클래스"):
                output_class_dir = output / class_dir.name
                output_class_dir.mkdir(parents=True, exist_ok=True)

                # 해당 클래스의 모든 이미지 파일 수집
                image_files = [
                    f
                    for f in class_dir.iterdir()
                    if f.is_file() and is_image_file(f.name)
                ]

                for file_path in tqdm(
                    image_files,
                    desc=f"  [{class_dir.name}]",
                    leave=False,
                    unit="파일",
                    file=sys.stdout,
                ):
                    dest_path = output_class_dir / file_path.name
                    # 중복 파일명 처리
                    if dest_path.exists():
                        base_name = file_path.stem
                        ext = file_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = output_class_dir / f"{base_name}_{counter}{ext}"
                            counter += 1
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
        else:
            # 단일 폴더 구조인 경우
            image_files = [
                f for f in existing.iterdir() if f.is_file() and is_image_file(f.name)
            ]

            for file_path in tqdm(
                image_files, desc="기존 데이터 복사 중", unit="파일", file=sys.stdout
            ):
                dest_path = output / file_path.name
                # 중복 파일명 처리
                if dest_path.exists():
                    base_name = file_path.stem
                    ext = file_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = output / f"{base_name}_{counter}{ext}"
                        counter += 1
                shutil.copy2(file_path, dest_path)
                copied_count += 1

    return copied_count


def copy_new_images(
    new_images: List[str], output_path: str, prefix: str = "new"
) -> int:
    """
    새로운 이미지 파일들을 출력 디렉토리로 복사합니다.

    Args:
        new_images: 복사할 이미지 파일 경로 리스트
        output_path: 출력 디렉토리 경로
        prefix: 파일명 충돌 시 사용할 접두사

    Returns:
        복사된 파일 개수
    """
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    used_names: Set[str] = set()

    for img_path in tqdm(
        new_images, desc="새로운 데이터 복사 중", unit="파일", file=sys.stdout
    ):
        src_path = Path(img_path)
        if not src_path.exists():
            tqdm.write(f"경고: 파일이 존재하지 않습니다: {img_path}")
            continue

        # 원본 파일명 사용
        dest_filename = src_path.name

        # 중복 파일명 처리
        if dest_filename in used_names or (output / dest_filename).exists():
            base_name = src_path.stem
            ext = src_path.suffix
            counter = 1
            while dest_filename in used_names or (output / dest_filename).exists():
                dest_filename = f"{prefix}_{base_name}_{counter}{ext}"
                counter += 1

        dest_path = output / dest_filename
        shutil.copy2(src_path, dest_path)
        used_names.add(dest_filename)
        copied_count += 1

    return copied_count


def build_dataset(existing_path: str, new_data_path: str, output_path: str):
    """
    기존 데이터셋과 새로운 데이터를 병합하여 새로운 데이터셋을 생성합니다.

    Args:
        existing_path: 기존 데이터셋 경로
        new_data_path: 추가할 데이터 경로
        output_path: 생성할 새로운 데이터셋 경로
    """
    print("=" * 60)
    print("데이터셋 병합 시작")
    print("=" * 60)

    # 출력 디렉토리 생성
    output = Path(output_path)
    if output.exists():
        # SLURM 배치 환경에서는 자동으로 덮어쓰기 (input()이 작동하지 않음)
        print(f"출력 디렉토리가 이미 존재합니다: {output_path}")
        print("자동으로 기존 디렉토리를 삭제하고 새로 생성합니다...")
        try:
            shutil.rmtree(output)
            print("기존 디렉토리 삭제 완료")
        except Exception as e:
            print(f"경고: 디렉토리 삭제 중 오류 발생: {e}")
            print("기존 디렉토리를 유지하고 계속 진행합니다...")
    output.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리 생성 완료: {output_path}")

    # 1. 기존 데이터셋 복사
    print(f"\n[1/2] 기존 데이터셋 복사 중: {existing_path}")
    print(f"출력 경로: {output_path}")
    sys.stdout.flush()  # SLURM 환경에서 출력 버퍼 플러시

    try:
        existing_count = copy_existing_dataset(existing_path, output_path)
        print(f"\n✓ 기존 데이터셋 복사 완료: {existing_count}개 파일")
        sys.stdout.flush()
    except Exception as e:
        import traceback

        print(f"\n✗ 기존 데이터셋 복사 실패: {e}")
        print("상세 오류 정보:")
        traceback.print_exc()
        sys.stdout.flush()
        return

    # 2. 새로운 데이터에서 이미지 찾기 및 복사
    print(f"\n[2/2] 새로운 데이터 탐색 및 복사 중: {new_data_path}")
    sys.stdout.flush()

    try:
        new_images = find_all_images(new_data_path)
        print(f"\n  발견된 이미지 파일: {len(new_images)}개")
        sys.stdout.flush()

        if new_images:
            new_count = copy_new_images(new_images, output_path, prefix="new")
            print(f"\n✓ 새로운 데이터 복사 완료: {new_count}개 파일")
            sys.stdout.flush()
        else:
            print("\n  경고: 새로운 데이터에서 이미지 파일을 찾을 수 없습니다.")
            sys.stdout.flush()
    except Exception as e:
        import traceback

        print(f"\n✗ 새로운 데이터 처리 실패: {e}")
        print("상세 오류 정보:")
        traceback.print_exc()
        sys.stdout.flush()
        return

    # 결과 요약
    total_files = existing_count + (len(new_images) if new_images else 0)
    print("\n" + "=" * 60)
    print("데이터셋 병합 완료!")
    print("=" * 60)
    print(f"출력 경로: {output_path}")
    print(f"기존 데이터: {existing_count}개 파일")
    print(f"새로운 데이터: {len(new_images)}개 파일")
    print(f"총 파일 수: {total_files}개")
    print("=" * 60)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="기존 데이터셋과 새로운 데이터를 병합하여 새로운 데이터셋을 생성합니다."
    )
    parser.add_argument(
        "--existing_path", type=str, required=True, help="기존 데이터셋 경로"
    )
    parser.add_argument(
        "--new_data_path", type=str, required=True, help="추가할 데이터 경로"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="생성할 새로운 데이터셋 경로"
    )

    args = parser.parse_args()

    # SLURM 환경에서 출력 버퍼링 비활성화
    (
        sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stdout, "reconfigure")
        else None
    )

    try:
        build_dataset(
            existing_path=args.existing_path,
            new_data_path=args.new_data_path,
            output_path=args.output_path,
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
    # 예제 사용법 (직접 실행 시)
    # build_dataset(
    #     existing_path="/purestorage/AILAB/AI_2/datasets/PersonReID/solider_surv_pre/image",
    #     new_data_path="/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-15",
    #     output_path="/purestorage/AILAB/AI_2/datasets/PersonReID/solider_surv_pre/image_merged"
    # )

    main()
