#!/usr/bin/env python3
"""
이미지 파일 복사 스크립트

디렉토리를 입력받아 최하위까지 탐색하여 이미지 파일만 찾아서 새로운 디렉토리에 복사합니다.
"""
import shutil
import sys
from pathlib import Path
from typing import List, Set

from tqdm import tqdm

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


def copy_images_to_directory(
    image_files: List[str], output_path: str, prefix: str = "copy"
) -> int:
    """
    이미지 파일들을 출력 디렉토리로 복사합니다.

    Args:
        image_files: 복사할 이미지 파일 경로 리스트
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
        image_files, desc="이미지 파일 복사 중", unit="파일", file=sys.stdout
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
        try:
            shutil.copy2(src_path, dest_path)
            used_names.add(dest_filename)
            copied_count += 1
        except Exception as e:
            tqdm.write(f"경고: 파일 복사 실패 ({img_path}): {e}")
            continue

    return copied_count


def copy_images_from_directory(input_path: str, output_path: str):
    """
    입력 디렉토리에서 이미지 파일을 찾아 출력 디렉토리로 복사합니다.

    Args:
        input_path: 입력 디렉토리 경로
        output_path: 출력 디렉토리 경로
    """
    print("=" * 60)
    print("이미지 파일 복사 시작")
    print("=" * 60)

    # 출력 디렉토리 생성
    output = Path(output_path)
    if output.exists():
        print(f"출력 디렉토리가 이미 존재합니다: {output_path}")
        print("기존 디렉토리를 유지하고 계속 진행합니다...")
    output.mkdir(parents=True, exist_ok=True)
    print(f"출력 디렉토리 생성 완료: {output_path}")

    # 이미지 파일 찾기
    print(f"\n이미지 파일 탐색 중: {input_path}")
    sys.stdout.flush()

    try:
        image_files = find_all_images(input_path)
        print(f"\n  발견된 이미지 파일: {len(image_files)}개")
        sys.stdout.flush()

        if image_files:
            copied_count = copy_images_to_directory(
                image_files, output_path, prefix="copy"
            )
            print(f"\n✓ 이미지 파일 복사 완료: {copied_count}개 파일")
            sys.stdout.flush()
        else:
            print("\n  경고: 이미지 파일을 찾을 수 없습니다.")
            sys.stdout.flush()
            return

    except Exception as e:
        import traceback

        print(f"\n✗ 이미지 파일 처리 실패: {e}")
        print("상세 오류 정보:")
        traceback.print_exc()
        sys.stdout.flush()
        return

    # 결과 요약
    print("\n" + "=" * 60)
    print("이미지 파일 복사 완료!")
    print("=" * 60)
    print(f"입력 경로: {input_path}")
    print(f"출력 경로: {output_path}")
    print(f"발견된 이미지: {len(image_files)}개")
    print(f"복사된 파일: {copied_count}개")
    print("=" * 60)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description="디렉토리에서 이미지 파일을 찾아 새로운 디렉토리에 복사합니다."
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="입력 디렉토리 경로"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="출력 디렉토리 경로"
    )

    args = parser.parse_args()

    # SLURM 환경에서 출력 버퍼링 비활성화
    (
        sys.stdout.reconfigure(line_buffering=True)
        if hasattr(sys.stdout, "reconfigure")
        else None
    )

    try:
        copy_images_from_directory(
            input_path=args.input_path, output_path=args.output_path
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
