# encoding: utf-8
"""
Custom Dataset for ReID
Data structure:
    root/
        ID1/
            image1.jpg
            image2.jpg
            ...
        ID2/
            image1.jpg
            ...
"""

import glob
import os
import os.path as osp
import random
import re

from .bases import BaseImageDataset


def extract_floor_from_filename(filename: str):
    """
    파일명에서 층 정보 추출

    예: 2025-10-15_1F_07-30-00_1_id356_frame_135757.jpg -> "1F"
    예: 2025-10-16_2F-out_07-30-00_1_id79_frame_051330.jpg -> "2F-out"

    Returns:
        str: 층 정보 (예: "1F", "2F", "3F", "4F") 또는 None
    """
    # 파일명에서 날짜 다음의 층 정보 추출
    # 패턴: 날짜_층정보_...
    match = re.match(r"\d{4}-\d{2}-\d{2}_([^_]+)_", filename)
    if match:
        floor = match.group(1)
        # 층 정보가 "1F", "2F", "3F", "4F" 형식인지 확인
        if re.match(r"\d+F(-.*)?", floor):
            return floor
    return None


class CustomDataset(BaseImageDataset):
    """
    Custom Dataset for ReID

    Data structure (방식 1 - 자동 분리):
        root/
            ID1/
                image1.jpg
                image2.jpg
                ...
            ID2/
                image1.jpg
                ...

    Data structure (방식 2 - 별도 디렉토리):
        train_dir/
            ID1/
                image1.jpg
                ...
        query_dir/
            ID1/
                image1.jpg
                ...
        gallery_dir/
            ID1/
                image1.jpg
                ...

    Args:
        root: Root directory containing ID subdirectories (방식 1 사용 시)
        train_dir: Training directory containing ID subdirectories (방식 2 사용 시)
        query_dir: Query directory containing ID subdirectories (방식 2 사용 시)
        gallery_dir: Gallery directory containing ID subdirectories (방식 2 사용 시)
        verbose: Whether to print dataset statistics
        pid_begin: Starting PID offset (default: 0)
        train_ratio: Ratio of images per ID to use for training (방식 1만 사용, default: 1.0)
        query_ratio: Ratio of images per ID to use for query (방식 1만 사용, default: 0.2)
        gallery_ratio: Ratio of images per ID to use for gallery (방식 1만 사용, default: 0.8)
        seed: Random seed for splitting (default: 42)
    """

    def __init__(
        self,
        root="",
        train_dir="",
        query_dir="",
        gallery_dir="",
        verbose=True,
        pid_begin=0,
        train_ratio=1.0,
        query_ratio=0.2,
        gallery_ratio=0.8,
        seed=42,
        camid_fixed=False,  # True면 모든 camid를 0으로 고정 (평가 시 제거 방지)
        **kwargs,
    ):
        super(CustomDataset, self).__init__()
        self.pid_begin = pid_begin
        self.train_ratio = train_ratio
        self.query_ratio = query_ratio
        self.gallery_ratio = gallery_ratio
        self.seed = seed
        self.camid_fixed = camid_fixed  # camid를 0으로 고정할지 여부

        # Set random seed for reproducibility
        random.seed(seed)

        # 방식 2: 별도 디렉토리 제공 시
        if train_dir and query_dir and gallery_dir:
            self.train_dir = train_dir
            self.query_dir = query_dir
            self.gallery_dir = gallery_dir
            self.use_separate_dirs = True

            self._check_before_run_separate()

            # 모든 디렉토리에서 ID 수집하여 일관된 PID 매핑 생성
            all_id_dirs = set()
            for dir_path in [self.train_dir, self.query_dir, self.gallery_dir]:
                if osp.exists(dir_path):
                    id_dirs = [
                        d
                        for d in os.listdir(dir_path)
                        if osp.isdir(osp.join(dir_path, d))
                    ]
                    all_id_dirs.update(id_dirs)
            self.all_id_dirs = sorted(list(all_id_dirs))
            self.id_to_pid = {
                id_dir: self.pid_begin + idx
                for idx, id_dir in enumerate(self.all_id_dirs)
            }

            # 각 디렉토리를 독립적으로 처리
            train = self._process_dir(self.train_dir, train=True, use_all=True)
            query = self._process_dir(
                self.query_dir, train=False, split="query", use_all=True
            )
            gallery = self._process_dir(
                self.gallery_dir, train=False, split="gallery", use_all=True
            )
        else:
            # 방식 1: 기존 방식 (root에서 자동 분리)
            self.dataset_dir = root
            self.use_separate_dirs = False

            self._check_before_run()

            # Process dataset
            train = self._process_dir(train=True)
            query = self._process_dir(train=False, split="query")
            gallery = self._process_dir(train=False, split="gallery")

        # Train 데이터셋의 PID를 0부터 연속적으로 재매핑
        # 이는 모델의 num_classes와 일치시키기 위해 필요합니다
        train_pids = set()
        for item in train:
            if len(item) == 5:
                _, pid, _, _, _ = item
            else:
                _, pid, _, _ = item
            train_pids.add(pid)

        # PID를 0부터 연속적으로 매핑
        pid2label = {pid: label for label, pid in enumerate(sorted(train_pids))}

        # Train 데이터셋의 PID 재매핑
        relabeled_train = []
        for item in train:
            if len(item) == 5:
                img_path, pid, camid, trackid, floor = item
                relabeled_pid = pid2label[pid]
                relabeled_train.append((img_path, relabeled_pid, camid, trackid, floor))
            else:
                img_path, pid, camid, trackid = item
                relabeled_pid = pid2label[pid]
                relabeled_train.append((img_path, relabeled_pid, camid, trackid))

        train = relabeled_train

        if verbose:
            print("=> CustomDataset loaded")
            # Train 데이터셋의 PID 범위 확인
            train_pids_check = set()
            for item in train:
                if len(item) == 5:
                    _, pid, _, _, _ = item
                else:
                    _, pid, _, _ = item
                train_pids_check.add(pid)
            if train_pids_check:
                min_pid = min(train_pids_check)
                max_pid = max(train_pids_check)
                print(
                    f"=> Train dataset PID range: {min_pid} to {max_pid} (total: {len(train_pids_check)} unique PIDs)"
                )
                if min_pid != 0 or max_pid != len(train_pids_check) - 1:
                    print(
                        f"   WARNING: PID range is not continuous from 0! Expected 0 to {len(train_pids_check)-1}, got {min_pid} to {max_pid}"
                    )
                else:
                    print(
                        f"   ✓ PID range is continuous from 0 to {len(train_pids_check)-1}"
                    )

            self.print_dataset_statistics(train, query, gallery)
            # Query와 Gallery 간 PID 불일치 확인
            self._check_query_gallery_pid_overlap(query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        (
            self.num_train_pids,
            self.num_train_imgs,
            self.num_train_cams,
            self.num_train_vids,
        ) = self.get_imagedata_info(self.train)
        (
            self.num_query_pids,
            self.num_query_imgs,
            self.num_query_cams,
            self.num_query_vids,
        ) = self.get_imagedata_info(self.query)
        (
            self.num_gallery_pids,
            self.num_gallery_imgs,
            self.num_gallery_cams,
            self.num_gallery_vids,
        ) = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if dataset directory exists (방식 1)"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")

    def _check_before_run_separate(self):
        """Check if separate directories exist (방식 2)"""
        if not osp.exists(self.train_dir):
            raise RuntimeError(
                f"Training directory '{self.train_dir}' is not available"
            )
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"Query directory '{self.query_dir}' is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(
                f"Gallery directory '{self.gallery_dir}' is not available"
            )

    def _process_dir(self, dataset_dir=None, train=True, split=None, use_all=False):
        """
        Process directory to create dataset list

        Args:
            dataset_dir: Directory to process (None이면 self.dataset_dir 사용)
            train: If True, process training set
            split: 'query' or 'gallery' for test set splitting
            use_all: If True, use all images (방식 2). If False, split by ratio (방식 1)
        """
        # Supported image extensions
        EXTs = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".JPG", ".JPEG", ".PNG")

        if dataset_dir is None:
            dataset_dir = self.dataset_dir

        dataset = []
        pid_container = set()
        camid_container = set()

        # Get all ID directories
        id_dirs = [
            d for d in os.listdir(dataset_dir) if osp.isdir(osp.join(dataset_dir, d))
        ]
        id_dirs = sorted(id_dirs)

        # Process each ID directory
        for id_dir in id_dirs:
            id_path = osp.join(dataset_dir, id_dir)

            # PID 매핑: 별도 디렉토리 방식이면 전체 ID 리스트에서 찾기
            if hasattr(self, "id_to_pid"):
                if id_dir not in self.id_to_pid:
                    continue  # 전체 ID 리스트에 없는 ID는 스킵
                pid = self.id_to_pid[id_dir]
            else:
                # 기존 방식: 인덱스 기반
                pid_idx = id_dirs.index(id_dir)
                pid = self.pid_begin + pid_idx

            pid_container.add(pid)

            # Get all images in this ID directory
            img_paths = []
            for ext in EXTs:
                img_paths.extend(glob.glob(osp.join(id_path, f"*{ext}")))
            img_paths = sorted(img_paths)

            if len(img_paths) == 0:
                print(f"Warning: No images found in {id_path}")
                continue

            # 방식 2: 모든 이미지 사용 (별도 디렉토리)
            if use_all:
                for img_idx, img_path in enumerate(img_paths):
                    # camid 할당: 고정 모드면 0, 아니면 이미지 인덱스 기반
                    if self.camid_fixed:
                        camid = 0
                    else:
                        camid = img_idx % 10
                    camid_container.add(camid)
                    # 층 정보 추출
                    filename = osp.basename(img_path)
                    floor = extract_floor_from_filename(filename)
                    dataset.append((img_path, pid, camid, 1, floor))
            else:
                # 방식 1: 비율에 따라 분리 (기존 방식)
                if train:
                    # For training, use train_ratio of images
                    num_train = max(1, int(len(img_paths) * self.train_ratio))
                    train_paths = random.sample(img_paths, num_train)
                    for img_idx, img_path in enumerate(train_paths):
                        # camid 할당: 고정 모드면 0, 아니면 이미지 인덱스 기반
                        if self.camid_fixed:
                            camid = 0
                        else:
                            camid = img_idx % 10
                        camid_container.add(camid)
                        # 층 정보 추출
                        filename = osp.basename(img_path)
                        floor = extract_floor_from_filename(filename)
                        # (img_path, pid, camid, trackid, floor) 형식으로 저장
                        dataset.append((img_path, pid, camid, 1, floor))
                else:
                    # For test set, split into query and gallery (non-overlapping)
                    # First, split images into query and gallery
                    num_query = max(1, int(len(img_paths) * self.query_ratio))
                    num_gallery = min(
                        len(img_paths) - num_query,
                        max(1, int(len(img_paths) * self.gallery_ratio)),
                    )

                    # Shuffle and split
                    shuffled_paths = img_paths.copy()
                    random.shuffle(shuffled_paths)

                    if split == "query":
                        query_paths = shuffled_paths[:num_query]
                        for img_idx, img_path in enumerate(query_paths):
                            # camid 할당: 고정 모드면 0, 아니면 이미지 인덱스 기반
                            if self.camid_fixed:
                                camid = 0
                            else:
                                camid = img_idx % 10
                            camid_container.add(camid)
                            # 층 정보 추출
                            filename = osp.basename(img_path)
                            floor = extract_floor_from_filename(filename)
                            dataset.append((img_path, pid, camid, 1, floor))
                    elif split == "gallery":
                        gallery_paths = shuffled_paths[
                            num_query : num_query + num_gallery
                        ]
                        for img_idx, img_path in enumerate(gallery_paths):
                            # camid 할당: 고정 모드면 0, 아니면 이미지 인덱스 기반
                            if self.camid_fixed:
                                camid = 0
                            else:
                                camid = img_idx % 10
                            camid_container.add(camid)
                            # 층 정보 추출
                            filename = osp.basename(img_path)
                            floor = extract_floor_from_filename(filename)
                            dataset.append((img_path, pid, camid, 1, floor))

        return dataset

    def _check_query_gallery_pid_overlap(self, query, gallery):
        """Query와 Gallery 간 PID 불일치를 확인하고 로그 출력"""
        import logging

        logger = logging.getLogger("transreid.check")

        # Query와 Gallery에서 PID 추출
        query_pids = set()
        query_pid_counts = {}
        query_pid_to_id_dir = {}  # PID -> 실제 ID 디렉토리 이름 매핑
        for item in query:
            if len(item) >= 2:
                pid = item[1]
                query_pids.add(pid)
                query_pid_counts[pid] = query_pid_counts.get(pid, 0) + 1
                # 실제 ID 디렉토리 이름 저장 (첫 번째 이미지만)
                if pid not in query_pid_to_id_dir:
                    img_path = item[0]
                    # 경로에서 ID 디렉토리 이름 추출
                    id_dir = osp.basename(osp.dirname(img_path))
                    query_pid_to_id_dir[pid] = id_dir

        gallery_pids = set()
        gallery_pid_counts = {}
        gallery_pid_to_id_dir = {}  # PID -> 실제 ID 디렉토리 이름 매핑
        for item in gallery:
            if len(item) >= 2:
                pid = item[1]
                gallery_pids.add(pid)
                gallery_pid_counts[pid] = gallery_pid_counts.get(pid, 0) + 1
                # 실제 ID 디렉토리 이름 저장 (첫 번째 이미지만)
                if pid not in gallery_pid_to_id_dir:
                    img_path = item[0]
                    # 경로에서 ID 디렉토리 이름 추출
                    id_dir = osp.basename(osp.dirname(img_path))
                    gallery_pid_to_id_dir[pid] = id_dir

        # 공통 PID
        common_pids = query_pids & gallery_pids
        # Query에만 있는 PID
        query_only_pids = query_pids - gallery_pids
        # Gallery에만 있는 PID
        gallery_only_pids = gallery_pids - query_pids

        # 로그 출력
        print("=" * 60)
        print("Query-Gallery PID Overlap Analysis")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("Query-Gallery PID Overlap Analysis")
        logger.info("=" * 60)
        logger.info(f"Total Query PIDs: {len(query_pids)}")
        logger.info(f"Total Gallery PIDs: {len(gallery_pids)}")
        logger.info(f"Common PIDs (appear in both): {len(common_pids)}")
        logger.info(f"Query-only PIDs (not in gallery): {len(query_only_pids)}")
        logger.info(f"Gallery-only PIDs (not in query): {len(gallery_only_pids)}")
        logger.info("")

        if len(common_pids) == 0:
            logger.warning(
                "⚠️  WARNING: No common PIDs between query and gallery! "
                "Evaluation will fail."
            )
        elif len(common_pids) < len(query_pids):
            logger.warning(
                f"⚠️  WARNING: Only {len(common_pids)}/{len(query_pids)} "
                f"query PIDs appear in gallery. "
                f"{len(query_only_pids)} query PIDs will be skipped during evaluation."
            )

        # Query에만 있는 PID 상세 정보 (최대 20개만 출력)
        if query_only_pids:
            query_only_list = sorted(list(query_only_pids))[:20]
            logger.info(f"Query-only PIDs (first 20): {query_only_list}")
            # 실제 ID 디렉토리 이름도 출력
            id_dirs = [
                query_pid_to_id_dir.get(pid, "unknown") for pid in query_only_list
            ]
            logger.info(f"  Corresponding ID directories: {id_dirs}")
            if len(query_only_pids) > 20:
                logger.info(f"... and {len(query_only_pids) - 20} more")

        # Gallery에만 있는 PID 상세 정보 (최대 20개만 출력)
        if gallery_only_pids:
            gallery_only_list = sorted(list(gallery_only_pids))[:20]
            logger.info(f"Gallery-only PIDs (first 20): {gallery_only_list}")
            # 실제 ID 디렉토리 이름도 출력
            id_dirs = [
                gallery_pid_to_id_dir.get(pid, "unknown") for pid in gallery_only_list
            ]
            logger.info(f"  Corresponding ID directories: {id_dirs}")
            if len(gallery_only_pids) > 20:
                logger.info(f"... and {len(gallery_only_pids) - 20} more")

        # 공통 PID의 실제 ID 디렉토리 이름 확인
        if common_pids:
            logger.info("")
            logger.info("Common PIDs ID Directory Mapping (first 10):")
            for pid in sorted(list(common_pids))[:10]:
                query_id = query_pid_to_id_dir.get(pid, "unknown")
                gallery_id = gallery_pid_to_id_dir.get(pid, "unknown")
                logger.info(
                    f"  PID {pid}: Query ID='{query_id}', Gallery ID='{gallery_id}'"
                )
                if query_id != gallery_id:
                    logger.warning(
                        f"    ⚠️  WARNING: PID {pid} maps to different ID directories! "
                        f"Query='{query_id}' vs Gallery='{gallery_id}'"
                    )

        # 공통 PID의 이미지 개수 통계
        if common_pids:
            common_query_counts = [
                query_pid_counts[pid] for pid in common_pids if pid in query_pid_counts
            ]
            common_gallery_counts = [
                gallery_pid_counts[pid]
                for pid in common_pids
                if pid in gallery_pid_counts
            ]
            logger.info("")
            logger.info("Common PIDs Image Count Statistics:")
            logger.info(
                f"  Query: min={min(common_query_counts)}, "
                f"max={max(common_query_counts)}, "
                f"avg={sum(common_query_counts)/len(common_query_counts):.2f}"
            )
            logger.info(
                f"  Gallery: min={min(common_gallery_counts)}, "
                f"max={max(common_gallery_counts)}, "
                f"avg={sum(common_gallery_counts)/len(common_gallery_counts):.2f}"
            )

        logger.info("=" * 60)
