#!/usr/bin/env python
"""
Test script for CCTV ReID dataset loading

This script tests:
1. Dataset class initialization
2. Train/query/gallery data loading
3. Dataset statistics
4. Sample image loading
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'data'))

# Import only the dataset class (avoid importing datamodule to skip dependencies)
import glob
import re
import os
import os.path as osp
from collections import defaultdict

# Import base classes directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bases",
    Path(__file__).parent.parent / 'src' / 'data' / 'bases.py'
)
bases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bases)
BaseImageDataset = bases.BaseImageDataset


class CCTVReID(BaseImageDataset):
    """CCTV ReID Dataset - copied for testing"""
    dataset_dir = 'cctv_reid_dataset_v2'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(CCTVReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'valid', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'valid', 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        train = self._process_train_dir(self.train_dir, relabel=True)
        query = self._process_valid_dir(self.query_dir, relabel=False)
        gallery = self._process_valid_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CCTVReID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available")

    def _extract_camera_id(self, filename):
        pattern = r'\d{4}-\d{2}-\d{2}_([^_]+)_\d{2}-\d{2}-\d{2}'
        match = re.search(pattern, filename)
        if match:
            camera_str = match.group(1)
            camera_map = {
                '1F': 0, '2F': 1, '2F-out': 2,
                '3F': 3, '3F-out': 4, '4F': 5, '4F-out': 6,
            }
            return camera_map.get(camera_str, 0)
        return 0

    def _process_train_dir(self, dir_path, relabel=False):
        person_dirs = [d for d in os.listdir(dir_path) if osp.isdir(osp.join(dir_path, d))]
        person_dirs = sorted(person_dirs)

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(person_dirs)}
        else:
            pid2label = {pid: pid for pid in person_dirs}

        dataset = []
        for person_id in person_dirs:
            person_path = osp.join(dir_path, person_id)
            img_paths = glob.glob(osp.join(person_path, '*.jpg'))

            if relabel:
                label = pid2label[person_id]
            else:
                label = person_id

            for img_path in sorted(img_paths):
                filename = osp.basename(img_path)
                camid = self._extract_camera_id(filename)
                trackid = 1
                dataset.append((img_path, self.pid_begin + label, camid, trackid))

        return dataset

    def _process_valid_dir(self, dir_path, relabel=False):
        person_dirs = [d for d in os.listdir(dir_path) if osp.isdir(osp.join(dir_path, d))]
        person_dirs = sorted(person_dirs)

        def extract_id(folder_name):
            match = re.search(r'ID(\d+)', folder_name)
            if match:
                return int(match.group(1))
            return 0

        person_id_nums = sorted([extract_id(pid) for pid in person_dirs])
        if relabel:
            pid2label = {pid_num: label for label, pid_num in enumerate(person_id_nums)}
        else:
            pid2label = {pid_num: pid_num for pid_num in person_id_nums}

        dataset = []
        for person_dir in person_dirs:
            person_id_num = extract_id(person_dir)
            person_path = osp.join(dir_path, person_dir)
            img_paths = glob.glob(osp.join(person_path, '*.jpg'))

            if relabel:
                label = pid2label[person_id_num]
            else:
                label = person_id_num

            for img_path in sorted(img_paths):
                filename = osp.basename(img_path)
                camid = self._extract_camera_id(filename)
                trackid = 1
                dataset.append((img_path, self.pid_begin + label, camid, trackid))

        return dataset


def test_cctv_dataset():
    """Test CCTV ReID dataset loading"""

    print("=" * 80)
    print("Testing CCTV ReID Dataset")
    print("=" * 80)

    # Initialize dataset
    dataset_root = "/purestorage/AILAB/AI_2/datasets/PersonReID"

    try:
        print(f"\nLoading dataset from: {dataset_root}")
        dataset = CCTVReID(root=dataset_root, verbose=True)

        print("\n" + "=" * 80)
        print("Dataset Information:")
        print("=" * 80)
        print(f"Number of training identities: {dataset.num_train_pids}")
        print(f"Number of training images: {dataset.num_train_imgs}")
        print(f"Number of training cameras: {dataset.num_train_cams}")
        print(f"Number of training tracklets: {dataset.num_train_vids}")
        print()
        print(f"Number of query identities: {dataset.num_query_pids}")
        print(f"Number of query images: {dataset.num_query_imgs}")
        print(f"Number of query cameras: {dataset.num_query_cams}")
        print()
        print(f"Number of gallery identities: {dataset.num_gallery_pids}")
        print(f"Number of gallery images: {dataset.num_gallery_imgs}")
        print(f"Number of gallery cameras: {dataset.num_gallery_cams}")

        # Show sample data
        print("\n" + "=" * 80)
        print("Sample Training Data (first 5):")
        print("=" * 80)
        for i, (img_path, pid, camid, trackid) in enumerate(dataset.train[:5]):
            print(f"{i+1}. PID: {pid:3d} | CamID: {camid} | Track: {trackid} | Path: {Path(img_path).name}")

        print("\n" + "=" * 80)
        print("Sample Query Data (first 5):")
        print("=" * 80)
        for i, (img_path, pid, camid, trackid) in enumerate(dataset.query[:5]):
            print(f"{i+1}. PID: {pid:3d} | CamID: {camid} | Track: {trackid} | Path: {Path(img_path).name}")

        print("\n" + "=" * 80)
        print("Sample Gallery Data (first 5):")
        print("=" * 80)
        for i, (img_path, pid, camid, trackid) in enumerate(dataset.gallery[:5]):
            print(f"{i+1}. PID: {pid:3d} | CamID: {camid} | Track: {trackid} | Path: {Path(img_path).name}")

        # Camera distribution
        print("\n" + "=" * 80)
        print("Camera Distribution (Training):")
        print("=" * 80)
        from collections import Counter
        train_cams = [camid for _, _, camid, _ in dataset.train]
        cam_counts = Counter(train_cams)
        for camid, count in sorted(cam_counts.items()):
            print(f"Camera {camid}: {count} images")

        print("\n" + "=" * 80)
        print("✅ Dataset loaded successfully!")
        print("=" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Error loading dataset!")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_cctv_dataset()
    sys.exit(0 if success else 1)
