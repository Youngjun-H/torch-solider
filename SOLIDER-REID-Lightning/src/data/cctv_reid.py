# encoding: utf-8
"""
CCTV ReID Dataset for Lightning Training

Dataset Structure:
    cctv_reid_dataset_v2/
        train/
            person_id_1/
                image1.jpg
                image2.jpg
                ...
            person_id_2/
                ...
        valid/
            query/
                ID1/
                    image1.jpg
                    ...
                ID2/
                    ...
            gallery/
                ID1/
                    image1.jpg
                    ...
                ID2/
                    ...

Each image filename contains camera information in format:
    YYYY-MM-DD_CAMERA_TIME_TRACKID_FRAMEID.jpg
    Example: 2025-10-15_1F_07-30-00_1_id563_frame_304039.jpg
"""

import glob
import re
import os
import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset


class CCTVReID(BaseImageDataset):
    """
    CCTV ReID Dataset

    Dataset statistics (approximate):
    # identities: ~57 (train) + ~60 (valid)
    # train images: varies per identity
    # query images: varies per identity
    # gallery images: varies per identity
    """
    dataset_dir = 'cctv_reid_dataset_v2'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(CCTVReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'valid', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'valid', 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin

        # Process train, query, gallery
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
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available")

    def _extract_camera_id(self, filename):
        """
        Extract camera ID from filename

        Filename format: YYYY-MM-DD_CAMERA_TIME_TRACKID_FRAMEID.jpg
        Example: 2025-10-15_1F_07-30-00_1_id563_frame_304039.jpg

        Camera naming convention:
            - 1F, 2F, 3F, 4F (floor levels)
            - 2F-out, 3F-out, 4F-out (outdoor cameras)
        """
        # Extract camera name (e.g., "1F", "2F-out", "3F-out", etc.)
        # Pattern: date_CAMERA_time_trackid_frameid
        pattern = r'\d{4}-\d{2}-\d{2}_([^_]+)_\d{2}-\d{2}-\d{2}'
        match = re.search(pattern, filename)

        if match:
            camera_str = match.group(1)
            # Map camera strings to IDs
            # Create a consistent mapping for all possible camera names
            camera_map = {
                '1F': 0,
                '2F': 1,
                '2F-out': 2,
                '3F': 3,
                '3F-out': 4,
                '4F': 5,
                '4F-out': 6,
            }
            return camera_map.get(camera_str, 0)  # Default to 0 if unknown
        else:
            # Fallback: use simple hashing if pattern not matched
            return 0

    def _process_train_dir(self, dir_path, relabel=False):
        """
        Process training directory

        Directory structure:
            train/
                person_id_1/
                    image1.jpg
                    image2.jpg
                person_id_2/
                    ...
        """
        # Get all person ID folders
        person_dirs = [d for d in os.listdir(dir_path) if osp.isdir(osp.join(dir_path, d))]
        person_dirs = sorted(person_dirs)

        # Create person ID mapping
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(person_dirs)}
        else:
            pid2label = {pid: pid for pid in person_dirs}

        dataset = []
        for person_id in person_dirs:
            person_path = osp.join(dir_path, person_id)
            img_paths = glob.glob(osp.join(person_path, '*.jpg'))

            # Get label for this person
            if relabel:
                label = pid2label[person_id]
            else:
                label = person_id

            for img_path in sorted(img_paths):
                # Extract camera ID from filename
                filename = osp.basename(img_path)
                camid = self._extract_camera_id(filename)

                # Track ID is set to 1 (single track per person)
                trackid = 1

                # Add to dataset: (img_path, pid, camid, trackid)
                dataset.append((img_path, self.pid_begin + label, camid, trackid))

        return dataset

    def _process_valid_dir(self, dir_path, relabel=False):
        """
        Process validation directory (query or gallery)

        Directory structure:
            query/ or gallery/
                ID1/
                    image1.jpg
                    image2.jpg
                ID2/
                    ...
        """
        # Get all person ID folders
        person_dirs = [d for d in os.listdir(dir_path) if osp.isdir(osp.join(dir_path, d))]
        person_dirs = sorted(person_dirs)

        # Extract numeric IDs from folder names (e.g., "ID1" -> 1)
        def extract_id(folder_name):
            match = re.search(r'ID(\d+)', folder_name)
            if match:
                return int(match.group(1))
            return 0

        # Create person ID mapping
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

            # Get label for this person
            if relabel:
                label = pid2label[person_id_num]
            else:
                label = person_id_num

            for img_path in sorted(img_paths):
                # Extract camera ID from filename
                filename = osp.basename(img_path)
                camid = self._extract_camera_id(filename)

                # Track ID is set to 1 (single track per person)
                trackid = 1

                # Add to dataset: (img_path, pid, camid, trackid)
                dataset.append((img_path, self.pid_begin + label, camid, trackid))

        return dataset
