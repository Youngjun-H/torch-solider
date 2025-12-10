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

from .bases import BaseImageDataset


class CustomDataset(BaseImageDataset):
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

    Args:
        root: Root directory containing ID subdirectories
        verbose: Whether to print dataset statistics
        pid_begin: Starting PID offset (default: 0)
        train_ratio: Ratio of images per ID to use for training (default: 1.0, use all)
        query_ratio: Ratio of images per ID to use for query (default: 0.2)
        gallery_ratio: Ratio of images per ID to use for gallery (default: 0.8)
        seed: Random seed for splitting (default: 42)
    """

    def __init__(
        self,
        root="",
        verbose=True,
        pid_begin=0,
        train_ratio=1.0,
        query_ratio=0.2,
        gallery_ratio=0.8,
        seed=42,
        **kwargs,
    ):
        super(CustomDataset, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = root
        self.train_ratio = train_ratio
        self.query_ratio = query_ratio
        self.gallery_ratio = gallery_ratio
        self.seed = seed

        # Set random seed for reproducibility
        random.seed(seed)

        self._check_before_run()

        # Process dataset
        train = self._process_dir(train=True)
        query = self._process_dir(train=False, split="query")
        gallery = self._process_dir(train=False, split="gallery")

        if verbose:
            print("=> CustomDataset loaded")
            self.print_dataset_statistics(train, query, gallery)

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
        """Check if dataset directory exists"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")

    def _process_dir(self, train=True, split=None):
        """
        Process directory to create dataset list

        Args:
            train: If True, process training set
            split: 'query' or 'gallery' for test set splitting
        """
        # Supported image extensions
        EXTs = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".JPG", ".JPEG", ".PNG")

        dataset = []
        pid_container = set()
        camid_container = set()

        # Get all ID directories
        id_dirs = [
            d
            for d in os.listdir(self.dataset_dir)
            if osp.isdir(osp.join(self.dataset_dir, d))
        ]
        id_dirs = sorted(id_dirs)

        # Process each ID directory
        for pid_idx, id_dir in enumerate(id_dirs):
            id_path = osp.join(self.dataset_dir, id_dir)
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

            # Split images for train/query/gallery
            if train:
                # For training, use train_ratio of images
                num_train = max(1, int(len(img_paths) * self.train_ratio))
                train_paths = random.sample(img_paths, num_train)
                for img_path in train_paths:
                    # Use directory name as camera ID (or assign sequentially)
                    camid = pid_idx % 10  # Simple camera assignment
                    camid_container.add(camid)
                    dataset.append((img_path, pid, camid, 1))
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
                    for img_path in query_paths:
                        camid = pid_idx % 10
                        camid_container.add(camid)
                        dataset.append((img_path, pid, camid, 1))
                elif split == "gallery":
                    gallery_paths = shuffled_paths[num_query : num_query + num_gallery]
                    for img_path in gallery_paths:
                        camid = pid_idx % 10
                        camid_container.add(camid)
                        dataset.append((img_path, pid, camid, 1))

        return dataset
