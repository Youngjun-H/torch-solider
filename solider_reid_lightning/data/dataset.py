"""Dataset classes for ReID."""

import glob
import os.path as osp
import re
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Read image from path."""
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    img = Image.open(img_path).convert("RGB")
    return img


class ReIDDataset(Dataset):
    """Base ReID dataset."""

    def __init__(self, data: List[Tuple[str, int, int, int]], transform=None):
        """
        Args:
            data: List of (img_path, pid, camid, viewid)
            transform: Image transform
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, pid, camid, viewid = self.data[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, viewid, img_path


class Market1501:
    """Market1501 dataset."""

    dataset_dir = "market1501"

    def __init__(self, root="", verbose=True):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self._print_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = len(set([x[1] for x in train]))
        self.num_train_cams = len(set([x[2] for x in train]))
        self.num_train_vids = len(set([x[3] for x in train]))

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available")

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        dataset = []

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 1))

        return dataset

    def _print_statistics(self, train, query, gallery):
        num_train_pids = len(set([x[1] for x in train]))
        num_train_imgs = len(train)
        num_query_pids = len(set([x[1] for x in query]))
        num_query_imgs = len(query)
        num_gallery_pids = len(set([x[1] for x in gallery]))
        num_gallery_imgs = len(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------------------")
        print(f"  train    | {num_train_pids:5d} | {num_train_imgs:8d}")
        print(f"  query    | {num_query_pids:5d} | {num_query_imgs:8d}")
        print(f"  gallery  | {num_gallery_pids:5d} | {num_gallery_imgs:8d}")
        print("  ----------------------------------------")


class MSMT17:
    """MSMT17 dataset."""

    dataset_dir = "MSMT17"

    def __init__(self, root="", verbose=True):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.test_dir = osp.join(self.dataset_dir, "test")
        self.list_train_path = osp.join(self.dataset_dir, "list_train.txt")
        self.list_val_path = osp.join(self.dataset_dir, "list_val.txt")
        self.list_query_path = osp.join(self.dataset_dir, "list_query.txt")
        self.list_gallery_path = osp.join(self.dataset_dir, "list_gallery.txt")

        self._check_before_run()

        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)

        if verbose:
            print("=> MSMT17 loaded")
            self._print_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = len(set([x[1] for x in train]))
        self.num_train_cams = len(set([x[2] for x in train]))
        self.num_train_vids = len(set([x[3] for x in train]))

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not osp.exists(self.test_dir):
            raise RuntimeError(f"'{self.test_dir}' is not available")

    def _process_dir(self, dir_path, list_path):
        with open(list_path, "r") as txt:
            lines = txt.readlines()

        dataset = []
        for img_info in lines:
            img_path, pid = img_info.split(" ")
            pid = int(pid)
            camid = int(img_path.split("_")[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid - 1, 1))

        return dataset

    def _print_statistics(self, train, query, gallery):
        num_train_pids = len(set([x[1] for x in train]))
        num_train_imgs = len(train)
        num_query_pids = len(set([x[1] for x in query]))
        num_query_imgs = len(query)
        num_gallery_pids = len(set([x[1] for x in gallery]))
        num_gallery_imgs = len(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------------------")
        print(f"  train    | {num_train_pids:5d} | {num_train_imgs:8d}")
        print(f"  query    | {num_query_pids:5d} | {num_query_imgs:8d}")
        print(f"  gallery  | {num_gallery_pids:5d} | {num_gallery_imgs:8d}")
        print("  ----------------------------------------")
