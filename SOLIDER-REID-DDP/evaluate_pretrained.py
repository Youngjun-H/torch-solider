"""
Evaluate SOLIDER Pretrained Model on ReID Datasets

This script evaluates the publicly released SOLIDER pretrained weights
using the same validation logic as SOLIDER-REID-DDP training.

Usage:
    python evaluate_pretrained.py --config configs/eval_market1501.yaml
    python evaluate_pretrained.py --config configs/eval_msmt17.yaml
    python evaluate_pretrained.py --config configs/eval_cctv_reid.yaml
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add SOLIDER-REID to path for model imports
SOLIDER_REID_PATH = Path(__file__).parent.parent / "SOLIDER-REID"
sys.path.insert(0, str(SOLIDER_REID_PATH))

from model.backbones.swin_transformer import (
    swin_base_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_tiny_patch4_window7_224,
)


# ============================================================================
# Logging Setup
# ============================================================================
def setup_logger(name, save_dir=None, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "eval_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ============================================================================
# Dataset Classes
# ============================================================================
def read_image(img_path):
    """Keep reading image until succeed."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print(f"IOError when reading '{img_path}'. Retrying...")
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path


class BaseImageDataset:
    """Base class of image reid dataset"""

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids.append(pid)
            cams.append(camid)
            tracks.append(trackid)
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self, train, query, gallery, logger):
        num_train_pids, num_train_imgs, num_train_cams, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, _ = self.get_imagedata_info(gallery)

        logger.info("Dataset statistics:")
        logger.info("  ----------------------------------------")
        logger.info("  subset   | # ids | # images | # cameras")
        logger.info("  ----------------------------------------")
        logger.info(f"  train    | {num_train_pids:5d} | {num_train_imgs:8d} | {num_train_cams:9d}")
        logger.info(f"  query    | {num_query_pids:5d} | {num_query_imgs:8d} | {num_query_cams:9d}")
        logger.info(f"  gallery  | {num_gallery_pids:5d} | {num_gallery_imgs:8d} | {num_gallery_cams:9d}")
        logger.info("  ----------------------------------------")


class Market1501(BaseImageDataset):
    """Market1501 Dataset"""
    dataset_dir = 'market1501'

    def __init__(self, root='', verbose=True, logger=None, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose and logger:
            logger.info("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery, logger)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not os.path.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not os.path.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available")

    def _process_dir(self, dir_path, relabel=False):
        import glob
        import re

        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

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


class MSMT17(BaseImageDataset):
    """MSMT17 Dataset"""
    dataset_dir = 'MSMT17_V1'

    def __init__(self, root='', verbose=True, logger=None, **kwargs):
        super().__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.test_dir = os.path.join(self.dataset_dir, 'test')
        self.list_train_path = os.path.join(self.dataset_dir, 'list_train.txt')
        self.list_query_path = os.path.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = os.path.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()

        train = self._process_dir(self.train_dir, self.list_train_path, relabel=True)
        query = self._process_dir(self.test_dir, self.list_query_path, relabel=False)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path, relabel=False)

        if verbose and logger:
            logger.info("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery, logger)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")

    def _process_dir(self, dir_path, list_path, relabel=False):
        with open(list_path, 'r') as f:
            lines = f.readlines()

        pid_container = set()
        for line in lines:
            line = line.strip()
            pid = int(line.split(' ')[1])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for line in lines:
            line = line.strip()
            img_name, pid = line.split(' ')[:2]
            pid = int(pid)
            camid = int(img_name.split('_')[2]) - 1
            img_path = os.path.join(dir_path, img_name)
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 1))
        return dataset


class CCTVReID(BaseImageDataset):
    """
    CCTV ReID Dataset

    Dataset structure:
        cctv_reid_dataset_v2/
            train/
                person_id_1/
                    image1.jpg
                    ...
            valid/
                query/
                    ID1/
                        image1.jpg
                        ...
                gallery/
                    ID1/
                        image1.jpg
                        ...
    """
    dataset_dir = ''

    def __init__(self, root='', verbose=True, logger=None, **kwargs):
        super().__init__()
        self.dataset_dir = root
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.query_dir = os.path.join(self.dataset_dir, 'valid', 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'valid', 'gallery')

        self._check_before_run()

        train = self._process_train_dir(self.train_dir, relabel=True)
        query = self._process_valid_dir(self.query_dir, relabel=False)
        gallery = self._process_valid_dir(self.gallery_dir, relabel=False)

        if verbose and logger:
            logger.info("=> CCTV ReID loaded")
            self.print_dataset_statistics(train, query, gallery, logger)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, _ = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, _ = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not os.path.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not os.path.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available")

    def _extract_camera_id(self, filename):
        """Extract camera ID from filename"""
        import re
        # Pattern: date_CAMERA_time_trackid_frameid
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
        """Process training directory with person_id subfolders"""
        import glob

        person_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        person_dirs = sorted(person_dirs)

        pid2label = {pid: label for label, pid in enumerate(person_dirs)} if relabel else {pid: pid for pid in person_dirs}

        dataset = []
        for person_id in person_dirs:
            person_path = os.path.join(dir_path, person_id)
            img_paths = glob.glob(os.path.join(person_path, '*.jpg'))

            label = pid2label[person_id] if relabel else person_id

            for img_path in sorted(img_paths):
                filename = os.path.basename(img_path)
                camid = self._extract_camera_id(filename)
                dataset.append((img_path, label, camid, 1))

        return dataset

    def _process_valid_dir(self, dir_path, relabel=False):
        """Process validation directory (query or gallery) with ID subfolders"""
        import glob
        import re

        person_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        person_dirs = sorted(person_dirs)

        def extract_id(folder_name):
            match = re.search(r'ID(\d+)', folder_name)
            return int(match.group(1)) if match else 0

        person_id_nums = sorted([extract_id(pid) for pid in person_dirs])
        pid2label = {pid_num: label for label, pid_num in enumerate(person_id_nums)} if relabel else {pid_num: pid_num for pid_num in person_id_nums}

        dataset = []
        for person_dir in person_dirs:
            person_id_num = extract_id(person_dir)
            person_path = os.path.join(dir_path, person_dir)
            img_paths = glob.glob(os.path.join(person_path, '*.jpg'))

            label = pid2label[person_id_num] if relabel else person_id_num

            for img_path in sorted(img_paths):
                filename = os.path.basename(img_path)
                camid = self._extract_camera_id(filename)
                dataset.append((img_path, label, camid, 1))

        return dataset


DATASET_FACTORY = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'cctv_reid': CCTVReID,
}


# ============================================================================
# Model Definition
# ============================================================================
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class SOLIDERReIDModel(nn.Module):
    """
    SOLIDER ReID Model for evaluation

    Uses Swin Transformer backbone with BNNeck for feature extraction.
    This is equivalent to build_transformer in SOLIDER-REID.
    """

    def __init__(
        self,
        num_classes,
        backbone_type='swin_base_patch4_window7_224',
        pretrain_path='',
        semantic_weight=0.2,
        img_size=(384, 128),
        drop_path_rate=0.1,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        neck='bnneck',
        neck_feat='before',
        feat_norm='yes',
    ):
        super().__init__()

        self.neck = neck
        self.neck_feat = neck_feat
        self.feat_norm = feat_norm

        # Backbone factory
        backbone_factory = {
            'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
            'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
            'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
        }

        if backbone_type not in backbone_factory:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # Create backbone
        self.base = backbone_factory[backbone_type](
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            pretrained=pretrain_path,
            convert_weights=False,
            semantic_weight=semantic_weight,
        )

        # Load pretrained weights
        if pretrain_path:
            self.base.init_weights(pretrain_path)

        self.in_planes = self.base.num_features[-1]
        self.num_classes = num_classes

        # BN Neck
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # Classifier (for completeness, not used in evaluation)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat, featmaps = self.base(x)

        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat, featmaps
        else:
            if self.neck_feat == 'after':
                return feat, featmaps
            else:
                return global_feat, featmaps

    def load_param(self, trained_path):
        """Load trained model parameters"""
        param_dict = torch.load(trained_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        elif 'model' in param_dict:
            param_dict = param_dict['model']
        elif 'teacher' in param_dict:
            param_dict = param_dict['teacher']

        for key in param_dict:
            if 'classifier' in key:
                continue
            clean_key = key.replace('module.', '')
            if clean_key in self.state_dict():
                try:
                    self.state_dict()[clean_key].copy_(param_dict[key])
                except Exception as e:
                    print(f"Failed to load {clean_key}: {e}")

        print(f'Loaded pretrained model from {trained_path}')


# ============================================================================
# Evaluation Metrics (same as SOLIDER-REID-DDP)
# ============================================================================
def euclidean_distance(qf: torch.Tensor, gf: torch.Tensor) -> np.ndarray:
    """Compute Euclidean distance between query and gallery features"""
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return dist_mat.cpu().numpy()


def cosine_distance(qf: torch.Tensor, gf: torch.Tensor) -> np.ndarray:
    """Compute cosine distance between query and gallery features"""
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    dist_mat = 1 - torch.mm(qf, gf.t())
    return dist_mat.cpu().numpy()


def eval_func(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50
):
    """
    Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    feat_norm: bool = True,
    max_rank: int = 50,
    distance_metric: str = 'euclidean',
) -> dict:
    """
    Full evaluation pipeline
    """
    if feat_norm:
        query_features = F.normalize(query_features, p=2, dim=1)
        gallery_features = F.normalize(gallery_features, p=2, dim=1)

    if distance_metric == 'cosine':
        distmat = cosine_distance(query_features, gallery_features)
    else:
        distmat = euclidean_distance(query_features, gallery_features)

    cmc, mAP = eval_func(
        distmat, query_pids, gallery_pids,
        query_camids, gallery_camids, max_rank
    )

    return {
        'mAP': mAP,
        'rank1': cmc[0],
        'rank5': cmc[4] if len(cmc) > 4 else cmc[-1],
        'rank10': cmc[9] if len(cmc) > 9 else cmc[-1],
        'cmc': cmc
    }


# ============================================================================
# Main Evaluation Function
# ============================================================================
def create_dataloader(config, logger):
    """Create validation dataloader"""

    val_transforms = T.Compose([
        T.Resize(config['data']['img_size_test']),
        T.ToTensor(),
        T.Normalize(
            mean=config['data']['augmentation']['pixel_mean'],
            std=config['data']['augmentation']['pixel_std']
        )
    ])

    dataset_name = config['data']['dataset']
    root_dir = config['data']['root_dir']

    if dataset_name not in DATASET_FACTORY:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = DATASET_FACTORY[dataset_name](root=root_dir, logger=logger)

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set,
        batch_size=config['data'].get('batch_size_test', 256),
        shuffle=False,
        num_workers=config['data'].get('num_workers', 8),
        pin_memory=True,
    )

    return val_loader, len(dataset.query), dataset.num_train_pids, dataset


def extract_features(model, dataloader, device, logger):
    """Extract features from dataloader"""
    model.eval()

    features = []
    pids = []
    camids = []

    logger.info("Extracting features...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Feature extraction"):
            imgs, pid, camid, _, _ = batch
            imgs = imgs.to(device)

            feat, _ = model(imgs)

            features.append(feat.cpu())
            pids.extend(pid)
            camids.extend(camid)

    features = torch.cat(features, dim=0)
    pids = np.array(pids)
    camids = np.array(camids)

    return features, pids, camids


def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup output directory
    output_dir = config.get('output_dir', './eval_outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger("evaluate", output_dir)
    logger.info(f"Config: {config_path}")
    logger.info(f"Config content: {yaml.dump(config, default_flow_style=False)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create dataloader
    val_loader, num_query, num_classes, dataset = create_dataloader(config, logger)
    logger.info(f"Number of query images: {num_query}")
    logger.info(f"Number of gallery images: {len(val_loader.dataset) - num_query}")

    # Create model
    model_config = config['model']
    model = SOLIDERReIDModel(
        num_classes=num_classes,
        backbone_type=model_config['name'],
        pretrain_path=model_config['pretrain_path'],
        semantic_weight=model_config.get('semantic_weight', 0.2),
        img_size=config['data']['img_size_test'],
        drop_path_rate=model_config.get('drop_path_rate', 0.1),
        drop_rate=model_config.get('drop_rate', 0.0),
        attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
        neck=model_config.get('neck', 'bnneck'),
        neck_feat=model_config.get('neck_feat', 'before'),
        feat_norm=model_config.get('feat_norm', 'yes'),
    )

    # Load additional trained weights if specified
    if config.get('test_weight'):
        model.load_param(config['test_weight'])

    model = model.to(device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Extract features
    features, pids, camids = extract_features(model, val_loader, device, logger)
    logger.info(f"Feature shape: {features.shape}")

    # Split query and gallery
    query_features = features[:num_query]
    gallery_features = features[num_query:]
    query_pids = pids[:num_query]
    gallery_pids = pids[num_query:]
    query_camids = camids[:num_query]
    gallery_camids = camids[num_query:]

    # Evaluate
    feat_norm = model_config.get('feat_norm', 'yes') == 'yes'
    distance_metric = config.get('distance_metric', 'euclidean')

    logger.info(f"Feature normalization: {feat_norm}")
    logger.info(f"Distance metric: {distance_metric}")

    results = evaluate(
        query_features, gallery_features,
        query_pids, gallery_pids,
        query_camids, gallery_camids,
        feat_norm=feat_norm,
        distance_metric=distance_metric,
    )

    # Print results
    logger.info("=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    logger.info(f"mAP: {results['mAP']:.1%}")
    logger.info(f"Rank-1: {results['rank1']:.1%}")
    logger.info(f"Rank-5: {results['rank5']:.1%}")
    logger.info(f"Rank-10: {results['rank10']:.1%}")
    logger.info("=" * 50)

    # Save results
    results_path = os.path.join(output_dir, 'results.yaml')
    save_results = {
        'mAP': float(results['mAP']),
        'rank1': float(results['rank1']),
        'rank5': float(results['rank5']),
        'rank10': float(results['rank10']),
        'config': config_path,
        'dataset': config['data']['dataset'],
        'model': model_config['name'],
        'pretrain_path': model_config['pretrain_path'],
    }
    with open(results_path, 'w') as f:
        yaml.dump(save_results, f)
    logger.info(f"Results saved to {results_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate SOLIDER Pretrained Model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    main(args.config)
