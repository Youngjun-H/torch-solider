import os

import torch
from PIL import Image


class SingleFolderDataset(torch.utils.data.Dataset):
    """
    단일 폴더에서 이미지를 로드하는 Dataset 클래스.

    클래스별 서브디렉토리 구조가 없는 경우 사용합니다.
    DINO와 같은 self-supervised learning에서 사용됩니다.
    """

    def __init__(self, root, transform=None):
        """
        Args:
            root: 이미지 파일이 있는 디렉토리 경로
            transform: 이미지 변환 함수 (예: DataAugmentationDINO)
        """
        self.root = root
        self.transform = transform
        self.samples = []

        # 이미지 확장자
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
        )

        # 모든 이미지 파일 찾기
        if not os.path.isdir(root):
            raise ValueError(f"Data directory does not exist: {root}")

        for filename in sorted(os.listdir(root)):
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(IMG_EXTENSIONS):
                self.samples.append(
                    (filepath, 0)
                )  # label은 0으로 고정 (self-supervised이므로)

        if len(self.samples) == 0:
            raise ValueError(f"No image files found in {root}")

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 에러 발생 시 첫 번째 이미지 반환
            path, target = self.samples[0]
            img = Image.open(path).convert("RGB")

        if self.transform is not None:
            # DataAugmentationDINO는 리스트를 반환 (multi-crop)
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.samples)
