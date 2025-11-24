"""Base DataModule for DINO and SOLIDER."""
import sys
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader
from torchvision import datasets

# lightning-solider 루트 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.augmentation import DataAugmentationDINO
from shared.dataset import SingleFolderDataset


class BaseDINODataModule(L.LightningDataModule):
    """
    DINO와 SOLIDER의 공통 DataModule 기능.
    
    ImageFolder와 SingleFolderDataset을 자동으로 선택하여 데이터를 로드합니다.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_path = args.data_path
        self.dataset = None

    def setup(self, stage=None):
        """
        데이터셋을 준비합니다.

        Args:
            stage: 'fit', 'validate', 'test', 'predict' 중 하나
                  None인 경우 모든 단계에 대해 설정됩니다.
        """
        # Transform 정의
        transform = DataAugmentationDINO(
            (self.args.height, self.args.width),
            (self.args.crop_height, self.args.crop_width),
            self.args.global_crops_scale,
            self.args.local_crops_scale,
            self.args.local_crops_number,
        )

        # 데이터셋 로직: ImageFolder를 우선 시도하고 실패시 SingleFolderDataset 사용
        try:
            self.dataset = datasets.ImageFolder(self.data_path, transform=transform)
            print(
                f"Using ImageFolder: {len(self.dataset)} images from {len(self.dataset.classes)} classes."
            )
        except (FileNotFoundError, ValueError, OSError):
            # ImageFolder가 실패하면 SingleFolderDataset 사용 (단일 폴더 구조)
            self.dataset = SingleFolderDataset(self.data_path, transform=transform)
            print(
                f"Using SingleFolderDataset: {len(self.dataset)} images from single folder."
            )

    def train_dataloader(self):
        """
        학습용 DataLoader를 반환합니다.

        Lightning은 자동으로 DDP 환경에서 DistributedSampler를 설정하므로
        shuffle=True만 지정하면 됩니다.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,  # Lightning이 DDP 환경에서 자동으로 DistributedSampler 처리
        )

