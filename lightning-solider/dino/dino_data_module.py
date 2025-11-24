"""DINO DataModule."""
from base.base_data_module import BaseDINODataModule


class DINODataModule(BaseDINODataModule):
    """
    DINO 학습을 위한 PyTorch Lightning DataModule.
    
    BaseDINODataModule을 상속받아 DINO 특화 기능을 구현합니다.
    현재는 Base와 동일하지만, 향후 DINO 특화 기능이 필요할 경우 확장 가능합니다.
    """
    pass

