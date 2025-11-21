import utils
from PIL import Image
from torchvision import transforms


class DataAugmentationDINO(object):
    """
    DINO 학습을 위한 데이터 증강 클래스.

    Multi-crop 전략을 사용하여:
    - 2개의 global crops (큰 크롭)
    - N개의 local crops (작은 크롭)
    을 생성합니다.
    """

    def __init__(
        self, size, crop_size, global_crops_scale, local_crops_scale, local_crops_number
    ):
        """
        Args:
            size: Global crop 크기 (height, width) 튜플
            crop_size: Local crop 크기 (height, width) 튜플
            global_crops_scale: Global crop의 scale 범위 (min, max) 튜플
            local_crops_scale: Local crop의 scale 범위 (min, max) 튜플
            local_crops_number: 생성할 local crop의 개수
        """
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # 이미지 크기에 따른 aspect ratio 설정
        if size == (224, 224):
            ratio = (0.75, 1.3333333333333333)
        elif size == (256, 128):
            ratio = (0.4, 0.6)
        elif size == (384, 128):
            ratio = (0.25, 0.4)
        else:
            # 기본값: 정사각형에 가까운 비율
            ratio = (0.75, 1.3333333333333333)

        print(
            f"DataAugmentationDINO: global_crops_scale={global_crops_scale}, size={size}, ratio={ratio}"
        )

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                    ratio=ratio,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=global_crops_scale,
                    interpolation=Image.BICUBIC,
                    ratio=ratio,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=crop_size,
                    scale=local_crops_scale,
                    interpolation=Image.BICUBIC,
                    ratio=ratio,
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        """
        이미지에 multi-crop 증강을 적용합니다.

        Args:
            image: PIL Image

        Returns:
            crops: 리스트 [global_crop1, global_crop2, local_crop1, ..., local_cropN]
        """
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
