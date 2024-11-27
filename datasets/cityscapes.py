import torchvision
import numpy as np
from PIL import Image
from typing import List, Any, Callable, Tuple
from collections import namedtuple

def get_cs_labeldata():
    cls_names = ['road', 'sidewalk', 'parking', 'rail track', 'building',
           'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
           'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
           'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'caravan', 'trailer', 'train',
           'motorcycle', 'bicycle']
    colormap = np.array([
            [128, 64, 128],
            [244, 35, 232],
            [250, 170, 160],
            [230, 150, 140],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [180, 165, 180],
            [150, 100, 100],
            [150, 120, 90],
            [153, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 0, 90],
            [0, 0, 110],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0],
            [220, 220, 220]])
    return cls_names, colormap

class CityscapesDataset(torchvision.datasets.Cityscapes):

    def __init__(self,
                 transforms: List[Callable],
                 *args: Any,
                 **kwargs: Any):

        super(CityscapesDataset, self).__init__(*args,
                                                **kwargs,
                                                target_type="semantic")
        self.transforms = transforms
        self.classes = ['road', 'sidewalk', 'parking', 'rail track', 'building',
           'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
           'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
           'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'caravan', 'trailer', 'train',
           'motorcycle', 'bicycle']
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        img_pth = self.images[index]
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, img_pth

def cityscapes(root: str,
               split: str,
               transforms: List[Callable]):
    return CityscapesDataset(root=root,
                             split=split,
                             transforms=transforms)

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                    'has_instances', 'ignore_in_eval', 'color'])

classes = ['road', 'sidewalk', 'parking', 'rail track', 'building',
           'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
           'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
           'terrain', 'sky', 'person', 'rider', 'car',
           'truck', 'bus', 'caravan', 'trailer', 'train',
           'motorcycle', 'bicycle']
