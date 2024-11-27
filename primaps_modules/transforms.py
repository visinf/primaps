import torch, random
import torchvision.transforms.functional as F
import torchvision.transforms as tf
import numpy as np
from PIL import Image
from typing import Tuple, List, Callable


class Compose:

    def __init__(self,
                 transforms: List[Callable], 
                 student_augs: bool = False):
        self.transforms = transforms
        self.student_augs = student_augs

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image, 
                 pseudo = None) -> Tuple[torch.Tensor, torch.Tensor]:

        for transform in self.transforms:
            if pseudo is None:
                img, gt = transform(img, gt)
            else:
                img, gt, pseudo = transform(img, gt, pseudo)
                
        if self.student_augs:
            aimg = img.clone()
            aimg, _ = RandGaussianBlur()(aimg, gt)
            if 0.5 > random.random():
                aimg, _ = ColorJitter()(aimg, gt)
            else:
                aimg, _ = MaskGrayscale()(aimg, gt)
                

        if pseudo is None and not self.student_augs: 
            return img, gt
        elif pseudo is None and self.student_augs:
            return img, gt, aimg
        elif pseudo is not None and not self.student_augs:
            return img, gt, pseudo
        else:
            return img, gt, aimg, pseudo

class ToTensor:

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image,
                 pseudo = None) -> Tuple[torch.Tensor, torch.Tensor]:

        img = F.to_tensor(np.array(img))
        gt = torch.from_numpy(np.array(gt)).unsqueeze(0)
        if pseudo is not None:
            pseudo = torch.from_numpy(np.array(pseudo)).unsqueeze(0)

        if pseudo is None: 
            return img, gt
        else:
            return img, gt, pseudo

class Resize:

    def __init__(self,
                 resize: Tuple[int]):

        self.img_resize = tf.Resize(size=resize,
                                    interpolation=tf.InterpolationMode.BILINEAR) 
        self.gt_resize = tf.Resize(size=resize,
                                   interpolation=tf.InterpolationMode.NEAREST) 

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image,
                 pseudo = None) -> Tuple[Image.Image, Image.Image]:

        img = self.img_resize(img)
        gt = self.gt_resize(gt)

        if pseudo is None: 
            return img, gt
        else:
            return img, gt, self.gt_resize(pseudo)

class ImgResize:

    def __init__(self,
                 resize: Tuple[int, int]):
        self.resize = resize
        self.num_pixels = self.resize[0]*self.resize[1]

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.prod(torch.tensor(img.shape[-2:])) > self.num_pixels:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear').squeeze(0)
        return img, gt

class ImgResizePIL:

    def __init__(self,
                 resize: Tuple[int]):
        self.resize = resize
        self.num_pixels = self.resize[0]*self.resize[1]

    def __call__(self,
                 img: Image) -> Image:
        if img.height*img.width > self.num_pixels:
            img = img.resize((self.resize[1], self.resize[0]), tf.InterpolationMode.BILINEAR)
        return img

class Normalize:

    def __init__(self,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):

        self.norm = tf.Normalize(mean=mean,
                                 std=std)

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor,
                 pseudo = None) -> Tuple[torch.Tensor, torch.Tensor]:

        img = self.norm(img)

        if pseudo is None: 
            return img, gt
        else:
            return img, gt, pseudo
    
class UnNormalize(object):
    def __init__(self,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2



class RandomHFlip:

    def __init__(self,
                 percentage: float = 0.5):

        self.percentage = percentage

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image,
                 pseudo = None) -> Tuple[Image.Image, Image.Image]:

        if random.random() < self.percentage:
            img = F.hflip(img)
            gt = F.hflip(gt)
            if pseudo is not None:
                pseudo = F.hflip(pseudo)

        if pseudo is None: 
            return img, gt
        else:
            return img, gt, pseudo


class RandomResizedCrop:

    def __init__(self,
                 crop_size: List[int],
                 crop_scale: List[float],
                 crop_ratio: List[float]):
        print('RandomResizedCrop ratio modified!!!')
        self.crop_scale = tuple(crop_scale)
        self.crop_ratio = tuple(crop_ratio)
        self.crop = tf.RandomResizedCrop(size=tuple(crop_size),
                                         scale=self.crop_scale,
                                         ratio=self.crop_ratio,)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image,
                 pseudo = None) -> Tuple[Image.Image, Image.Image]:

        i, j, h, w = self.crop.get_params(img=img,
                                          scale=self.crop.scale,
                                          ratio=self.crop.ratio)
        img = F.resized_crop(img, i, j, h, w, self.crop.size, tf.InterpolationMode.BILINEAR) 
        gt = F.resized_crop(gt, i, j, h, w, self.crop.size, tf.InterpolationMode.NEAREST) 
        if pseudo is not None:
            pseudo = F.resized_crop(pseudo, i, j, h, w, self.crop.size, tf.InterpolationMode.NEAREST)

        if pseudo is None: 
            return img, gt
        else:
            return img, gt, pseudo

class CenterCrop:

    def __init__(self,
                 crop_size: int):

        self.crop = tf.CenterCrop(size=crop_size)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image,
                 pseudo = None) -> Tuple[Image.Image, Image.Image]:

        img = self.crop(img)
        gt = self.crop(gt)

        if pseudo is None: 
            return img, gt
        else:
            return img, gt, self.crop(pseudo)
    
class PyramidCenterCrop:

    def __init__(self, 
                 crop_size: List[int],
                 scales: List[float]):

        self.crop_size = crop_size
        self.scales = scales
        self.crop = tf.CenterCrop(size=crop_size)


    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:
        
        imgs = []
        gts = []
        for s in self.scales:
            new_size = (int(self.crop_size*1/s), int(self.crop_size*1/s*(img.shape[2]/img.shape[1])))
            img = tf.Resize(size=new_size, interpolation=tf.InterpolationMode.BILINEAR)(img)
            gt = tf.Resize(size=new_size, interpolation=tf.InterpolationMode.NEAREST)(gt)
            imgs.append(self.crop(img))
            gts.append(self.crop(gt))

        return torch.stack(imgs), torch.stack(gts)






class IdsToTrainIds:

    def __init__(self,
                 source: str):

        self.source = source
        self.first_nonvoid = 7


    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.source == 'cityscapes':
            gt = gt.to(dtype=torch.int64) - self.first_nonvoid
            gt[gt>26] = 255
            gt[gt<0] = 255
        elif self.source == 'cocostuff':
            gt = gt.to(dtype=torch.int64)
        elif self.source == 'potsdam':
            gt = gt.to(dtype=torch.int64)
        return img, gt


class ColorJitter:
    def __init__(self, percentage: float = 0.3, brightness: float = 0.1,
                 contrast: float = 0.1, saturation: float = 0.1, hue: float = 0.1):

        self.percentage = percentage
        self.jitter = tf.ColorJitter(brightness=brightness,
                                     contrast=contrast,
                                     saturation=saturation,
                                     hue=hue)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image,
                 pseudo = None) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.percentage:
            img = self.jitter(img)
        
        if pseudo is None: 
            return img, gt
        else:
            return img, gt, pseudo

class MaskGrayscale:

    def __init__(self, percentage: float = 0.1):
        self.percentage = percentage

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image,
                 pseudo = None) -> Tuple[Image.Image, Image.Image]:
        if self.percentage > random.random():
            img = tf.Grayscale(num_output_channels=3)(img) 
        if pseudo is None: 
            return img, gt
        else:
            return img, gt, pseudo

class RandGaussianBlur:

    def __init__(self, radius: List[float] = [.1, 2.]):
        self.radius = radius

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image,
                 pseudo = None) -> Tuple[Image.Image, Image.Image]:

        radius = random.uniform(self.radius[0], self.radius[1])
        img = tf.GaussianBlur(kernel_size=21, sigma=radius)(img)

        if pseudo is None: 
            return img, gt
        else:
            return img, gt, pseudo
