import os
import random
from os.path import join
import numpy as np
import torch.multiprocessing
from scipy.io import loadmat
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset

def get_pd_labeldata():
    cls_names = ['road', 'building', 'vegetation']
    colormap = np.array([
            [58, 0, 68], #[158, 0, 0],[58, 0, 68],
            [0, 130, 122], #[107, 130, 148], 
            [255, 230, 0], #[101, 192, 0],[0, 130, 122],
            [0, 0, 0]])
    return cls_names, colormap

class potsdam(Dataset):
    def __init__(self, transforms, split, root):
        super(potsdam, self).__init__()
        self.split = split
        self.root = root
        self.transform = transforms
        split_files = {
            "train": ["labelled_train.txt"],
            "unlabelled_train": ["unlabelled_train.txt"],
            # "train": ["unlabelled_train.txt"],
            "val": ["labelled_test.txt"],
            "train+val": ["labelled_train.txt", "labelled_test.txt"],
            "all": ["all.txt"]
        }
        assert self.split in split_files.keys()

        self.files = []
        for split_file in split_files[self.split]:
            with open(join(self.root, split_file), "r") as f:
                self.files.extend(fn.rstrip() for fn in f.readlines())

        self.coarse_labels = True
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id + ".mat"))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id + ".mat"))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        img, label = self.transform(img, label)

        if self.coarse_labels:
            new_label_map = torch.ones_like(label)*255
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        # mask = (label > 0).to(torch.float32)
        return img, label, image_id

    def __len__(self):
        return len(self.files)
    
classes = ['road', 'building', 'vegetation']


class PotsdamRaw(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(PotsdamRaw, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdamraw", "processed")
        self.transform = transform
        self.target_transform = target_transform
        self.files = []
        for im_num in range(38):
            for i_h in range(15):
                for i_w in range(15):
                    self.files.append("{}_{}_{}.mat".format(im_num, i_h, i_w))

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)