import os
from PIL import Image
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    def __init__(self, 
                 root, 
                 transforms,
                 student_augs, 
                 ):
        super(PrecomputedDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.student_augs = student_augs

        self.image_files = []
        self.label_files = []
        self.pseudo_files = []
        for file in os.listdir(os.path.join(self.root, 'imgs')):
            self.image_files.append(os.path.join(self.root, 'imgs', file))
            self.label_files.append(os.path.join(self.root, 'gts', file))
            self.pseudo_files.append(os.path.join(self.root, 'pseudos', file))


    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        pseudo_path = self.pseudo_files[index]

        img = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        pseudo = Image.open(pseudo_path)

        if self.student_augs:
            img, label, aimg, pseudo = self.transforms(img, label, pseudo)
            return img, label.long(), aimg, pseudo.long()
        else:
            img, label, pseudo = self.transforms(img, label, pseudo)
            return img, label.long(), pseudo.long()

    def __len__(self):
        return len(self.image_files)