from torchvision import transforms
import torch
import os
import pandas as pd
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Dataset
import random
from pathlib import Path
from utils.variables import *
# from torchtoolbox.transform import Cutout
# from monai import transforms

image_size = 224
padding_size = 0
crop_size = 224
train_high_compose = transforms.Compose([
    # Mycrop(),
    transforms.Resize((image_size + padding_size, image_size + padding_size)),
    transforms.RandomCrop(crop_size),
    # MyMask(),
    # transforms.CenterCrop((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomInvert(p=0.5),
    # transforms.RandomAffine(8, translate=(.15, .15)),
    # transforms.RandomAffine(degrees=0, scale=(0.8,1), fillcolor=(0,0,0)),
    # transforms.ColorJitter(brightness=(0.5, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
])

train_low_compose = transforms.Compose([
    # Mycrop(),
    transforms.Resize((image_size + padding_size, image_size + padding_size)),
    transforms.RandomCrop(crop_size),
    # MyMask(),
    # transforms.CenterCrop((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(90),
    transforms.RandomInvert(p=0.5),
    # transforms.ColorJitter(brightness=(0.5, 0.9)),
    # transforms.RandomAffine(8, translate=(.15, .15)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
])


test_compose = transforms.Compose([
    # Mycrop(),
    # Brightness_reduce(),
    transforms.Resize((image_size, image_size)),
    # MyMask(),
    # transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CancerSeT_CSV(Dataset):

    def __init__(self, root, type_):
        self.root = Path(root)
        self.type_ = type_
        if type_ == "train":
            self.csv = self.root / "ISIC_2019_train.csv"   #
            self.transform_high = train_high_compose
            self.transform_low = train_low_compose
        elif type_ == "val":
            self.csv = self.root / "ISIC_2019_val.csv"  # BM_test.csv"
            self.transform_high = test_compose
            self.transform_low = test_compose
        elif type_ == "test":
            self.csv = self.root / "ISIC_2019_test.csv"  # BM_test.csv"
            self.transform_high = test_compose
            self.transform_low = test_compose
        self.check_files(self.csv)
        try:
            self.csv = pd.read_csv(self.csv)
        except:
            self.csv = pd.read_csv(self.csv, encoding='gbk')
        self.csv = self.csv.dropna()
        self.csv['image'] = self.csv['image'].astype(str)


        print("loading good_bad")
        self.people_classfiy = self.csv.loc[:, 'Cancer_1'].map(
            lambda x: 0 if x == 'good' else (1 if x == 'bad' else 2))

        self.people_classfiy.index = self.csv['image']
        self.people_classfiy = self.people_classfiy.to_dict()

        self.pic_0 = []
        self.pic_1 = []
        self.pic_files = []
        for p in self.people_classfiy:
            if type_ == 'train':
                pic_file = self.root / str(p+".jpg")  # person
                pic_file = str(pic_file)

            elif type_ == 'val':
                pic_file = self.root / str(p + ".jpg") # person
                pic_file = str(pic_file)


            elif type_ == 'test':
                pic_file = self.root / str(p + ".jpg") # person
                pic_file = str(pic_file)


            self.pic_files = []
            if self.people_classfiy[p] == 0:
                self.pic_0.append(pic_file)
            elif self.people_classfiy[p] == 1:
                self.pic_1.append(pic_file)

        print(len(self.pic_0), len(self.pic_1))
        self.pic_files = self.pic_0 + self.pic_1
        if type_ == 'train':
            random.shuffle(self.pic_0)
            random.shuffle(self.pic_1)


        # if type_ == 'train':
        #     self.files = []
        #     if len(self.pic_0) >= len(self.pic_1):
        #         ratio_1 = int(len(self.pic_0) // len(self.pic_1))
        #         distance = len(self.pic_0) - (ratio_1 * len(self.pic_1))
        #         self.pic_1 = (ratio_1) * self.pic_1 + self.pic_1[0: distance]
        #     else:
        #         ratio_0 = int(len(self.pic_1) // len(self.pic_0))
        #         distance = len(self.pic_1) - (ratio_0 * len(self.pic_0))
        #         self.pic_0 = (ratio_0) * self.pic_0 + self.pic_0[0: distance]
        #
        #     self.pic_files = self.pic_0 + self.pic_1
        #     print("After copying", len(self.pic_0), len(self.pic_1))
        random.shuffle(self.pic_files)





    def check_files(self, file):
        print("files:", file)
        assert Path(file).exists(), FileExistsError('{}不存在'.format(str(file)))

    def __len__(self):
        return len(self.pic_files)

    def __getitem__(self, index):
        img_single = Image.open(self.pic_files[index])
        img_single = img_single.convert("RGB")
        people = str(self.pic_files[index].split('/')[-1])
        people = os.path.splitext(people)[0]
        cancer = self.csv.loc[self.csv['image'] == str(people), "Cancer_1"].iloc[0]
        id = str(people)
        y = self.people_classfiy[str(people)]
        if y == 0:
            img_data = self.transform_low(img_single)
        else:
            img_data = self.transform_high(img_single)
        if img_data.shape[0] == 1:
            img_data = torch.cat([img_data] * 3, 0)
        rs = {
            "img": img_data,
            "labels": torch.Tensor([y])[0],
            "id": id,
            "cancer": cancer,
            "image_path": str(self.pic_files[index])
        }
        return rs


