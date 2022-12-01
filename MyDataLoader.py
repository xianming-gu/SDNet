import os
import cv2
import argparse
import torch
import numpy as np
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from MyOption import args


class TrainData(data.Dataset):
    def __init__(self, transform=None):
        super(TrainData, self).__init__()
        self.dir_prefix = args.dir_train
        self.img1_dir = os.listdir(self.dir_prefix + args.img_type1)
        self.img2_dir = os.listdir(self.dir_prefix + args.img_type2)

        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        assert len(self.img1_dir) == len(self.img2_dir)
        return len(self.img1_dir)

    def __getitem__(self, index):
        if args.img_type1 == 'CT/':
            img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
        else:
            img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index])
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
            img1 = img1[:, :, 0:1]
            img1 = img1.squeeze()
        img2 = cv2.imread(self.dir_prefix + args.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)

        img1_p, img2_p = self.get_patch(img1, img2)
        if self.transform:
            img1_p = self.transform(img1_p)
            img2_p = self.transform(img2_p)

        return img1_p, img2_p  # 1,256,256

    def get_patch(self, img1, img2):
        h, w = img1.shape[:2]
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img1_p = img1[y:y + stride, x:x + stride]
        img2_p = img2[y:y + stride, x:x + stride]

        return img1_p, img2_p


class TestData(data.Dataset):
    def __init__(self, transform=None):
        super(TestData, self).__init__()
        self.transform = transform
        self.dir_prefix = args.dir_test

        self.img1_dir = os.listdir(self.dir_prefix + args.img_type1)
        self.img2_dir = os.listdir(self.dir_prefix + args.img_type2)

    def __getitem__(self, index):
        img_name = str(self.img1_dir[index])
        if args.img_type1 == 'CT/':
            img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dir_prefix + args.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img_name, img1, img2  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256
        else:
            img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index])
            # img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dir_prefix + args.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)  # CT/PET/SPECT 256,256,3

            img1_Y = img1[:, :, 0:1]
            img1_CrCb = img1[:, :, 1:3].transpose(2, 0, 1)

            if self.transform:
                img1_Y = self.transform(img1_Y)
                img2 = self.transform(img2)

            return img_name, img1_Y, img2, img1_CrCb  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256

    def __len__(self):
        assert len(self.img1_dir) == len(self.img2_dir)
        return len(self.img1_dir)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    TrainDataset = TrainData(transform=transform)
    train_loader = data.DataLoader(TrainDataset,
                                   batch_size=1,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=2,
                                   pin_memory=True)
    i = 0
    for idx, data in enumerate(train_loader):
        # print(len(data))
        img1, _ = data
        print(img1)
        print(img1.shape)
        i = i + 1
    print(i)  # 133

    # TestDataset = TestData(transform=transform)
    # test_loader = data.DataLoader(TestDataset,
    #                               batch_size=1,
    #                               shuffle=True,
    #                               drop_last=True,
    #                               num_workers=2,
    #                               pin_memory=True)
    # j = 0
    # for idx, [img_name, data] in enumerate(test_loader):
    #     # print(len(data))
    #     j = j + 1
    # print(j)  # 24
