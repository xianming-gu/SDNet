import os
import numpy as np
import torch
from model import SDNet as MyNet
import cv2
from time import time
from MyDataLoader import TestData
import torch.utils.data as data
import torchvision.transforms as transforms
import argparse
from MyOption import args

DEVICE = args.DEVICE
EPS = 1e-8


def Mytest():
    os.makedirs(args.img_save_dir, exist_ok=True)
    model_path = args.model_save_path + '/' + str(args.epoch) + '/'
    # model_path = './modelsave/PET/300/'

    net = MyNet()
    net.eval()
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(model_path + args.model_save_name))

    transform = transforms.Compose([transforms.ToTensor()])
    test_set = TestData(transform)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False,
                                  num_workers=0, pin_memory=False)
    with torch.no_grad():
        if args.img_type1 == 'CT/':
            for batch, [img_name, img1, img2] in enumerate(test_loader):  # CT-MRI Fusion
                print("test for image %s" % img_name[0])
                img1 = img1.to(DEVICE)
                img2 = img2.to(DEVICE)
                fused_img, _, _ = net(img1, img2)
                fused_img = (fused_img - fused_img.min()) / (fused_img.max() - fused_img.min()) * 255.
                fused_img = fused_img.cpu().numpy().squeeze()
                # fused_img = fused_img.astype(np.uint8)
                cv2.imwrite('%s/%s' % (args.img_save_dir, img_name[0]), fused_img)
        else:
            for batch, [img_name, img1_Y, img2, img1_CrCb] in enumerate(test_loader):  # PET/SPECT-MRI Fusion
                # print(img1_Y.shape)  # 1,1,256,256  # img1：PET_Y/SPECT_Y
                # print(img2.shape)  # 1,1,256,256  # img2：MRI
                # print(img1_CrCb.shape)  # 1,2,256,256  # img1_CrCb：PET/SPECT_CrCb

                print("test for image %s" % img_name[0])

                img1_Y = img1_Y.to(DEVICE)
                img2 = img2.to(DEVICE)

                fused_img_Y, _, _ = net(img1_Y, img2)

                fused_img_Y = (fused_img_Y - fused_img_Y.min()) / (fused_img_Y.max() - fused_img_Y.min()) * 255.
                fused_img_Y = fused_img_Y.cpu().numpy()

                fused_img = np.concatenate((fused_img_Y, img1_CrCb), axis=1).squeeze()
                fused_img = np.transpose(fused_img, (1, 2, 0))
                fused_img = fused_img.astype(np.uint8)
                fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)

                cv2.imwrite('%s/%s' % (args.img_save_dir, img_name[0]), fused_img)

    print('test results in ./%s/' % args.img_save_dir)
    print('Finish!')


if __name__ == '__main__':
    Mytest()
