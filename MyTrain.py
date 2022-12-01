import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import torch.utils.data as data
import torchvision.transforms as transforms
# from loss import grad_loss, ints_loss
from loss import loss_total
from MyDataLoader import TrainData
from MyOption import args

from model import SDNet as net


# warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # 设置随机数种子
    setup_seed(args.seed)
    model_path = args.model_save_path + '/' + str(args.epoch) + '/'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(args.img_save_dir, exist_ok=True)
    # os.makedirs('./modelsave')

    lr = args.lr

    # device handling
    if args.DEVICE == 'cpu':
        device = 'cpu'
    else:
        device = args.DEVICE
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # prepare model folder
    os.makedirs(args.temp_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = TrainData(transform=transform)
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=2,
                                   pin_memory=True)
    # model = net(img_size=args.imgsize, dim=256)
    model = net()

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    loss_plt = []
    for epoch in range(0, args.epoch):
        # os.makedirs(args.result + '/' + '%d' % (epoch + 1), exist_ok=True)

        loss_mean = []
        for idx, datas in enumerate(train_loader):
            # print(len(data))
            img1, img2 = datas
            # 训练模型
            model, img_fusion, loss_per_img = train(model, img1, img2, lr, device)
            loss_mean.append(loss_per_img)

        # print loss
        sum_list = 0
        for item in loss_mean:
            sum_list += item
        sum_per_epoch = sum_list / len(loss_mean)
        print('Epoch--%d\tLoss:%.5f' % (epoch + 1, sum_per_epoch))
        loss_plt.append(sum_per_epoch.detach().cpu().numpy())

        # save info to txt file
        strain_path = args.temp_dir + '/temp_loss.txt'
        Loss_file = 'Epoch--' + str(epoch + 1) + '\t' + 'Loss:' + str(sum_per_epoch.detach().cpu().numpy())
        with open(strain_path, 'a') as f:
            f.write(Loss_file + '\r\n')

    # 保存模型
    torch.save(model.state_dict(), model_path + args.model_save_name)
    print('model save in %s' % model_path)

    # 输出损失函数曲线
    plt.figure()
    x = range(0, args.epoch)  # x和y的维度要一样
    y = loss_plt
    plt.plot(x, y, 'r-')  # 设置输出样式
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(model_path + '/loss.png')  # 保存训练损失曲线图片
    plt.show()  # 显示曲线


def train(model, img1, img2, lr, device):
    model.to(device)
    model.train()

    img1 = img1.to(device)
    img2 = img2.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr)

    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    img_fusion, x1_de, x2_de = model(img1, img2)
    # img_fusion = img_fusion.cpu()
    img_fusion = img_fusion.to(device)
    x1_de = x1_de.to(device)
    x2_de = x2_de.to(device)

    # loss_total = compute_loss(img_fusion, img_cat, img_s, img_f)
    loss = loss_total(img_fusion, x1_de, x2_de, img1, img2)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return model, img_fusion, loss


if __name__ == '__main__':
    main()
