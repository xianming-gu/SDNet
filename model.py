import torch
import torch.nn as nn



class SDNet(nn.Module):
    def __init__(self):
        super(SDNet, self).__init__()
        self.conv11 = nn.Sequential((nn.Conv2d(1, 16, 5, 1, 2)), nn.LeakyReLU())
        self.conv12 = nn.Sequential((nn.Conv2d(1, 16, 5, 1, 2)), nn.LeakyReLU())

        self.conv21 = nn.Sequential((nn.Conv2d(16, 16, 3, 1, 1)), nn.LeakyReLU())
        self.conv22 = nn.Sequential((nn.Conv2d(16, 16, 3, 1, 1)), nn.LeakyReLU())

        self.conv31 = nn.Sequential((nn.Conv2d(32, 16, 3, 1, 1)), nn.LeakyReLU())
        self.conv32 = nn.Sequential((nn.Conv2d(32, 16, 3, 1, 1)), nn.LeakyReLU())

        self.conv41 = nn.Sequential((nn.Conv2d(48, 16, 3, 1, 1)), nn.LeakyReLU())
        self.conv42 = nn.Sequential((nn.Conv2d(48, 16, 3, 1, 1)), nn.LeakyReLU())

        self.fuse = nn.Sequential((nn.Conv2d(128, 1, 1, 1, 0)), nn.Tanh())
        self.decom = nn.Sequential((nn.Conv2d(1, 128, 1, 1, 0)), nn.LeakyReLU())

        self.conv51 = nn.Sequential((nn.Conv2d(128, 16, 3, 1, 1)), nn.LeakyReLU())
        self.conv52 = nn.Sequential((nn.Conv2d(128, 16, 3, 1, 1)), nn.LeakyReLU())

        self.conv61 = nn.Sequential((nn.Conv2d(16, 4, 3, 1, 1)), nn.LeakyReLU())
        self.conv62 = nn.Sequential((nn.Conv2d(16, 4, 3, 1, 1)), nn.LeakyReLU())

        self.conv71 = nn.Sequential((nn.Conv2d(4, 1, 3, 1, 1)), nn.Tanh())
        self.conv72 = nn.Sequential((nn.Conv2d(4, 1, 3, 1, 1)), nn.Tanh())

    def forward(self, x1, x2):
        x11 = self.conv11(x1)
        x12 = self.conv21(x11)
        x13 = self.conv31(torch.cat([x11, x12], dim=1))
        x14 = self.conv41(torch.cat([x11, x12, x13], dim=1))

        x21 = self.conv12(x2)
        x22 = self.conv22(x21)
        x23 = self.conv32(torch.cat([x21, x22], dim=1))
        x24 = self.conv42(torch.cat([x21, x22, x23], dim=1))

        x_fuse = self.fuse(torch.cat([x11, x12, x13, x14, x21, x22, x23, x24], dim=1))

        x1_de = self.conv71(self.conv61(self.conv51(self.decom(x_fuse))))
        x2_de = self.conv72(self.conv62(self.conv52(self.decom(x_fuse))))

        return x_fuse, x1_de, x2_de


if __name__ == '__main__':
    x1 = torch.randn(1, 1, 256, 256)
    x2 = torch.randn(1, 1, 256, 256)

    net = SDNet()
    x_fuse, x1_de, x2_de = net(x1, x2)
    print(x_fuse.shape)
    print(x1_de.shape)
    print(x2_de.shape)
