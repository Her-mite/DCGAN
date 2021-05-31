# 生成对抗网络模型，训练模型生成仿真图片。
import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
from random import randint
import matplotlib.pyplot as plt

class NetG(nn.Module):
    def __init__(self, ngf, nz):
        super(NetG, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        # Sequential是一个时序容器modules会以他们传入时的顺序添加到容器中
        # ConvTranspose2d是二维卷积转置操作，将普通卷积的输出作为转置卷积的输入
        # BatchNorm2d是一个批标准化操作，对每一个小批量中计算各维度的均值和标准差
        # inplace=True代表的对源数据进行原地操作
        self.layer1 = nn.Sequential(
            # 参数分别有：输入通道数，输出通道数，卷积核大小，计算步长大小，补充0的层数，偏置
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        # layer2输出尺寸(ngf*4)x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # layer3输出尺寸(ngf*2)x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        # layer4输出尺寸(ngf)x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # layer5输出尺寸 3x96x96
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )

    # 定义NetG的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()
        # layer1 输入 3 x 96 x 96, 输出 (ndf) x 32 x 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2 输出 (ndf*2) x 16 x 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3 输出 (ndf*4) x 8 x 8
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4 输出 (ndf*8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 输出一个数(概率)
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # 定义NetD的前向传播
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.squeeze(1)
        out = out.squeeze(1)
        out = out.squeeze(1)

        # print(x)
        return out


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)  # 每次处理的图片数量
parser.add_argument('--imageSize', type=int, default=96)  # 图片像素大小
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')  # 图片信息维度
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=4000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--data_path', default='data/', help='folder to train data')
# parser.add_argument('--outf', default='img/', help='folder to output images and model checkpoints')
parser.add_argument('--data_path', default='augment/typeB/', help='folder to train data') # GAN训练集
parser.add_argument('--outf', default='GAN_pic/typeB/', help='folder to output images and model checkpoints') # GAN训练结果集
opt = parser.parse_args()

# 图像读入与预处理 包括统一大小、转化为张量、归一化操作
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

# 将指定路径的数据集加载到程序中
dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,  # 丢弃最后一批数据
)

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)

# 以以二分类的交叉熵作为损失函数，以Adam优化函数优化随机目标函数
criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# 将一个批次的数据传入label中
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

loss_D = []
loss_G = []

for epoch in range(1, opt.epoch + 1):
    for i, (imgs, _) in enumerate(dataloader):
        # 固定生成器G，训练鉴别器D，将梯度初始化为0
        optimizerD.zero_grad()
        # 让D尽可能的把真图片判别为1,计算D网络判别结果和实际结果的的交叉熵损失，
        imgs=imgs.to(device)
        output = netD(imgs)
        label.data.fill_(real_label)
        label=label.to(device)
        errD_real = criterion(output, label)
        errD_real.backward()
        # 让D尽可能把假图片判别为0
        label.data.fill_(fake_label)
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
        noise=noise.to(device)
        fake = netG(noise)  # 生成假图
        output = netD(fake.detach())  # 避免梯度传到G，因为不需要计算G的梯度。不加则反向传播会进行计算，节约时间和内存
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label = label.to(device)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        
        # 每两个epoch将值填入loss_D和l中H中，作为后期画图用数据
        if epoch % 2 == 0:
            loss_D.append(errD.item())
            loss_G.append(errG.item())

		# 将数据显示到控制台，直观看出当前两个网络的损失函数变化情况
        print("test")
        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))
        
    # 每三十个epoch保存当前网络G生成图片效果
    if epoch % 30 == 0:
        vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
        
	# 每500epoch保存当前网络G和网络D的神经模型作为后续使用
    if epoch % 500 == 0:
        torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))

# 将损失函数变化显示出来
# index = np.arange(100)
# y1 = loss_D
# y2 = loss_G
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(index, y1)
# ax1.set_xlabel('epoch')
# ax1.set_label('loss')
# ax1.set_ylabel('loss')
# ax1.set_xlim(0, 120)
# ax1.set_title('loss and correct')
# ax2 = ax1.twinx()
# ax2.set_label('correct')
# ax2.plot(index, y2, 'r')
# ax2.set_ylabel('correct')
# plt.show()