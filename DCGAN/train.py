import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from model import NetD, NetG

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)  # 每次处理的图片数量
parser.add_argument('--imageSize', type=int, default=96)  # 图片像素大小
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=30000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='data/', help='folder to train data')
parser.add_argument('--outf', default='imgs/', help='folder to output images and model checkpoints')
opt = parser.parse_args()
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像读入与预处理 包括统一大小、转化为张量、归一化操作
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Scale(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,  # 丢弃最后一批数据
)

netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

for epoch in range(1, opt.epoch + 1):
    for i, (imgs, _) in enumerate(dataloader):
        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()
        # 让D尽可能的把真图片判别为1
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

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))
    if epoch % 20 == 0:
        vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)

    if epoch % 500 == 0:
        torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))
