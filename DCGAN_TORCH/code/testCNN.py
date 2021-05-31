# 以卷积神经网络来对图像进行模型训练和分类工作
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os.path
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=32)  # 图像像素大小
parser.add_argument('--batchSize', type=int, default=29)  # 验证图像总数量
parser.add_argument('--path', type=str, default="/Users/hujq/Documents/code/nodejs/react/GANDiff/stone_train/")# 验证数据集
# parser.add_argument('--path', type=str, default="/Users/hujq/Documents/code/nodejs/react/GANDiff/stone_train/")# 原图测试数据集

opt = parser.parse_args()

# 加载分类测试用数据
def loadtestdata():
    # 测试数据集所在路径
    path=opt.path
    testset = torchvision.datasets.ImageFolder(path,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.imageSize, opt.imageSize)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                   transforms.ToTensor()])
                                               )
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False)
    return testloader


class Net(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10个输出

    def forward(self, x):  # 前向传播

        x = self.pool(F.relu(self.conv1(x)))  # F就是torch.nn.functional
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        # 从卷基层到全连接层的维度转换

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载上次训练结果模型，方便在上次基础上继续训练
def reload_net():
    trainednet = torch.load('net.pkl')
    return trainednet

def test():

    # 分类类型标识
    classes = ('typeA', 'typeB')

    testloader = loadtestdata()
    net = reload_net()  # 使用模型
    dataiter = iter(testloader)
    images, labels = dataiter.next()  #
    print('\nGroundTruth:'
          , " ".join('%5s' % classes[labels[j]] for j in range(29)))  # 打印前25个GT（test集里图片的标签）
    outputs = net(Variable(images))

    _, predicted = torch.max(outputs.data, 1)
    print('Predicted:  ', " ".join('%5s' % classes[predicted[j]] for j in range(opt.batchSize)))

    cc = 0  # 识别正确图片数量
    for j in range(opt.batchSize):
        if classes[labels[j]] == classes[predicted[j]]:
            cc += 1
    print(cc)

    index = np.arange(opt.batchSize+1) 
    real = [] 
    predi = []
    real.append('xx')
    predi.append('xx')
    for j in range(opt.batchSize):
        real.append(classes[labels[j]])
        predi.append(classes[predicted[j]])
    width = 0.3
    plt.bar(x=index, height=np.array(real), width=width, color='yellow', label=u'real')
    plt.bar(x=index + width, height=np.array(predi), width=width, color='green', label=u'predict')
    plt.xlabel('epoch')
    plt.ylabel('real/predict')
    plt.legend(loc='best')
    plt.show()
            # 打印前25个预测值
            # imshow(torchvision.utils.make_grid(images, nrow=5))  # nrow是每行显示的图片数量，缺省值为8

            # test()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()
            print('total: %d' % total)
            print('Accuracy of the network on the 10000 test images: %d %%' % (
                            100 * correct / total))

if __name__ == "__main__":
    test()                        