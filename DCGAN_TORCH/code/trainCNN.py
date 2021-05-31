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
from PIL import Image


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

# 分类类型标识
classes = ('typeA', 'typeB')

# 加载训练CNN网络的数据样本
def loadtraindata():
    path = "/Users/hujq/Documents/code/nodejs/react/GANDiff/cut_pic/CNN/train/"  # 路径
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((32, 32)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.CenterCrop(32),
                                                    transforms.ToTensor()])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)
    return trainloader



# 训练CNN网络
def trainandsave():
    trainloader = loadtraindata()
    # 神经网络结构
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    cor = []  # 定义正确率和损失函数随epoch增加的变化
    los = []
    max = 0
    # 训练部分
    for epoch in range(200):  # 训练的数据量为200个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成50张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss和correct进行输出
        correct = 0.0
        for i, data in enumerate(trainloader, 0):
            # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)  # 转换数据格式用Variable

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进CNN网络net

            # 正确率
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.data).sum().item()
            # print('corret', i, correct)

            loss = criterion(outputs, labels)   # 计算损失值
            loss.backward()  # loss反向传播 该图会动态生成并自动微分，会自动计算图中参数的导数；
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.item()  # loss累加

			# 限制训练次数不能超过最大值
            if i>max:
                max = i

            # 假设总数为sum 
            if i % (sum/4) == sum/4-1:
                print('[%d, %5d] loss: %.3f  acc: %.1f%%' %
                      (epoch + 1, i + 1, running_loss / (sum/4), (sum*correct)/100))  # 然后再除以200，就得到这两百次的平均损失值

            if i % 25 == 24:
                print('[%d, %5d] loss: %.3f  acc: %.1f%%' %
                      (epoch + 1, i + 1, running_loss / 25, (100*correct)/100))  # 然后再除以200，就得到这两百次的平均损失值
                los.append(running_loss/50)
                cor.append((50*correct)/100)
                running_loss = 0.0  # 这一个50次结束后，就把running_loss归零，下一个200次继续使用
                correct = 0.0
        print(max)
        
    # 展示训练效果变化图片
    index = np.arange(200)
    y1 = los
    y2 = cor
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(index, y1)
    ax1.set_xlabel('epoch')
    ax1.set_label('loss')
    ax1.set_ylabel('loss')
    ax1.set_xlim(0, 200)
    ax1.set_title('loss and correct')
    ax2 = ax1.twinx()
    ax2.set_label('correct')
    ax2.plot(index, y2, 'r')
    ax2.set_ylabel('correct')

    plt.show()
    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'net.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参
    # trainandsave()

# # 加载上次训练结果模型，方便在上次基础上继续训练
# def reload_net():
#     trainednet = torch.load('net.pkl')
#     return trainednet


# 使用CNN网络进行分类
if __name__ == "__main__":
    trainandsave()
