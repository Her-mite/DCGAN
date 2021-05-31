# DCGAN 模型架构
1、生成器和判别器均不采用池化层，而采用（带步长的）的卷积层；其中判别器采用普通卷积（Conv2D），而生成器采用反卷积（DeConv2D）；

2、在生成器和判别器上均使用Batch Normalization；

3、在生成器除输出层外的所有层上使用RelU激活函数，而输出层使用Tanh激活函数；

4、在判别器的所有层上使用LeakyReLU激活函数；

5、卷积层之后不使用全连接层；

6、判别器的最后一个卷积层之后也不用Global Pooling，而是直接Flatten

# 安装pytorch 
https://pytorch.org/ 选择自己电脑对应版本的安装命令进行安装 mac  python pip安装：pip3 install torch torchvision torchaudio

# 关键代码
当前所有关键代码均在DCGAN_TORCH文件夹下，其他两个文件的神经网络是使用TensorFlow实现， 具体有地方可能没有调通。现在使用pytorch模块实现了全体方法

# 数据预处理
根据数据编号， 将数据分为20组， 分别命名为typeA、typeB、typeC....
存放目录： ./stone_train/typeA...
观察文件夹目录下图像， 对于沙砾占比较大的图像进行裁切
根据实际情况调整remove_sand.py参数，裁切图像，保留有效部分，删除冗余图像<br/>
```python remove_sand.py```

# 图像增强
参考博客: https://www.cnblogs.com/zhangxiann/p/13570884.htm
对图像进行，随机水平、垂直翻转， 随机角度旋转， 随机色度、亮度、饱和度、对比度的变化， 的操作进行图像增强
```python augment.py```<br/>
图像增强后的文件存放目录：./augment/typeA/image...

# 图像生成
trainGAN.py需注意以下两行参数，分别为GAN训练集存放目录和生成图像存放目录
parser.add_argument('--data_path', default='augment/typeB/', help='folder to train data') # GAN训练集
parser.add_argument('--outf', default='GAN_pic/typeB/', help='folder to output images and model checkpoints') # GAN训练结果集<br/> 
```python trainGAN.py```<br/>
生成图像耗时较久， 需要至少两千批次后的图像才能有较好效果

# 生成图像切割
在cut_image.py文件中修改文件所在路径、文件名和输出文件所在路径信息
<br/> ```python cut_image.py```
对上一步操作中， GAN生成图像效果较好的图像进行裁切，保留成像效果较好的图像， 作为后续CNN训练集和验证集

# CNN训练数据
将上一步裁切的图像，各种随机选择1/10图像存放到对应类型验证集，其余图像存入其训练集
训练集： cut_pic/CNN/train/typeA...
验证集： cut_pic/CNN/valid/typeB... 
其中“/Users/hujq/Documents/code/nodejs/react/GANDiff/cut_pic/CNN/train/”为训练集绝对路径
python trainCNN.py

# CNN验证模型正确率
python testCNN.py12
以下参数根据实际情况修改
parser.add_argument('--batchSize', type=int, default=29)  # 各类验证图像总数量
parser.add_argument('--path', type=str, default="/Users/hujq/Documents/code/nodejs/react/GANDiff/stone_train/") # 测试集路径








