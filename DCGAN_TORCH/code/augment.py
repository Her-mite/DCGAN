# 对于所有图像， 进行裁切、旋转等方式增多数据集

import os
from PIL import Image
from torchvision import transforms
import random

# 增广数据集
def augment(image_path, imagename):
    # 获取随机数作为新图命名依据
    rand_num = str(random.random()).split('.')[1][9: ]

    # 获取图像实例
    image=Image.open(image_path)
  
    # 原图重设大小 
    resize_image=transforms.Resize(size=(256, 256))(image)
    resize_image.save(root+"augment/"+typeone +'/image/'+imagename.split('.')[0]+rand_num+'resize.jpg')
    
    # 图像翻转
    resize_image = transforms.RandomHorizontalFlip(p=1)(resize_image)   # p表示概率
    resize_image.save(root+"augment/"+typeone +'/image/'+imagename.split('.')[0]+rand_num+'horizon.jpg')
    resize_image = transforms.RandomVerticalFlip(p=1)(resize_image)
    resize_image.save(root+"augment/"+typeone +'/image/'+imagename.split('.')[0]+rand_num+'vertical.jpg')


    # 随机旋转图像角度
    for x in range(5):
        rotation_image= transforms.RandomRotation(degrees=(10, 80))(resize_image)
        rotation_image.save(root+"augment/"+typeone +'/image/'+imagename.split('.')[0]+rand_num+str(x)+'rotation.jpg')

    # 随机修改光学特性
    for x in range(5):
        color_image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(image)
        resize_color_image = transforms.Resize(size=(256,256))(color_image)
        resize_color_image.save(root+"augment/"+typeone +'/image/'+imagename.split('.')[0]+rand_num+str(x)+'color.jpg')

    print(imagename, "图像增广完成")


    # 随机裁切大小为256*256的图像
    # for x in range(5):
    #     random_image= transforms.RandomCrop(size=(256, 256))(image)
    #     random_image.save(root+"augment/"+typeone +'/image/'+imagename.split('.')[0]+rand_num+str(x)+'random.jpg')

    # 在图像的上下左右以及中心裁剪出尺寸为 size 的 5 张图片
    # five_images = transforms.FiveCrop(size=(256,256))(image)
    # i=0
    # for five_image in five_images:
    #     i += 1
    #     five_image.save(root+"augment/"+typeone +'/image/'+imagename.split('.')[0]+rand_num+str(i)+'five.jpg')

    




# 获取所有类别
root='/Users/hujq/Documents/code/nodejs/react/GANDiff/'
path=root+'stone_train/'  # 原始图像存储路径
if os.path.exists(path):
    types=os.listdir(path)

# 获取各种类别的所有图像并处理
for typeone in types:
    if os.path.exists(path+typeone) and typeone != '.DS_Store':
        images = os.listdir(path+typeone)
        print('当前处理类别：', typeone)
        for imagename in images: # 对每张图像进行增广处理
            if imagename != '.DS_Store':
                image_path = path + typeone +'/' +imagename # 拼接路径
                augment(image_path, imagename)

         



