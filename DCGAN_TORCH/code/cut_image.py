# 将生成网络模型的生成图片进行处理和切割。

from PIL import Image
import sys

# 将图片填充为正方形
def fill_image(image):
    width, height = image.size
    # 选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    # 生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='blue')
    # 将之前的图粘贴在新图上，居中
    if width > height:  # 原图宽大于高，则填充图片的竖直维度
        # (x,y)二元组表示粘贴上图相对下图的起始位置
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))

    return new_image


def cut_image(image):
    width, height = image.size
    item_width = int(width / 8)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 8):  # 两重循环，生成若干张图片基于原图的位置
        for j in range(0, 8):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j*item_width, i*item_width, (j+1)*item_width, (i+1)*item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    return image_list

#保存
def save_images(image_list, filename):
    index = 1
    for image in image_list:
        image.save('/Users/hujq/Documents/code/nodejs/react/GANDiff/cut_pic/typeB/'+filename.split('.')[0]+str(index) + '.jpg', 'JPEG')
        index += 1

if __name__ == '__main__':
    filename="fake_samples_epoch_1860.png" # 需要进行分隔的文件名
    file_path = "/Users/hujq/Documents/code/nodejs/react/GANDiff/GAN_pic/typeB/"+filename  # 文件名绝对路径
    image = Image.open(file_path)
    #image.show()
    image = fill_image(image)
    image_list = cut_image(image)
    save_images(image_list, filename)

