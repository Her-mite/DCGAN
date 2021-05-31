import tensorflow as tf
import numpy as np
import os

import time

# -*- coding: utf-8 -*-
"""                          对初始图像及标签数据格式进行转换                          """
def get_files(file_dir, labelfile):
    image_list = []
    label_list = []
    len_file = len(os.listdir(file_dir))                        #统计知道哪个文件夹中图像文件的个数。
    len_label = len(open(labelfffile, "r").readlines())           #统计labelfile文件有多少行。如果两者不相等，说明
    if len_file == len_label:                                   #labelfile文件有误。
        print('num of image indentify to num of labels')
        print('the len of file is%d.' % len_file)
    txt_file = open(labelfile,"r")
    # print(txt_file)
    content = os.listdir(file_dir)
    # content.sort(key=lambda x: int(x[3:-4]))
    for file in content:#
        image_list.append(file_dir + "\\" + file)
        print(file_dir + "\\" + file)
        one_content = txt_file.readline()
        name = one_content.split(sep=' ')

        print(name[0]+"===="+file)
        if name[0] == file:
            name1 = name[1].split(sep='\r')
            label_list.append(int(name1[0]))
        else:
            print('file name is different from label name!\n')
    txt_file.close()
    print('There are %d images \n there are %d labels\n' % (len(image_list), len(label_list)))
    # temp = np.array([image_list, label_list])                   #把文件名序列和标签序列组成二维数组。
    # temp = temp.transpose()                                     #对二维数组转置
    # np.random.shuffle(temp)                                     #打乱数组各行顺序
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]                   #字符型标签转换哼整数型标签
    return image_list, label_list


"""                          对数据进行分块                          """


def get_batch(image_list, label_list, image_W, image_H, batch_size, capacity):
    image_cast = tf.cast(image_list, tf.string)                                 #将python.list类型转换成tf
    label_cast = tf.cast(label_list, tf.int32)                                  #辨别的格式
    input_queue = tf.train.slice_input_producer([image_cast, label_cast])       #生成队列
    labels = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    images = tf.image.decode_jpeg(image_contents, channels=3)
    images = tf.image.resize_image_with_crop_or_pad(images, image_W, image_H)
    images = tf.image.per_image_standardization(images)                         #标准化数据
    image_batch, label_batch = tf.train.batch([images, labels], batch_size=batch_size, num_threads=64,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

# get_files(r'E:\pythontest\CNN\images\train\main',r'E:\pythontest\CNN\images\train\table.txt')