# C:\graduation_project\pre\img143(153,154).jpg
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2 as cv
import numpy as np
import tensorflow as tf
from CNN import input_data
from CNN import model
from collections import Counter
# from Graduition_projict import thread1 as th
# from Graduition_projict import golbal as gv
import codecs

image_size = 227
MODEL_SAVE_PATH = r"E:\pythontest\CNN\images"
MODEL_NAME = "model.ckpt-5000.meta"

def get_one_image(img_dir):
    image = cv.imread(img_dir)

    cv.imshow(img_dir, image)
    # h, w, c = image.shape
    # # print('h:%d  w:%d' % (h, w))
    # edge = max([h, w])
    # # print(edge)
    # new_img = np.zeros((edge, edge, 3), dtype=np.uint8)  # 矩形-》正方形
    # if h == w:
    #     new_img[0:h, 0:w] = image[0:h, 0:w]
    # else:
    #     if h > w:
    #         new_img[0:h, int((h - w) / 2):int(h - ((h - w) / 2))] = image[0:h, 0:w]
    #     else:
    #         if w > h:
    #             new_img[int((w - h) / 2):int(w - (w - h) / 2), 0:w] = image[0:h, 0:w]
    #
    # # cv.imshow("image", new_img)
    image = cv.resize(image, (227, 227), interpolation=cv.INTER_CUBIC)  # 输入变为227*227

    image_arr = np.array(image)
    return image_arr


def find_top_2(temp_list):
    counter_words = Counter(temp_list)
    result = []
    # most_counter = counter_words.most_common(1)
    # print(most_counter)
    most_counter = counter_words.most_common(2)
    # print(most_counter)
    if len(most_counter) == 1:
        t = most_counter[0]
        # print(t)
        result.append(most_counter[0][0])
        # print(result)
    if len(most_counter) == 2:
        result.append(most_counter[0][0])
        result.append(most_counter[1][0])
        # print(result)
    return result


def forecast(img_path):
    log_dir = r"E:\pythontest\CNN\images"

    image_arr = get_one_image(img_path)

    j = 0
    temp_list = []
    while j < 3:
        with tf.Graph().as_default():
            image = tf.cast(image_arr, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [1, image_size, image_size, 3])
            # print(image.shape)
            p = model.inference(image, 1, 5)
            logits = tf.nn.softmax(p)
            x = tf.placeholder(tf.float32, shape=[image_size, image_size, 3])
            saver = tf.train.Saver()
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # print('Loading success')
                else:
                    print('No checkpoint')
                prediction = sess.run(logits, feed_dict={x: image_arr})
                max_index = np.argmax(prediction)
                # print(max_index)
                temp_list.append(max_index)
        j += 1
    result = find_top_2(temp_list)
    print(result)


if __name__ == "__main__":
    forecast(r'E:\pythontest\CNN\images\train\beach\img25.jpg')
#     forecast(r'C:\Users\Administrator\Desktop\1\auto.jpg')
#     forecast(r'C:\Users\Administrator\Desktop\1\hand.jpg')
    cv.waitKey(0)
