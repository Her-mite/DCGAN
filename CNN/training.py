import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 5
IMG_W = 98
IMG_H = 98
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 30000            #37000*6 / 16
learning_rate = 0.0001


def run_training():
    train_dir = r'E:\pythontest\CNN\images\train\main'        # 训练集
    train_labelfile = r'E:\pythontest\CNN\images\train\table.txt'
    logs_train_dir = r'E:\pythontest\CNN\images'
    val_dir = r'E:\pythontest\CNN\images\test\main'             # 测试集
    val_labelfile = r'E:\pythontest\CNN\images\test\table.txt'
    logs_val_dir = r'E:\pythontest\CNN\images'

    train_list, trainlabel_list = input_data.get_files(train_dir, train_labelfile)
    val_list, vallable_list = input_data.get_files(val_dir, val_labelfile)
    train_batch, train_label_batch = input_data.get_batch(train_list, trainlabel_list, IMG_W, IMG_H, TRAIN_BATCH_SIZE,
                                                          CAPACITY)
    val_batch, val_label_batch = input_data.get_batch(val_list, vallable_list, IMG_W, IMG_H, VAL_BATCH_SIZE, CAPACITY)
    logits = model.inference(train_batch, TRAIN_BATCH_SIZE, N_CLASSES)

    loss = model.losses(logits, train_label_batch)
    train_op = model.trainning(loss, learning_rate)
    acc = model.evaluation(logits, train_label_batch)

    x_train = tf.placeholder(tf.float32, shape=[TRAIN_BATCH_SIZE, IMG_W, IMG_H, 3])
    y_train = tf.placeholder(tf.int16, shape=[TRAIN_BATCH_SIZE])

    x_val = tf.placeholder(tf.float32, shape=[VAL_BATCH_SIZE, IMG_W, IMG_H, 3])
    y_val = tf.placeholder(tf.int16, shape=[VAL_BATCH_SIZE])

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir, sess.graph)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                                feed_dict={x_train: tra_images, y_train: tra_labels})
                if step % 50 == 0:
                    print('step %d,train loss = %.2f,train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, acc], feed_dict={x_val: val_images, y_val: val_labels})
                    print('**  Step %d , val loss = %.2f, val accuracy = %.2f%%**' % (step, val_loss, val_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    val_writer.add_summary(summary_str, step)
                if step % 5000 == 0 or(step+1) ==MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                    saver.save(sess,checkpoint_path,global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    run_training()
