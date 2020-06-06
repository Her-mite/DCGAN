import tensorflow as tf


def inference(images, batch_size, n_classes):
    with tf.variable_scope('conv1_lrn') as scope:
        weights = tf.get_variable('weights', shape=[11, 11, 3, 96], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[96], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 4, 4, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        norm1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('pooling1') as scope:
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling1')

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', shape=[5, 5, 96, 256], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling2')
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 256, 384], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[384], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 384, 384], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[384], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME', )
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name='conv4')
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weights', shape=[3, 3, 384, 256], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name='conv5')
    with tf.variable_scope('pooling6') as scope:
        pool6 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling6')
    with tf.variable_scope('locol7') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[dim, 4096], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        local7 = tf.nn.relu(tf.matmul(reshape, weights))
        local7 = tf.nn.dropout(local7, keep_prob=0.5)
    with tf.variable_scope('local8') as scope:
        weights = tf.get_variable('weights', shape=[4096, 4096], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[4096], dtype=tf.float32, initializer=tf.constant_initializer(0.))
        local8 = tf.nn.relu(tf.matmul(local7, weights))
        local8 = tf.nn.dropout(local8, keep_prob=0.5)
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear', shape=[4096, n_classes], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', shape=[n_classes], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softtmax_linear = tf.add(tf.matmul(local8, weights), biases, name='softmax_linear')
    return softtmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss')as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
