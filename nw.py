import tensorflow as tf

def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def deep_CNN(im_x, batch_size, n_classes):
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 256], 0.1), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([256]), name='biases', dtype=tf.float32)
        h_conv1 = tf.nn.relu(conv2d(im_x, w_conv1) + b_conv1, name='conv1')
        tf.summary.image('image', im_x, batch_size)

    with tf.variable_scope('pooling1_lrn') as scope:
        pool = max_pool_2x2(h_conv1, 'pooling1')
        norm1 = tf.nn.lrn(pool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 256, 128], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2) + b_conv2, name='conv2')

    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pool_2x2(h_conv2, 'pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    with tf.variable_scope('conv_3') as scope:
        w_conv3 = tf.Variable(weight_variable([3, 3, 128, 64], 0.1), name='weights', dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)
        h_conv3 = tf.nn.relu(conv2d(norm2, w_conv3) + b_conv3, name='conv3')

    with tf.variable_scope('pooling3_lrn') as scpoe:
        pool3 = max_pool_2x2(h_conv3, name='pooling3')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 64], 0.005), name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name='local3')

    h_fc1_dropout = tf.nn.dropout(h_fc1, 0.4, name='dropout')

    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([64, 64], 0.005), name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2, name='local4')

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([64, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2, weights), biases, name='softmax_linear')
    return softmax_linear

"""
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.4, name='dropout')

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([256, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')
    return softmax_linear
"""

def losses(logits, lb_y):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lb_y, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar('loss', loss)
    return loss

def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=5000, decay_rate=0.9,
                                                   staircase=True, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        tf.summary.scalar('learning_rate', learning_rate)
    return train_op, learning_rate

def evaluation(logits, lb_y):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, lb_y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy
