import os
import numpy as np
import tensorflow as tf
from pre import get_file, get_batch, valuation_batch
from nw import deep_CNN, losses, trainning, evaluation

ratio = 0.2
num_epochs = 1
n_class = 5
image_h = 320
image_w = 240
target_height = 400
target_width = 320
batch_size = 32
capacity = 256
max_step = 200000
learning_rate = 0.0001
train_dir = 'E:\\DeepLearn\\picture'
logs_train_dir = 'E:\\DeepLearn\\Classify_picture\\self\\model\\'

train, train_label, valuation, valuation_label = get_file(train_dir, ratio)
train_batch, train_label_batch = get_batch(train, train_label, image_h, image_w, target_height, target_width, batch_size, capacity)
val_image_batch, val_label_batch = valuation_batch(valuation, valuation_label, image_h, image_w, target_height, target_width, batch_size, capacity)

is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
x = tf.cond(is_training, lambda: train_batch, lambda: val_image_batch)
y = tf.cond(is_training, lambda: train_label_batch, lambda: val_label_batch)

train_logits = deep_CNN(x, batch_size, n_class)
#lish_train_logits = deep_CNN(val_image_batch, val_label_batch, n_class)
train_loss = losses(train_logits, y)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, y)

summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    for step in np.arange(max_step):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc= sess.run([train_op, train_loss, train_acc], feed_dict={is_training: True})

        if step % 100 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            print('The learning rate is %s' %(_[1]))
            val_loss, val_acc = sess.run([train_loss, train_acc], feed_dict={is_training: False})
            print('The valuation loss = %.2f, valuation accuracy = %.2f%%' % (val_loss, val_acc * 100.0))
            print('-----------------------------------------------------')

        if step % 200 == 0 or (step + 1) == max_step:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            summary_str = sess.run(summary_op, feed_dict={is_training: True})
            train_writer.add_summary(summary_str, step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
coord.join(threads)
sess.close()
