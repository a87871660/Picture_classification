import os
import numpy as np
import tensorflow as tf
from pre import get_file, get_batch, valuation_batch
from nw import deep_CNN, losses, trainning, evaluation

#ratio = 0.2
n_class = 5
image_h = 260
image_w = 220
target_height = 300
target_width = 260
batch_size = 16
capacity = 256
max_step = 200000
learning_rate = 0.001
train_dir = 'E:\\DeepLearn\\picture_train'
valuation_dir = 'E:\\DeepLearn\\picture_valuation'
logs_train_dir = 'E:\\DeepLearn\\Classify_picture\\self\\model\\train\\'
logs_valuation_dir = 'E:\\DeepLearn\\Classify_picture\\self\\model\\valuation\\'

train, train_label = get_file(train_dir)
valuation, valuation_label = get_file(valuation_dir)
train_batch, train_label_batch = get_batch(train, train_label, image_h, image_w, target_height, target_width, batch_size, capacity)
val_image_batch, val_label_batch = valuation_batch(valuation, valuation_label, image_h, image_w, target_height, target_width, batch_size, capacity)

is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
image_batch = tf.cond(is_training, lambda: train_batch, lambda: val_image_batch)
label_batch = tf.cond(is_training, lambda: train_label_batch, lambda: val_label_batch)

train_logits = deep_CNN(image_batch, batch_size, n_class)
train_loss = losses(train_logits, label_batch)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, label_batch)

#可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
summary_op = tf.summary.merge_all()
sess = tf.Session()
# 指定一个文件用来保存图
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
valuation_writer = tf.summary.FileWriter(logs_valuation_dir)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# 开启一个协调器
coord = tf.train.Coordinator()
# 使用start_queue_runners 启动队列填充
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    for step in np.arange(max_step):
        if coord.should_stop():
            #使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，会抛出一个 OutofRangeError 的异常，这时候就应该停止Sesson中的所有线程了;
            break

        _, summary_training, tra_loss, tra_acc= sess.run([train_op, summary_op, train_loss, train_acc], feed_dict={is_training: True})

        if step % 100 == 0:
            summary_valuation, val_loss, val_acc = sess.run([summary_op, train_loss, train_acc], feed_dict={is_training: False})
            print('Step %d, train loss = %.3f, valuation loss = %.3f, train accuracy = %.2f%%, valuation accuracy = %.2f%%, learning rate is %f' % (step, tra_loss, val_loss, tra_acc * 100.0, val_acc * 100.0, _[1]))
            #print('Step %d, train loss = %.3f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc))
            #print('The valuation loss = %.3f, valuation accuracy = %.2f%%' % (val_loss, val_acc))
            #print('The learning rate is %s' % (_[1]))
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            train_writer.add_summary(summary_training, step)
            valuation_writer.add_summary(summary_valuation, step)

        """
        if step % 100 == 0 or (step + 1) == max_step:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            train_writer.add_summary(summary_training, step)
            valuation_writer.add_summary(summary_valuation, step)
        """

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
coord.join(threads)
sess.close()



'''
if step % 2000 == 0 or (step + 1) == max_step:
    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
    # os.path.join连接两个或更多的路径名组件
    #saver.save(sess, checkpoint_path, global_step=step)
    summary.value.add(tag='evl_loss', simple_value=evl_loss)
    summary.value.add(tag='evl_acc', simple_value=evl_acc)
    saver.save(sess, checkpoint_path, global_step=step)
'''
