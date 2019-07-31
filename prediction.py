import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from nw import deep_CNN

N_CLASSES = 5

img_dir = 'E:\\DeepLearn\\picture\\'
log_dir = 'E:\\DeepLearn\\Classify_picture\\self\\model\\train\\'
lists = ['CAMARY', 'CHR', 'HLD', 'LEVIN', 'YARIS']


# 从测试集中随机挑选一张图片，重置尺寸，转化成矩阵
def get_one_image(img_dir):
    imgs = os.listdir(img_dir)
    img_num = len(imgs)
    # print(imgs, img_num)
    idn = np.random.randint(0, img_num)
    image = imgs[idn]
    image_dir = img_dir + image
    print(image_dir)
    image = Image.open(image_dir)
    image = image.resize([160, 200])
    plt.imshow(image)
    plt.show()
    # np.array转换成多维数组
    image_arr = np.array(image)
    return image_arr


def test(image_arr):
    with tf.Graph().as_default():
        # 生成成指定的格式
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        # tf.reshape函数重塑张量
        image = tf.reshape(image, [1, 160, 200, 3])
        p = deep_CNN(image, 1, N_CLASSES)
        logits = tf.nn.softmax(p)
        # placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
        # 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        x = tf.placeholder(tf.float32)
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # 调用saver.restore()函数，加载训练好的网络模型
            print('Loading success')
        prediction = sess.run(logits, feed_dict={x: image_arr})
        max_index = np.argmax(prediction)
        print('预测的标签为：', max_index, lists[max_index])
        print('预测的结果为：', prediction)

if __name__ == '__main__':
    img = get_one_image(img_dir)
    test(img)
