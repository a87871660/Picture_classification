import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from nw import deep_CNN

N_CLASSES = 5

img_dir = 'E:\\DeepLearn\\picture\\'
log_dir = 'E:\\DeepLearn\\Classify_picture\\self\\model\\train\\'
lists = ['C', 'R', 'H', 'L', 'Y']

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
    image_arr = np.array(image)
    return image_arr


def test(image_arr):
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 160, 200, 3])
        p = deep_CNN(image, 1, N_CLASSES)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32)
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success')
        prediction = sess.run(logits, feed_dict={x: image_arr})
        max_index = np.argmax(prediction)
        print('The picture is ', max_index, lists[max_index])
        print('The probability is ', prediction)

if __name__ == '__main__':
    img = get_one_image(img_dir)
    test(img)
