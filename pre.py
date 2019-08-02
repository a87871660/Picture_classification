import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import *

C = []
label_C = []
R = []
label_R = []
H = []
label_H = []
L = []
label_L = []
Y = []
label_Y = []

def get_file (file_dir):
    for file_1 in os.listdir(file_dir + '\\C'):
        C.append(file_dir + '\\C' + '\\' + file_1)
        label_C.append(0)
    for file_2 in os.listdir(file_dir + '\\R'):
        R.append(file_dir + '\\R' + '\\' +  file_2)
        label_R.append(1)
    for file_3 in os.listdir(file_dir + '\\H'):
        H.append(file_dir + '\\H' + '\\' + file_3)
        label_H.append(2)
    for file_4 in os.listdir(file_dir + '\\L'):
        L.append(file_dir + '\\L' + '\\' +  file_4)
        label_L.append(3)
    for file_5 in os.listdir(file_dir + '\\Y'):
        Y.append(file_dir + '\\Y' + '\\' +  file_5)
        label_Y.append(4)
    print('There are %d C\n There are %d R\n There are %d H\n There are %d L\n There are %d Y' %(len(C), len(R), len(H), len(L), len(Y)))

    image_list = np.hstack((C, R, H, L, Y))
    label_list = np.hstack((label_C, label_R, label_H, label_L, label_Y))

    temp = np.array([image_list, label_list])

    # translate [file_dir, label], [file_dir, label]...
    temp = temp.transpose()

    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    """
    n_sample = len(label_list)
    n_val = int(math.ceil(n_sample * ratio))
    n_train = n_sample - n_val
    tra_iamges = image_list[0:n_train]
    tra_labels = label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = image_list[n_train:-1]
    val_labels = label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    """

    return image_list, label_list

def get_batch(image, label, image_h, image_w, target_height, target_width, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label], num_epochs=None, shuffle=False)
    label = input_queue[1]
    # Tensor("input_producer/GatherV2_1:0", shape=(), dtype=int32)
    image_contants = tf.read_file(input_queue[0])
    # Tensor("ReadFile:0", shape=(), dtype=string)
    image = tf.image.decode_jpeg(image_contants, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)#!!
    image = tf.image.resize_images(image, [image_h, image_w])
    image = tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
    #image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)
    #image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=capacity - 1)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)#??
    return image_batch, label_batch

def valuation_batch(val_iamge, val_label, image_h, image_w, target_height, target_width, batch_size, capacity):
    val_iamge = tf.cast(val_iamge, tf.string)
    val_label = tf.cast(val_label, tf.int32)
    input_queue = tf.train.slice_input_producer([val_iamge, val_label], num_epochs=None, shuffle=False)
    val_label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    val_iamge = tf.image.decode_jpeg(image_contents, channels=3)
    val_iamge = tf.image.resize_images(val_iamge, [image_h, image_w])
    val_iamge = tf.image.resize_image_with_crop_or_pad(val_iamge, target_height, target_width)#!!
    image_batch, label_batch = tf.train.batch([val_iamge, val_label], batch_size=batch_size, num_threads=16, capacity=capacity)
    val_label_batch = tf.reshape(label_batch, [batch_size])
    val_image_batch = tf.cast(image_batch, tf.float32)
    return val_image_batch, val_label_batch
