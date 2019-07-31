import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import *

CAMARY = []
label_CAMARY = []
CHR = []
label_CHR = []
HLD = []
label_HLD = []
LEVIN = []
label_LEVIN = []
YARIS = []
label_YARIS = []

def get_file (file_dir):
    for file_1 in os.listdir(file_dir + '\\CAMARY'):
        CAMARY.append(file_dir + '\\CAMARY' + '\\' + file_1)
        label_CAMARY.append(0)
    for file_2 in os.listdir(file_dir + '\\CHR'):
        CHR.append(file_dir + '\\CHR' + '\\' +  file_2)
        label_CHR.append(1)
    for file_3 in os.listdir(file_dir + '\\HLD'):
        HLD.append(file_dir + '\\HLD' + '\\' + file_3)
        label_HLD.append(2)
    for file_4 in os.listdir(file_dir + '\\LEVIN'):
        LEVIN.append(file_dir + '\\LEVIN' + '\\' +  file_4)
        label_LEVIN.append(3)
    for file_5 in os.listdir(file_dir + '\\YARIS'):
        YARIS.append(file_dir + '\\YARIS' + '\\' +  file_5)
        label_YARIS.append(4)
    print('There are %d CAMARY\nThere are %d CHR\n There are %d HLD\n There are %d LEVIN\n There are %d YARIS' %(len(CAMARY), len(CHR), len(HLD), len(LEVIN), len(YARIS)))

    # 按水平方向堆叠数组构成一个新的数组
    image_list = np.hstack((CAMARY, CHR, HLD, LEVIN, YARIS))
    label_list = np.hstack((label_CAMARY, label_CHR, label_HLD, label_LEVIN, label_YARIS))

    # 存储单一数据类型的多维数组，省去指针，减少数组运算的内存和cpu
    temp = np.array([image_list, label_list])

    # 转置两个数组
    # translate [file_dir, label], [file_dir, label]...
    temp = temp.transpose()

    # random
    np.random.shuffle(temp)

    # 提取文件路径和标签到新的list
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    # 将标签转换为数字形式
    label_list = [int(float(i)) for i in label_list]

    """
    # 将部分转为训练集、部分转为测试集
    # math.ceil向上取整
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
    # step_1, 将上面生成的list传入get_batch, 转换类型，生成一个输入队列queue
    # tf.cast()用来做类型转换
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # tf.train.slice_input_producer是一个tensor生成器
    # 作用是按照设定，每次从一个tensor列表中按顺序或随机抽取一个tensor放入文件队列中
    input_queue = tf.train.slice_input_producer([image, label], num_epochs=None, shuffle=False)
    #第三个参数shuffle： bool类型，设置是否打乱样本的顺序。一般情况下，如果shuffle=True，生成的样本顺序就被打乱了，在批处理的时候不需要再次打乱样本，使用 tf.train.batch函数就可以了;如果shuffle=False,就需要在批处理时候使用 tf.train.shuffle_batch函数打乱样本。
    label = input_queue[1]
    # Tensor("input_producer/GatherV2_1:0", shape=(), dtype=int32)
    image_contants = tf.read_file(input_queue[0])
    # Tensor("ReadFile:0", shape=(), dtype=string)
    # step_2, 将图片解码，使用相同类型的图片
    image = tf.image.decode_jpeg(image_contants, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)#!!
    # step_3, 图片预处理
    image = tf.image.resize_images(image, [image_h, image_w])
    image = tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
    #image = tf.image.per_image_standardization(image)
    # step_4, 生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32
    # label_batch: 1D tensor [batch_size], dtype = tf.int32
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)
    #image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=capacity - 1)
    # 重新排列label, 行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    #image_batch = tf.cast(image_batch, tf.unit8)    # 显示彩色图像
    #image_batch = tf.cast(image_batch, tf.float32)  # 显示灰度图
    image_batch = tf.cast(image_batch, tf.float32)#??
    return image_batch, label_batch

def valuation_batch(val_iamge, val_label, image_h, image_w, target_height, target_width, batch_size, capacity):
    val_iamge = tf.cast(val_iamge, tf.string)
    val_label = tf.cast(val_label, tf.int32)
    input_queue = tf.train.slice_input_producer([val_iamge, val_label], num_epochs=None, shuffle=False)
    val_label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])# tf.read_file()从队列中读取图像
    val_iamge = tf.image.decode_jpeg(image_contents, channels=3)
    val_iamge = tf.image.resize_images(val_iamge, [image_h, image_w])
    val_iamge = tf.image.resize_image_with_crop_or_pad(val_iamge, target_height, target_width)#!!
    image_batch, label_batch = tf.train.batch([val_iamge, val_label], batch_size=batch_size, num_threads=16, capacity=capacity)
    val_label_batch = tf.reshape(label_batch, [batch_size])
    val_image_batch = tf.cast(image_batch, tf.float32)
    return val_image_batch, val_label_batch
