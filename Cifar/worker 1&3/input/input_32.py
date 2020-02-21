import pickle
import numpy as np
import tensorflow as tf

'''返回的dict是一个包含0-9数字的list列表，如[0,2,4,6,3,5,...,4,2,5]'''


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


'''将上面的dict转换成对应的one-hot矩阵，shape=[n_sample,n_class]'''


def onehot(labels):
    n_sample = len(labels)  # 数据集的数量
    n_class = max(labels) + 1  # one_hot分类的数量
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


def get_train():
    # 训练数据集
    data1 = unpickle('./data/cifar-10-batches-py/data_batch_1')
    data2 = unpickle('./data/cifar-10-batches-py/data_batch_2')
    data3 = unpickle('./data/cifar-10-batches-py/data_batch_3')
    data4 = unpickle('./data/cifar-10-batches-py/data_batch_4')
    data5 = unpickle('./data/cifar-10-batches-py/data_batch_5')

    x_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']), axis=0)
    y_train = np.concatenate((data1['labels'], data2['labels'], data3['labels'], data4['labels'], data5['labels']),
                             axis=0)
    # 转换格式
    y_train = onehot(y_train)
    return x_train, y_train


def get_test():
    # 测试集
    test = unpickle('./data/cifar-10-batches-py/test_batch')
    x_test = test['data']
    y_test = onehot(test['labels'])
    return x_test, y_test


x_train, y_train = get_train()
x_test, y_test = get_test()


def image_train_change(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])
    # 图片随机剪裁
    image = tf.random_crop(image, [24, 24, 3])
    # 图片随机翻转
    image = tf.image.random_flip_left_right(image)
    # 图片随机调整亮度
    image = tf.image.random_brightness(image, max_delta=63)
    # 图片随机调整对比度
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # 归一化处理
    image = tf.image.per_image_standardization(image)
    return image, label


def next_train_batch(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.map(image_train_change, num_parallel_calls=10)
    dataset = dataset.prefetch(-1)
    dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
    iteration = dataset.make_one_shot_iterator()
    one_element = iteration.get_next()
    return one_element[0], one_element[1]


def image_test_change(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])
    # 图片随机剪裁
    image = tf.image.resize_image_with_crop_or_pad(image, 24, 24)
    # 归一化处理
    image = tf.image.per_image_standardization(image)
    return image, label


def next_test_batch(batch_size=len(x_test)):
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.map(image_test_change, num_parallel_calls=10)
    dataset = dataset.prefetch(-1)
    dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
    iteration = dataset.make_one_shot_iterator()
    one_element = iteration.get_next()
    return one_element[0], one_element[1]
