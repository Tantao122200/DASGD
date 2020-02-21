import os
import random
import tensorflow as tf


def get_label_id_label_number_dir():
    my_label_id_label_number = {}
    with open('./tiny-imagenet-200/wnids.txt') as metafile:
        records = metafile.readlines()
        for i in range(len(records)):
            fields = records[i].strip().split('\t')
            label = i
            text = fields[0]
            my_label_id_label_number[text] = label

    return my_label_id_label_number


label_id_label_number = get_label_id_label_number_dir()


def get_train_data(data_dir):
    x = []
    y = []

    id_list = os.listdir(data_dir)

    images_labels_list = []
    for id in id_list:
        id_dir = os.path.join(data_dir, id)
        images_dir = os.path.join(id_dir, "images")
        image_list = os.listdir(images_dir)
        label_number = str(label_id_label_number[id])
        for image in image_list:
            image_path = os.path.join(images_dir, image)
            images_labels_list.append(image_path + ',' + label_number)

    random.shuffle(images_labels_list)

    for item in images_labels_list:
        image_path, label = item.split(",")[0], item.split(",")[1]
        x.append(image_path)
        y.append(int(label))

    return x, y


def get_test_image_name_label_id_dir():
    image_name_label_id = {}
    with open('./tiny-imagenet-200/val/val_annotations.txt') as metafile:
        records = metafile.readlines()
        for i in range(len(records)):
            fields = records[i].strip().split('\t')
            label = fields[1]
            text = fields[0]
            image_name_label_id[text] = label

    return image_name_label_id


image_name_label_id = get_test_image_name_label_id_dir()


def get_test_data(data_dir):
    x = []
    y = []

    images_labels_list = []
    images_dir = os.path.join(data_dir, "images")
    image_name_list = os.listdir(images_dir)
    for image_name in image_name_list:
        image_path = os.path.join(images_dir, image_name)
        label_id = image_name_label_id[image_name]
        label_number = str(label_id_label_number[label_id])
        images_labels_list.append(image_path + ',' + label_number)

    random.shuffle(images_labels_list)

    for item in images_labels_list:
        image_path, label = item.split(",")[0], item.split(",")[1]
        x.append(image_path)
        y.append(int(label))

    return x, y

#200类 每一类是500张图片，一共100000张图片
train_x, train_y = get_train_data(data_dir="./tiny-imagenet-200/train")
train_y = tf.one_hot(train_y, 200, 1, 0)
#测试集有10000张图片
test_x, test_y = get_test_data(data_dir="./tiny-imagenet-200/val")
test_y = tf.one_hot(test_y, 200, 1, 0)


def decode_image(image_path):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_float = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image_float, [64, 64])

    return image_resized


def image_train_change(image_path, label):
    image_string = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.random_crop(image, [56, 56, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)

    return image, label


def image_test_change(image_path, label):
    image_string = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.random_crop(image, [56, 56, 3])
    image = tf.image.per_image_standardization(image)
    return image, label


def next_train_batch(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset = dataset.map(image_train_change, num_parallel_calls=10)
    dataset = dataset.shuffle(buffer_size=5000).batch(batch_size, drop_remainder=True).repeat()
    iteration = dataset.make_one_shot_iterator()
    one_element = iteration.get_next()
    return one_element[0], one_element[1]


def next_test_batch(batch_size=2000):
    dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    dataset = dataset.map(image_test_change, num_parallel_calls=10)
    # dataset = dataset.prefetch(-1)
    # dataset = dataset.repeat().batch(batch_size, drop_remainder=True)
    dataset = dataset.shuffle(buffer_size=5000).batch(batch_size, drop_remainder=True).repeat()
    iteration = dataset.make_one_shot_iterator()
    one_element = iteration.get_next()
    return one_element[0], one_element[1]
