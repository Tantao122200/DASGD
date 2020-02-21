import tensorflow as tf
import numpy as np
from myoptimizer import GD,ASGD_MK,ASGD_MT
from input import input_32
from config import worker_hosts


BATCH_SIZE = 500
COUNT = 200
TRAININR_STEP = COUNT * 200

global_step = tf.Variable(0, name="global_step", trainable=False)
boundaries = [COUNT * 40, COUNT * 80,COUNT*120,COUNT*160]
values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)


def train_input():
    return input_32.next_train_batch(BATCH_SIZE)


def test_input():
    return input_32.next_test_batch()


def conv_2d(inputs, filters, kernel_size):

  stddev = np.sqrt(2 / (np.prod(kernel_size) * int(inputs.shape[3])))
  out = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                         padding='same', activation=tf.nn.relu,
                         kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0))
  return out


def dense_relu(inputs, units, name=None):
  stddev = np.sqrt(2 / int(inputs.shape[1]))
  out = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                        kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                        name=name)
  return out


def dense(inputs, units, name=None):

  stddev = np.sqrt(2 / int(inputs.shape[1]))
  out = tf.layers.dense(inputs, units,
                        kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                        name=name)
  return out


"""
Tiny ImageNet Model
1. conv-conv-maxpool
2. conv-conv-maxpool
3. conv-conv-maxpool
4. conv-conv-conv-maxpool
5. fc-4096 (ReLU)
6. fc-4096 (ReLU)
7. fc-200
"""

def inference(x_image, dropout_rate):

  # (N, 56, 56, 3)
  out = conv_2d(x_image, 64, (3, 3))
  out = conv_2d(out, 64, (3, 3))
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2))

  # (N, 28, 28, 64)
  out = conv_2d(out, 128, (3, 3))
  out = conv_2d(out, 128, (3, 3))
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2))

  # (N, 14, 14, 128)
  out = conv_2d(out, 256, (3, 3))
  out = conv_2d(out, 256, (3, 3))
  out = conv_2d(out, 256, (3, 3))
  out = tf.layers.max_pooling2d(out, (2, 2), (2, 2))

  # (N, 7, 7, 256)
  out = conv_2d(out, 512, (3, 3))
  out = conv_2d(out, 512, (3, 3))
  out = conv_2d(out, 512, (3, 3))

  out = tf.contrib.layers.flatten(out)
  out = dense_relu(out, 4096)
  out = tf.nn.dropout(out, dropout_rate)

  out = dense_relu(out, 4096)
  out = tf.nn.dropout(out, dropout_rate)

  logits = dense(out, 200)

  return logits


def get_loss(logits, labels):
    weight_decay = 5e-4
    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    # l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * weight_decay
    total_loss = cross_entropy_mean + l2_loss
    return cross_entropy_mean, total_loss


def get_acc(logits, labels):
    top1_correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    top1_accuracy = tf.reduce_mean(tf.cast(top1_correct_prediction, tf.float32))

    top5_correct_prediction = tf.nn.in_top_k(logits, tf.argmax(labels, 1), k=5)
    top5_accuracy = tf.reduce_mean(tf.cast(top5_correct_prediction, tf.float32))
    return top1_accuracy, top5_accuracy


def add_loss_summaries(total_loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  return loss_averages_op


def get_op(yan_chi, total_loss):
    loss_averages_op = add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        # opt = GD.GradientDescentOptimizer(learning_rate)
        # opt = ASGD_MK.ASGDMK(learning_rate, yanchi=yan_chi, count=len(worker_hosts))
        opt = ASGD_MT.ASGDMT(learning_rate=learning_rate, yanchi=yan_chi, momentum=0.5)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op, learning_rate, global_step
