import tensorflow as tf
from myoptimizer import GD, ASGD_MK, ASGD_MT
from input import input_32
from config import worker_hosts

DATA_DIR = "./data"
BATCH_SIZE = 128
COUNT = 390
TRAININR_STEP = COUNT * 160

global_step = tf.Variable(0, name="global_step", trainable=False)
boundaries = [COUNT * 40, COUNT * 80, COUNT * 120]
values = [0.1, 0.01, 0.001, 0.0001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)


def train_input():
    return input_32.next_train_batch(BATCH_SIZE)


def test_input():
    return input_32.next_test_batch()


def variable_with_weight_decay(shape, stddev, wd):
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def variable_on_cpu(shape, initializer):
    with tf.device('/cpu:0'):
        initial = tf.constant(value=initializer, shape=shape, dtype=tf.float32)
        var = tf.Variable(initial)
    return var


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def norm(value):
    return tf.nn.lrn(value, 4, bias=1.0, alpha=0.001 / 9, beta=0.75)


def inference(x_image):
    wc1 = variable_with_weight_decay(shape=[5, 5, 3, 64], stddev=5e-2, wd=None)
    bc1 = variable_on_cpu([64], 0.0)
    conv1 = conv2d(x_image, wc1) + bc1
    conv1 = tf.nn.relu(conv1)

    pool1 = max_pool(conv1)
    norm1 = norm(pool1)

    wc2 = variable_with_weight_decay(shape=[5, 5, 64, 64], stddev=5e-2, wd=None)
    bc2 = variable_on_cpu([64], 0.1)
    conv2 = conv2d(norm1, wc2) + bc2
    conv2 = tf.nn.relu(conv2)

    norm2 = norm(conv2)
    pool2 = max_pool(norm2)

    reshape = tf.keras.layers.Flatten()(pool2)
    dim = reshape.get_shape()[1].value

    wfc1 = variable_with_weight_decay(shape=[dim, 1024], stddev=0.04, wd=0.004)
    bfc1 = variable_on_cpu([1024], 0.1)
    fc1 = tf.matmul(reshape, wfc1) + bfc1
    fc1 = tf.nn.relu(fc1)

    wfc2 = variable_with_weight_decay(shape=[1024, 1024], stddev=0.04, wd=0.004)
    bfc2 = variable_on_cpu([1024], 0.1)
    fc2 = tf.matmul(fc1, wfc2) + bfc2
    fc2 = tf.nn.relu(fc2)

    wfc3 = variable_with_weight_decay(shape=[1024, 10], stddev=1 / 1024.0, wd=None)
    bfc3 = variable_on_cpu([10], 0.0)
    fc3 = tf.matmul(fc2, wfc3) + bfc3

    return fc3


def get_loss(logits, labels):
    cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    tf.add_to_collection('losses', cross_entropy_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return cross_entropy_mean, total_loss


def get_acc(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def add_loss_summaries(total_loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  return loss_averages_op


def get_op(yan_chi, total_loss):
    loss_averages_op = add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        #opt = GD.GradientDescentOptimizer(learning_rate)
        #opt = ASGD_MK.ASGDMK(learning_rate, yanchi=yan_chi, count=len(worker_hosts))
        opt = ASGD_MT.ASGDMT(learning_rate=learning_rate, yanchi=yan_chi, momentum=0.8)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    return variables_averages_op, learning_rate, global_step
