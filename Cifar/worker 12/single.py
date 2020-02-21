import tensorflow as tf
from model import LeNet5

global yan_chi, images, labels
global acc, loss, c_loss
global train_op, lr, global_step
global x_train, y_train
global x_test, y_test


def build_model():
    global yan_chi, images, labels
    global acc, loss, c_loss
    global train_op, lr, global_step
    global x_train, y_train
    global x_test, y_test

    yan_chi = tf.placeholder(tf.float32)
    images = tf.placeholder(tf.float32, [None, 24, 24, 3])
    labels = tf.placeholder(tf.float32, [None, 10])

    x_train, y_train = LeNet5.train_input()
    x_test, y_test = LeNet5.test_input()

    logits = LeNet5.inference(images)
    acc = LeNet5.get_acc(logits, labels)
    c_loss, loss = LeNet5.get_loss(logits, labels)
    train_op, lr, global_step = LeNet5.get_op(yan_chi, loss)


def train_model(session, step):
    global yan_chi, images, labels
    global acc, loss, c_loss
    global train_op, lr, global_step
    global x_train, y_train
    global x_test, y_test

    g_step = session.run(global_step)
    train_x, train_y = session.run([x_train, y_train])
    my_yan_chi = g_step - step
    feed_dict = {images: train_x, labels: train_y, yan_chi: my_yan_chi}
    _, train_loss_value, cross_loss_value, train_acc_value, lr_value = session.run([train_op, loss, c_loss, acc, lr],
                                                                                   feed_dict=feed_dict)
    current_info = "type is train,global_step is %d,loss is %.5f,corss_loss is %.5f,acc is %.5f,yan_chi is %.4f,lr is %.5f" % (
        g_step, train_loss_value, cross_loss_value, train_acc_value, my_yan_chi, lr_value)
    yield (current_info)  # 用yield传出需要打印的信息

    if g_step % LeNet5.COUNT == 0:
        test_x, test_y = session.run([x_test, y_test])
        feed_dict = {images: test_x, labels: test_y}
        test_loss_value, cross_loss_value, test_acc_value, lr_value = session.run([loss, c_loss, acc, lr],
                                                                                  feed_dict=feed_dict)
        current_info = "type is test,global_step is %d,loss is %.5f,corss_loss is %.5f,acc is %.5f,yan_chi is %.4f,lr is %.5f" % (
            g_step, test_loss_value, cross_loss_value, test_acc_value, my_yan_chi, lr_value)
        yield (current_info)
