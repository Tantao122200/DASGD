import tensorflow as tf
from model import vgg16

global yan_chi, images, labels, dropout_rate
global top1_acc, top5_acc, c_loss, loss
global train_op, lr, global_step
global x_train, y_train
global x_test, y_test


def build_model():
    global yan_chi, images, labels, dropout_rate
    global top1_acc, top5_acc, c_loss, loss
    global train_op, lr, global_step
    global x_train, y_train
    global x_test, y_test

    yan_chi = tf.placeholder(tf.float32)
    images = tf.placeholder(tf.float32, [None, 56, 56, 3])
    labels = tf.placeholder(tf.float32, [None, 200])
    dropout_rate = tf.placeholder(tf.float32)

    x_train, y_train = vgg16.train_input()
    x_test, y_test = vgg16.test_input()

    logits = vgg16.inference(images, dropout_rate)
    top1_acc, top5_acc = vgg16.get_acc(logits, labels)
    c_loss, loss = vgg16.get_loss(logits, labels)
    train_op, lr, global_step = vgg16.get_op(yan_chi, loss)


def train_model(session, step):
    global yan_chi, images, labels, dropout_rate
    global top1_acc, top5_acc, c_loss, loss
    global train_op, lr, global_step
    global x_train, y_train
    global x_test, y_test

    g_step = session.run(global_step)
    train_x, train_y = session.run([x_train, y_train])
    my_yan_chi = g_step - step
    feed_dict = {images: train_x, labels: train_y, yan_chi: my_yan_chi, dropout_rate:0.5}
    _, train_loss_value, cross_loss_value, train_acc1_value, train_acc5_value, lr_value = session.run([train_op, loss, c_loss, top1_acc, top5_acc, lr],
                                                                                   feed_dict=feed_dict)
    current_info = "type is train,global_step is %d,train_loss is %.5f,corss_loss is %.5f,top1_acc is %.5f,top5_acc is %.5f,yan_chi is %.4f,lr is %.5f" % (
        g_step, train_loss_value, cross_loss_value, train_acc1_value, train_acc5_value, my_yan_chi, lr_value)
    yield (current_info)  # 用yield传出需要打印的信息

    if (g_step % vgg16.COUNT == 0 or g_step % vgg16.COUNT == 1 or g_step % vgg16.COUNT == 2) and g_step > 0:
        test_x, test_y = session.run([x_test, y_test])
        feed_dict = {images: test_x, labels: test_y, dropout_rate:1.0}
        test_loss_value, cross_loss_value, test_acc1_value, test_acc5_value, lr_value = session.run([loss, c_loss, top1_acc, top5_acc, lr],
                                                                                  feed_dict=feed_dict)
        current_info = "type is test,global_step is %d,test_loss is %.5f,corss_loss is %.5f,top1_acc is %.5f,top5_acc is %.5f,yan_chi is %.4f,lr is %.5f" % (
            g_step, test_loss_value, cross_loss_value, test_acc1_value, test_acc5_value,  my_yan_chi, lr_value)
        yield (current_info)
