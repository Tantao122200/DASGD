from matplotlib import pyplot as plt
import pandas as pd
import os, shutil
# from scipy.interpolate import spline
# import numpy as np

# 线条标记
# 圆圈 o 小菱形 d 菱形 D
# 正方形 s 五边形 p 六边形1 h 六边形2 H 八边形 8
# 水平线 _ 竖线 | 加号 + 点 . 像素 ,  星号 * x X 无 None '' ' '
# 1角朝上三角形 ^ 1角朝下三角形 v 1角朝左三角形 < 1角朝右三角形 >

mymarker = ["o", "d", "S"]


def mypath(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def smooth(x, y, weight=0.99):
    y_ = []
    last = y[0]
    for point in y:
        value = last * weight + (1 - weight) * point
        y_.append(value)
        last = value
    return x, y_

def show(i, x, name_x, y, name_y, label, marker):
    # x_new = np.linspace(x.min(), x.max(),1000)
    # y_new = spline(x, y, x_new)
    # x_new, y_new = smooth(x, y)
    plt.figure(i)
    plt.ylabel(name_y)
    plt.xlabel(name_x)
    plt.plot(x, y, label=label)
    plt.legend()
    plt.savefig(path + str(i) + ".png")


def picture(iter_step, loss_total, acc_total, lr_total, time_total, name_, marker):
    x_total = [iter_step, iter_step, iter_step, time_total, time_total, iter_step]
    y_total = [loss_total, acc_total, time_total, loss_total, acc_total, lr_total]
    name_y_total = ["loss", "acc", "time", "loss", "acc", "lr"]
    name_x_total = ["iteration", "iteration", "iteration", "time", "time", "iteration"]
    name = ["the loss with iteration", "the acc with iteration", "the time of iteration",
            "the loss with time", "the acc with time", "the lr with iteration"]
    for i in range(1, 7):
        plt.figure(i)
        plt.title(name[i - 1])
        plt.grid()
        show(i, x_total[i - 1], name_x_total[i - 1], y_total[i - 1], name_y_total[i - 1], name_, marker)


if __name__ == "__main__":
    path = "./test/ASGD_1_3_3_3/"
    mypath(path=path)

    # train_result = pd.read_csv("./train_ASGD_0.1_62400_128_1.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGD_0.1_62400_128_1", mymarker[0])
    # train_result = pd.read_csv("./train_ASGD_0.1_62400_128_3.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGD_0.1_62400_128_3", mymarker[0])
    # train_result = pd.read_csv("./train_ASGDMK_0.1_62400_128_3.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGDMK_0.1_62400_128_3", mymarker[0])
    # train_result = pd.read_csv("./train_ASGDMT_0.1_62400_128_3.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGDMT_0.1_62400_128_3", mymarker[0])
    # train_result = pd.read_csv("./train_ASGD_0.1_62400_128_6.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGD_0.1_62400_128_6", mymarker[0])
    # train_result = pd.read_csv("./train_ASGDMK_0.1_62400_128_6.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGDMK_0.1_62400_128_6", mymarker[0])
    # train_result = pd.read_csv("./train_ASGDMT_0.1_62400_128_6.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGDMT_0.1_62400_128_6", mymarker[0])

    train_result = pd.read_csv("./test_ASGD_0.1_62400_128_1.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
            "ASGD_0.1_62400_128_1", mymarker[0])
    train_result = pd.read_csv("./test_ASGD_0.1_62400_128_3.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
            "ASGD_0.1_62400_128_3", mymarker[0])
    train_result = pd.read_csv("./test_ASGDMK_0.1_62400_128_3.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
            "ASGDMK_0.1_62400_128_3", mymarker[0])
    train_result = pd.read_csv("./test_ASGDMT_0.1_62400_128_3.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
            "ASGDMT_0.1_62400_128_3", mymarker[0])
    # train_result = pd.read_csv("./test_ASGD_0.1_62400_128_6.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGD_0.1_62400_128_6", mymarker[0])
    # train_result = pd.read_csv("./test_ASGDMK_0.1_62400_128_6.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGDMK_0.1_62400_128_6", mymarker[0])
    # train_result = pd.read_csv("./test_ASGDMT_0.1_62400_128_6.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture(train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGDMT_0.1_62400_128_6", mymarker[0])
