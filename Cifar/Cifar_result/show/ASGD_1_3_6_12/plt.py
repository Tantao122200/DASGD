from matplotlib import pyplot as plt
import pandas as pd
import os, shutil

def mypath(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def show(i, x, name_x, y, name_y, label):
    plt.figure(i)
    plt.ylabel(name_y)
    plt.xlabel(name_x)
    plt.plot(x, y, label=label)
    plt.legend()
    plt.savefig(path + str(i) + ".png")


def picture(type, iter_step, loss_total, acc_total, lr_total, time_total, name_):
    x_total = [iter_step, iter_step, iter_step, time_total, time_total, iter_step]
    y_total = [loss_total, acc_total, time_total, loss_total, acc_total, lr_total]
    name_y_total = ["loss", "acc", "time", "loss", "acc", "lr"]
    name_x_total = ["iteration", "iteration", "iteration", "time", "time", "iteration"]
    name = ["the " + type + " loss with iteration", "the " + type + " acc with iteration", "the " + type + " time of iteration",
            "the " + type + " loss with time", "the " + type + " acc with time", "the " + type + " lr with iteration"]
    for i in range(1, 7):
        x = list(x_total[i - 1])
        y = list(y_total[i - 1])
        plt.figure(i)
        plt.title(name[i - 1])
        plt.grid()
        if i==4 or i==5:
            m=[]
            for k in range(len(x)):
                for j in range(k+1,len(x)):
                    if x[k]>x[j]:
                        m.append(k)
                        break
            if len(m) != 0:
                for j in reversed(m):
                    del x[j]
                    del y[j]
        show(i, x, name_x_total[i - 1], y, name_y_total[i - 1], name_)


if __name__ == "__main__":
    path = "./train/ASGD_1_3_6_12/"
    mypath(path=path)

    train_result = pd.read_csv("./train_ASGD_0.1_62400_128_1.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    picture("training", train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
            "SGD")
    train_result = pd.read_csv("./train_ASGD_0.1_62400_128_3.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    picture("training", train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
            "ASGD_3")
    train_result = pd.read_csv("./train_ASGD_0.1_62400_128_6.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    picture("training", train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
            "ASGD_6")
    train_result = pd.read_csv("./train_ASGD_0.1_62400_128_12.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    picture("training", train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
            "ASGD_12")


    # path = "./test/ASGD_1_3_6_12/"
    # mypath(path=path)
    #
    # train_result = pd.read_csv("./test_ASGD_0.1_62400_128_1.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture("test", train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "SGD")
    # train_result = pd.read_csv("./test_ASGD_0.1_62400_128_3.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture("test", train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGD_3")
    # train_result = pd.read_csv("./test_ASGD_0.1_62400_128_6.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture("test", train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGD_6")
    # train_result = pd.read_csv("./test_ASGD_0.1_62400_128_12.csv", header=None, usecols=[0, 2, 3, 5, 6]).values.T
    # picture("test", train_result[0], train_result[1], train_result[2], train_result[3], train_result[4],
    #         "ASGD_12")
