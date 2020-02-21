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


def picture(type, iter_step, loss_total, top1_total, top5_total, lr_total, time_total, name_):
    x_total = [iter_step, iter_step, iter_step, iter_step, time_total,  time_total, time_total, iter_step]
    y_total = [loss_total, top1_total, top5_total, time_total, loss_total, top1_total, top5_total, lr_total]
    name_y_total = ["loss", "acc_top1", "acc_top5", "time", "loss", "acc_top1", "acc_top5", "lr"]
    name_x_total = ["iteration", "iteration", "iteration", "iteration", "time", "time", "time", "iteration"]
    name = ["the " + type + " loss with iteration", "the " + type + " acc_top1 with iteration", "the " + type + " acc_top5 with iteration",  "the " + type + " time of iteration",
            "the " + type + " loss with time", "the " + type + " acc_top1 with time", "the " + type + " acc_top5 with time", "the " + type + " lr with iteration"]
    for i in range(1, 9):
        x = list(x_total[i - 1])
        y = list(y_total[i - 1])
        plt.figure(i)
        plt.title(name[i - 1])
        plt.grid()
        if i==5 or i==6 or i==7:
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
    path = "./test/ASGD_6_6_6/"
    mypath(path=path)

    test_result = pd.read_csv("./test_ASGD_0.1_40000_500_6.csv", header=None, usecols=[0, 2, 3, 4, 6, 7]).values.T
    picture("test", test_result[0], test_result[1], test_result[2], test_result[3], test_result[4],
            test_result[5],
            "ASGD_6")
    test_result = pd.read_csv("./test_ASGDMK_0.1_40000_500_6.csv", header=None, usecols=[0, 2, 3, 4, 6, 7]).values.T
    picture("test", test_result[0], test_result[1], test_result[2], test_result[3], test_result[4],
            test_result[5],
            "MDCASGD_6")
    test_result = pd.read_csv("./test_ASGDMT_0.1_40000_500_6.csv", header=None, usecols=[0, 2, 3, 4, 6, 7]).values.T
    picture("test", test_result[0], test_result[1], test_result[2], test_result[3], test_result[4],
            test_result[5],
            "DASGD_6")

