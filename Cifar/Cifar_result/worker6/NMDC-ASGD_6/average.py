import pandas as pd
import numpy as np
import os, shutil

MYCOUNT = 5
import csv


def mypath(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def read(in_dir, out_dir):
    data = []
    for i in range(1, MYCOUNT + 1):
        temp_data = pd.read_csv(in_dir + str(i) + ".csv", header=None, usecols=[0, 1, 2, 3, 4, 5, 6]).values
        data.extend(temp_data.tolist())
    data.sort()
    out = open(out_dir, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for i in range(161):
        temp = []
        for row in data:
            if row[0] < i * 390:
                pass
            elif row[0] == i * 390:
                row[0] = i
                temp.append(row)
            else:
                break
        if len(temp) > 0:
            csv_write.writerow(list_avg(temp))


def list_add(a, b):
    a = np.array(a)
    b = np.array(b)
    return list(a + b)


def list_avg(temp):
    sum = temp[0]
    for i in range(1, len(temp)):
        sum = list_add(sum, temp[i])
    sum = np.array(sum)
    return list(sum / len(temp))


if __name__ == "__main__":
    mypath("./msg")
    in_dir = ["./train_ASGDMT_0.8_0.1_62400_128_6_", "./test_ASGDMT_0.8_0.1_62400_128_6_"]
    out_dir = ["./msg/train_ASGDMT_0.1_62400_128_6.csv", "./msg/test_ASGDMT_0.1_62400_128_6.csv"]
    for index in range(len(in_dir)):
        read(in_dir=in_dir[index], out_dir=out_dir[index])
