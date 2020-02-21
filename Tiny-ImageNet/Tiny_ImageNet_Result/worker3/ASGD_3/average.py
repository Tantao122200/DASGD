import pandas as pd
import numpy as np
import os, shutil

MYCOUNT = 1
import csv


def mypath(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def read(in_dir, out_dir):
    data = []
    for i in range(1, MYCOUNT + 1):
        temp_data = pd.read_csv(in_dir + str(i) + ".csv", header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7]).values
        data.extend(temp_data.tolist())
    data.sort()
    out = open(out_dir, 'a', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for i in range(201):
        temp = []
        for row in data:
            if row[0] < i:
                pass
            elif row[0] == i:
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
    in_dir = ["./deal_data/train_ASGD_0.1_40000_500_3_", "./deal_data/test_ASGD_0.1_40000_500_3_"]
    out_dir = ["./msg/train_ASGD_0.1_40000_500_3.csv", "./msg/test_ASGD_0.1_40000_500_3.csv"]
    for index in range(len(in_dir)):
        read(in_dir=in_dir[index], out_dir=out_dir[index])
