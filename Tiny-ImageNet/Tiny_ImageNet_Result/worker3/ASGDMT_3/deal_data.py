import pandas as pd
import numpy as np
import os, shutil

MYCOUNT = 1
import csv


def mypath(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def deal_data(in_dir, deal_dir):
    for i in range(1, MYCOUNT + 1):
        temp_data = pd.read_csv(in_dir + str(i) + ".csv", header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7]).values
        temp_data=temp_data.tolist()
        temp = []
        for row in temp_data:
            row[0] = int(row[0] / 200)
            temp.append(row)
        mydeal_dir=deal_dir + str(i) + ".csv"
        out = open(mydeal_dir, 'a', newline='')
        csv_write = csv.writer(out, dialect='excel')
        for i in range(len(temp)):
            csv_write.writerow(temp[i])


if __name__ == "__main__":
    mypath("./deal_data")
    in_dir = ["./data/train_ASGDMT_0.1_40000_500_3_", "./data/test_ASGDMT_0.1_40000_500_3_"]
    deal_dir = ["./deal_data/train_ASGDMT_0.1_40000_500_3_", "./deal_data/test_ASGDMT_0.1_40000_500_3_"]
    for index in range(len(in_dir)):
        deal_data(in_dir=in_dir[index], deal_dir=deal_dir[index])
