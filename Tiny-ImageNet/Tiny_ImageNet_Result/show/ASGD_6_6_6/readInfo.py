import pandas as pd
MYCOUNT = 6

def read_me(in_dir):
    for i in range(6, MYCOUNT + 1):
        temp_data = pd.read_csv(in_dir + str(i) + ".csv", header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7]).values
        temp_data=temp_data.tolist()
        temp=temp_data[-1]
        print(temp)

if __name__ == "__main__":
    in_dir = ["./train_ASGD_0.1_40000_500_", "./test_ASGD_0.1_40000_500_"]
    info=["train","test"]
    for index in range(len(in_dir)):
        print(info[index])
        read_me(in_dir=in_dir[index])
