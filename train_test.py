import pandas as pd
from sklearn.model_selection import train_test_split
from data_process import data_get
import numpy as np

def MorganFingerprint_train_test3():
    data = data_get()
    x = data.drop(["粘度（mPa·s）"],axis=1)
    # y = (data["粘度（mPa·s）"]-data["粘度（mPa·s）"].mean())/data["粘度（mPa·s）"].std()
    y = np.log(data["粘度（mPa·s）"])
    # y = data["粘度（mPa·s）"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    a = x_train["HBA_MorganFingerprint"] + x_train["HBD_MorganFingerprint"]
    a = a.reset_index(drop=True)
    # b = x_train[["water0","water1","water2","M","温度","M(HBA)","M(HBD)","ratio"]]
    b = x_train[["water0", "water1", "water2", "温度", "M(HBA)", "M(HBD)", "ratio"]]
    b = b.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    for i in range(0, len(a)):
        a[i].append(b.loc[i, "water0"])
        a[i].append(b.loc[i, "water1"])
        a[i].append(b.loc[i, "water2"])
        # a[i].append(b.loc[i, "M"])
        a[i].append(b.loc[i, "温度"])
        a[i].append(b.loc[i, "M(HBA)"])
        a[i].append(b.loc[i, "M(HBD)"])
        a[i].append(b.loc[i, "ratio"])
    train_x = a

    a1 = x_test["HBA_MorganFingerprint"] + x_test["HBD_MorganFingerprint"]
    a1 = a1.reset_index(drop=True)
    # b1 = x_test[["water0","water1","water2","M","温度","M(HBA)","M(HBD)","ratio"]]
    b1 = x_test[["water0", "water1", "water2", "温度", "M(HBA)", "M(HBD)", "ratio"]]
    b1 = b1.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    for i in range(0, len(a1)):
        a1[i].append(b1.loc[i, "water0"])
        a1[i].append(b1.loc[i, "water1"])
        a1[i].append(b1.loc[i, "water2"])
        # a1[i].append(b1.loc[i, "M"])
        a1[i].append(b1.loc[i, "温度"])
        a1[i].append(b1.loc[i, "M(HBA)"])
        a1[i].append(b1.loc[i, "M(HBD)"])
        a1[i].append(b1.loc[i, "ratio"])
    test_x = a1
    return train_x, test_x, y_train, y_test

def MorganFingerprint_train_test4():
    data = data_get()
    x = data.drop(["粘度（mPa·s）"],axis=1)
    # y = (data["粘度（mPa·s）"]-data["粘度（mPa·s）"].mean())/data["粘度（mPa·s）"].std()
    y = np.log(data["粘度（mPa·s）"])
    # y = data["粘度（mPa·s）"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    a = x_train["HBA_MorganFingerprint"] + x_train["HBD_MorganFingerprint"]
    a = a.reset_index(drop=True)
    b = x_train[["M","温度","M(HBA)","M(HBD)","ratio"]]
    b = b.reset_index(drop=True)
    for i in range(0, len(a)):
        a[i].append(b.loc[i, "M"])
        a[i].append(b.loc[i, "温度"])
        a[i].append(b.loc[i, "M(HBA)"])
        a[i].append(b.loc[i, "M(HBD)"])
        a[i].append(b.loc[i, "ratio"])
    train_x = pd.DataFrame(a.tolist())

    a1 = x_test["HBA_MorganFingerprint"] + x_test["HBD_MorganFingerprint"]
    a1 = a1.reset_index(drop=True)
    b1 = x_test[["M","温度","M(HBA)","M(HBD)","ratio"]]
    b1 = b1.reset_index(drop=True)
    for i in range(0, len(a1)):
        a1[i].append(b1.loc[i, "M"])
        a1[i].append(b1.loc[i, "温度"])
        a1[i].append(b1.loc[i, "M(HBA)"])
        a1[i].append(b1.loc[i, "M(HBD)"])
        a1[i].append(b1.loc[i, "ratio"])
    test_x = pd.DataFrame(a1.tolist())
    return train_x, test_x, y_train, y_test

def MorganFingerprint_train_test5():
    data = data_get()
    x = data.drop(["粘度（mPa·s）"],axis=1)
    # y = (data["粘度（mPa·s）"]-data["粘度（mPa·s）"].mean())/data["粘度（mPa·s）"].std()
    y = np.log(data["粘度（mPa·s）"])
    # y = data["粘度（mPa·s）"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    a = x_train["HBA_MorganFingerprint"] + x_train["HBD_MorganFingerprint"]
    a = a.reset_index(drop=True)
    b = x_train[["water0","water1","water2","温度","M(HBA)","M(HBD)","ratio"]]
    b = b.reset_index(drop=True)
    for i in range(0, len(a)):
        a[i].append(b.loc[i, "water0"])
        a[i].append(b.loc[i, "water1"])
        a[i].append(b.loc[i, "water2"])
        a[i].append(b.loc[i, "温度"])
        a[i].append(b.loc[i, "M(HBA)"])
        a[i].append(b.loc[i, "M(HBD)"])
        a[i].append(b.loc[i, "ratio"])
    train_x = a

    a1 = x_test["HBA_MorganFingerprint"] + x_test["HBD_MorganFingerprint"]
    a1 = a1.reset_index(drop=True)
    b1 = x_test[["water0","water1","water2","温度","M(HBA)","M(HBD)","ratio"]]
    b1 = b1.reset_index(drop=True)
    for i in range(0, len(a1)):
        a1[i].append(b1.loc[i, "water0"])
        a1[i].append(b1.loc[i, "water1"])
        a1[i].append(b1.loc[i, "water2"])
        a1[i].append(b1.loc[i, "温度"])
        a1[i].append(b1.loc[i, "M(HBA)"])
        a1[i].append(b1.loc[i, "M(HBD)"])
        a1[i].append(b1.loc[i, "ratio"])
    test_x = a1
    return train_x, test_x, y_train, y_test

def MorganFingerprint_train_test6():
    data = data_get()
    x = data.drop(["粘度（mPa·s）"],axis=1)
    # y = (data["粘度（mPa·s）"]-data["粘度（mPa·s）"].mean())/data["粘度（mPa·s）"].std()
    y = data["粘度（mPa·s）"]
    # y = data["粘度（mPa·s）"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    a = x_train["HBA_MorganFingerprint"] + x_train["HBD_MorganFingerprint"]
    a = a.reset_index(drop=True)
    # b = x_train[["water0","water1","water2","M","温度","M(HBA)","M(HBD)","ratio"]]
    b = x_train[["water0", "water1", "water2", "温度", "M(HBA)", "M(HBD)", "ratio"]]
    b = b.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    for i in range(0, len(a)):
        a[i].append(b.loc[i, "water0"])
        a[i].append(b.loc[i, "water1"])
        a[i].append(b.loc[i, "water2"])
        # a[i].append(b.loc[i, "M"])
        a[i].append(b.loc[i, "温度"])
        a[i].append(b.loc[i, "M(HBA)"])
        a[i].append(b.loc[i, "M(HBD)"])
        a[i].append(b.loc[i, "ratio"])
    train_x = a

    a1 = x_test["HBA_MorganFingerprint"] + x_test["HBD_MorganFingerprint"]
    a1 = a1.reset_index(drop=True)
    # b1 = x_test[["water0","water1","water2","M","温度","M(HBA)","M(HBD)","ratio"]]
    b1 = x_test[["water0", "water1", "water2", "温度", "M(HBA)", "M(HBD)", "ratio"]]
    b1 = b1.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    for i in range(0, len(a1)):
        a1[i].append(b1.loc[i, "water0"])
        a1[i].append(b1.loc[i, "water1"])
        a1[i].append(b1.loc[i, "water2"])
        # a1[i].append(b1.loc[i, "M"])
        a1[i].append(b1.loc[i, "温度"])
        a1[i].append(b1.loc[i, "M(HBA)"])
        a1[i].append(b1.loc[i, "M(HBD)"])
        a1[i].append(b1.loc[i, "ratio"])
    test_x = a1
    return train_x, test_x, y_train, y_test

# train_x, test_x, y_train, y_test = MorganFingerprint_train_test3()
# data = train_x
# data = data.append(test_x)
# data = data.reset_index(drop=True)
# y_train = y_train.append(y_test)
# y_train = y_train.reset_index(drop=True)
# for i in range(len(data)):
#     data[i].append(y_train[i])
# data = data.tolist()
# data = pd.DataFrame(data)
# data.to_csv("data.csv",index=False)

train_x,test_x,train_y,test_y = MorganFingerprint_train_test6()

train_x  = train_x.values.tolist()
test_x = test_x.values.tolist()
x = pd.DataFrame(train_x)
x = x.reset_index(drop=True)
x = x.values.tolist();
y = train_y.tolist()
for i in range(len(x)):
    x[i].append(y[i])
x = pd.DataFrame(x)
x.to_csv("./data/train.csv",index=False)

data = pd.DataFrame(test_x)
data = data.reset_index(drop=True)
data = data.values.tolist();
test_y = test_y.tolist()
for i in range(len(data)):
    data[i].append(test_y[i])
data = pd.DataFrame(data)
data.to_csv("./data/test.csv",index=False)