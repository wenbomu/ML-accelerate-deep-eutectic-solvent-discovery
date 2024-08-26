import pandas as pd
from model import svr, RF, XGBR, Net, k_fold
from sklearn.metrics import mean_squared_error,r2_score 
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import torch
import torch.nn as nn
import sys
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

# data = pd.read_csv("gan_train1.csv")
# X = data.drop(["108"],axis=1)
# y = np.exp(data["108"])
# train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

scaler = MinMaxScaler()
data = pd.read_csv("gan_train.csv")
data[["114"]] = scaler.fit_transform(data[["114"]])
X = data.drop(["114"],axis=1)
# y = np.log(data["117"]+1)
y = data["114"]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train = pd.concat([train_x,train_y],axis=1)
test = pd.concat([test_x,test_y],axis=1)
train.to_csv("conclude/train.csv")
test.to_csv("conclude/test.csv")

#svr
svr = svr()
svr.fit(train_x,train_y)
print(svr.best_estimator_())
svr_reg = svr.best_estimator_()
joblib.dump(svr_reg, 'model/svr.pkl')
print(mean_squared_error(train_y,svr_reg.predict(train_x)))
print(r2_score(train_y,svr_reg.predict(train_x)))
print(mean_squared_error(test_y,svr_reg.predict(test_x)))
print(r2_score(test_y,svr_reg.predict(test_x)))

#RF
rf = RF()
rf.fit(train_x,train_y)
print(rf.best_estimator_())
rf_reg = rf.best_estimator_()
joblib.dump(rf_reg, 'model/rf.pkl')
print(mean_squared_error(train_y,rf_reg.predict(train_x)))
print(r2_score(train_y,rf_reg.predict(train_x)))
print(mean_squared_error(test_y,rf_reg.predict(test_x)))
print(r2_score(test_y,rf_reg.predict(test_x)))

#xgbr
xgbr = XGBR()
xgbr.fit(train_x,train_y)
print(xgbr.best_estimator_())
xgbr_reg = xgbr.best_estimator_()
xgbr_reg.save_model('model/xgbr.bin')
print(mean_squared_error(train_y,xgbr_reg.predict(train_x)))
print(r2_score(train_y,xgbr_reg.predict(train_x)))
print(mean_squared_error(test_y,xgbr_reg.predict(test_x)))
print(r2_score(test_y, xgbr_reg.predict(test_x)))

#net
train_x  = train_x.values.tolist()
test_x = test_x.values.tolist()
device = "cuda" if torch.cuda.is_available() else "cpu"

x_train = torch.tensor(train_x)
# x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
x_train = x_train.to(device)
x_train = x_train.to(torch.float)
y_train = torch.tensor(np.array(train_y))
# y_train = y_train.reshape(y_train.shape[0],1)
y_train = y_train.to(device)
y_train = y_train.to(torch.float)

x_test = torch.tensor(test_x) 
# x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
x_test = x_test.to(device)
x_test = x_test.to(torch.float)
y_test = torch.tensor(np.array(test_y))
# y_test = y_test.reshape(y_test.shape[0],1)
y_test = y_test.to(device)
y_test = y_test.to(torch.float)

model =  Net(114).to(device)
k_fold(model,x_train,y_train,num_epochs = 128,learning_rate = 0.001,batch_size = 128)
print(mean_squared_error(y_train.detach().cpu().numpy(),model(x_train).detach().cpu().numpy()))
print(r2_score(y_train.detach().cpu().numpy(), model(x_train).detach().cpu().numpy()))
print(mean_squared_error(y_test.detach().cpu().numpy(),model(x_test).detach().cpu().numpy()))
print(r2_score(y_test.detach().cpu().numpy(), model(x_test).detach().cpu().numpy()))
torch.save(model, 'model/net.pth') 