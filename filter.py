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

train = pd.read_csv("conclude/train.csv")
test = pd.read_csv("conclude/test.csv")
train = train.drop("Unnamed: 0",axis=1)
test = test.drop("Unnamed: 0",axis=1)

# #无结构
# fea = []
# for i in range(100,115):
#     fea.append(str(i))
# train = train[fea]
# test = test[fea]

# # 第一层
# train = train.drop(["108","112","104","105","100"],axis=1)
# test = test.drop(["108","112","104","105","100"],axis=1)
# #第二层
# train = train.drop(["108","112","106","104","100","113","111"],axis=1)
# test = test.drop(["108","112","106","104","100","113","111"],axis=1)
# #第三层
# train = train.drop(["108","112","106","104","100","113","111","102"],axis=1)
# test = test.drop(["108","112","106","104","100","113","111","102"],axis=1)
#第四层
train = train.drop(["108","112","106","104","100","113","111","102","103","110","101"],axis=1)
test = test.drop(["108","112","106","104","100","113","111","102","103","110","101"],axis=1)

train_x = train.drop("114",axis=1)
train_y = train["114"]
test_x = test.drop("114",axis=1)
test_y = test["114"]

# #svr
# svr = svr()
# svr.fit(train_x,train_y)
# print(svr.best_estimator_())
# svr_reg = svr.best_estimator_()
# joblib.dump(svr_reg, 'model/svr0.pkl')
# print(mean_squared_error(train_y,svr_reg.predict(train_x)))
# print(r2_score(train_y,svr_reg.predict(train_x)))
# print(mean_squared_error(test_y,svr_reg.predict(test_x)))
# print(r2_score(test_y,svr_reg.predict(test_x)))

# #svr
# svr = svr()
# svr.fit(train_x,train_y)
# print(svr.best_estimator_())
# svr_reg = svr.best_estimator_()
# joblib.dump(svr_reg, 'model/svr.pkl')
# print(mean_squared_error(train_y,svr_reg.predict(train_x)))
# print(r2_score(train_y,svr_reg.predict(train_x)))
# print(mean_squared_error(test_y,svr_reg.predict(test_x)))
# print(r2_score(test_y,svr_reg.predict(test_x)))

# #svr
# svr = svr()
# svr.fit(train_x,train_y)
# print(svr.best_estimator_())
# svr_reg = svr.best_estimator_()
# joblib.dump(svr_reg, 'model/svr1.pkl')
# print(mean_squared_error(train_y,svr_reg.predict(train_x)))
# print(r2_score(train_y,svr_reg.predict(train_x)))
# print(mean_squared_error(test_y,svr_reg.predict(test_x)))
# print(r2_score(test_y,svr_reg.predict(test_x)))

# # svr
# svr = svr()
# svr.fit(train_x,train_y)
# print(svr.best_estimator_())
# svr_reg = svr.best_estimator_()
# joblib.dump(svr_reg, 'model/svr2.pkl')
# print(mean_squared_error(train_y,svr_reg.predict(train_x)))
# print(r2_score(train_y,svr_reg.predict(train_x)))
# print(mean_squared_error(test_y,svr_reg.predict(test_x)))
# print(r2_score(test_y,svr_reg.predict(test_x)))

# # svr
# svr = svr()
# svr.fit(train_x,train_y)
# print(svr.best_estimator_())
# svr_reg = svr.best_estimator_()
# joblib.dump(svr_reg, 'model/svr3.pkl')
# print(mean_squared_error(train_y,svr_reg.predict(train_x)))
# print(r2_score(train_y,svr_reg.predict(train_x)))
# print(mean_squared_error(test_y,svr_reg.predict(test_x)))
# print(r2_score(test_y,svr_reg.predict(test_x)))

# # svr
# svr = svr()
# svr.fit(train_x,train_y)
# print(svr.best_estimator_())
# svr_reg = svr.best_estimator_()
# joblib.dump(svr_reg, 'model/svr4.pkl')
# print(mean_squared_error(train_y,svr_reg.predict(train_x)))
# print(r2_score(train_y,svr_reg.predict(train_x)))
# print(mean_squared_error(test_y,svr_reg.predict(test_x)))
# print(r2_score(test_y,svr_reg.predict(test_x)))

# #RF
# rf = RF()
# rf.fit(train_x,train_y)
# print(rf.best_estimator_())
# rf_reg = rf.best_estimator_()
# joblib.dump(rf_reg, 'model/rf0.pkl')
# print(mean_squared_error(train_y,rf_reg.predict(train_x)))
# print(r2_score(train_y,rf_reg.predict(train_x)))
# print(mean_squared_error(test_y,rf_reg.predict(test_x)))
# print(r2_score(test_y,rf_reg.predict(test_x)))

# #RF
# rf = RF()
# rf.fit(train_x,train_y)
# print(rf.best_estimator_())
# rf_reg = rf.best_estimator_()
# joblib.dump(rf_reg, 'model/rf.pkl')
# print(mean_squared_error(train_y,rf_reg.predict(train_x)))
# print(r2_score(train_y,rf_reg.predict(train_x)))
# print(mean_squared_error(test_y,rf_reg.predict(test_x)))
# print(r2_score(test_y,rf_reg.predict(test_x)))

# #RF
# rf = RF()
# rf.fit(train_x,train_y)
# print(rf.best_estimator_())
# rf_reg = rf.best_estimator_()
# joblib.dump(rf_reg, 'model/rf1.pkl')
# print(mean_squared_error(train_y,rf_reg.predict(train_x)))
# print(r2_score(train_y,rf_reg.predict(train_x)))
# print(mean_squared_error(test_y,rf_reg.predict(test_x)))
# print(r2_score(test_y,rf_reg.predict(test_x)))

# #RF
# rf = RF()
# rf.fit(train_x,train_y)
# print(rf.best_estimator_())
# rf_reg = rf.best_estimator_()
# joblib.dump(rf_reg, 'model/rf2.pkl')
# print(mean_squared_error(train_y,rf_reg.predict(train_x)))
# print(r2_score(train_y,rf_reg.predict(train_x)))
# print(mean_squared_error(test_y,rf_reg.predict(test_x)))
# print(r2_score(test_y,rf_reg.predict(test_x)))

# #RF
# rf = RF()
# rf.fit(train_x,train_y)
# print(rf.best_estimator_())
# rf_reg = rf.best_estimator_()
# joblib.dump(rf_reg, 'model/rf3.pkl')
# print(mean_squared_error(train_y,rf_reg.predict(train_x)))
# print(r2_score(train_y,rf_reg.predict(train_x)))
# print(mean_squared_error(test_y,rf_reg.predict(test_x)))
# print(r2_score(test_y,rf_reg.predict(test_x)))

# #RF
# rf = RF()
# rf.fit(train_x,train_y)
# print(rf.best_estimator_())
# rf_reg = rf.best_estimator_()
# joblib.dump(rf_reg, 'model/rf4.pkl')
# print(mean_squared_error(train_y,rf_reg.predict(train_x)))
# print(r2_score(train_y,rf_reg.predict(train_x)))
# print(mean_squared_error(test_y,rf_reg.predict(test_x)))
# print(r2_score(test_y,rf_reg.predict(test_x)))

# #xgbr
# xgbr = XGBR()
# xgbr.fit(train_x,train_y)
# print(xgbr.best_estimator_())
# xgbr_reg = xgbr.best_estimator_()
# xgbr_reg.save_model('model/xgbr0.bin')
# print(mean_squared_error(train_y,xgbr_reg.predict(train_x)))
# print(r2_score(train_y,xgbr_reg.predict(train_x)))
# print(mean_squared_error(test_y,xgbr_reg.predict(test_x)))
# print(r2_score(test_y, xgbr_reg.predict(test_x)))

# #xgbr
# xgbr = XGBR()
# xgbr.fit(train_x,train_y)
# print(xgbr.best_estimator_())
# xgbr_reg = xgbr.best_estimator_()
# xgbr_reg.save_model('model/xgbr.bin')
# print(mean_squared_error(train_y,xgbr_reg.predict(train_x)))
# print(r2_score(train_y,xgbr_reg.predict(train_x)))
# print(mean_squared_error(test_y,xgbr_reg.predict(test_x)))
# print(r2_score(test_y, xgbr_reg.predict(test_x)))

# #xgbr
# xgbr = XGBR()
# xgbr.fit(train_x,train_y)
# print(xgbr.best_estimator_())
# xgbr_reg = xgbr.best_estimator_()
# xgbr_reg.save_model('model/xgbr1.bin')
# print(mean_squared_error(train_y,xgbr_reg.predict(train_x)))
# print(r2_score(train_y,xgbr_reg.predict(train_x)))
# print(mean_squared_error(test_y,xgbr_reg.predict(test_x)))
# print(r2_score(test_y, xgbr_reg.predict(test_x)))

# #xgbr
# xgbr = XGBR()
# xgbr.fit(train_x,train_y)
# print(xgbr.best_estimator_())
# xgbr_reg = xgbr.best_estimator_()
# xgbr_reg.save_model('model/xgbr2.bin')
# print(mean_squared_error(train_y,xgbr_reg.predict(train_x)))
# print(r2_score(train_y,xgbr_reg.predict(train_x)))
# print(mean_squared_error(test_y,xgbr_reg.predict(test_x)))
# print(r2_score(test_y, xgbr_reg.predict(test_x)))

# #xgbr
# xgbr = XGBR()
# xgbr.fit(train_x,train_y)
# print(xgbr.best_estimator_())
# xgbr_reg = xgbr.best_estimator_()
# xgbr_reg.save_model('model/xgbr3.bin')
# print(mean_squared_error(train_y,xgbr_reg.predict(train_x)))
# print(r2_score(train_y,xgbr_reg.predict(train_x)))
# print(mean_squared_error(test_y,xgbr_reg.predict(test_x)))
# print(r2_score(test_y, xgbr_reg.predict(test_x)))

#xgbr
xgbr = XGBR()
xgbr.fit(train_x,train_y)
print(xgbr.best_estimator_())
xgbr_reg = xgbr.best_estimator_()
xgbr_reg.save_model('model/xgbr4.bin')
print(mean_squared_error(train_y,xgbr_reg.predict(train_x)))
print(r2_score(train_y,xgbr_reg.predict(train_x)))
print(mean_squared_error(test_y,xgbr_reg.predict(test_x)))
print(r2_score(test_y, xgbr_reg.predict(test_x)))

# #net
# train_x  = train_x.values.tolist()
# test_x = test_x.values.tolist()
# device = "cuda" if torch.cuda.is_available() else "cpu"

# x_train = torch.tensor(train_x)
# # x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
# x_train = x_train.to(device)
# x_train = x_train.to(torch.float)
# train_y = [[i] for i in np.array(train_y)]
# y_train = torch.tensor(train_y)
# # y_train = torch.tensor(np.array(train_y))

# # y_train = y_train.reshape(y_train.shape[0],1)
# y_train = y_train.to(device)
# y_train = y_train.to(torch.float)

# x_test = torch.tensor(test_x)
# # x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
# x_test = x_test.to(device)
# x_test = x_test.to(torch.float)
# # y_test = torch.tensor(np.array(test_y))
# test_y = [[i] for i in np.array(test_y)]
# y_test = torch.tensor(test_y)
# # y_test = y_test.reshape(y_test.shape[0],1)
# y_test = y_test.to(device)
# y_test = y_test.to(torch.float)

# model =  Net(14).to(device)
# k_fold(model,x_train,y_train,num_epochs = 128,learning_rate = 0.001,batch_size = 128)
# model.eval()
# print(mean_squared_error(y_train.detach().cpu().numpy(),model(x_train).detach().cpu().numpy()))
# print(r2_score(y_train.detach().cpu().numpy(), model(x_train).detach().cpu().numpy()))
# print(mean_squared_error(y_test.detach().cpu().numpy(),model(x_test).detach().cpu().numpy()))
# print(r2_score(y_test.detach().cpu().numpy(), model(x_test).detach().cpu().numpy()))
# torch.save(model, 'model/net0.pth') 

# #net
# train_x  = train_x.values.tolist()
# test_x = test_x.values.tolist()
# device = "cuda" if torch.cuda.is_available() else "cpu"

# x_train = torch.tensor(train_x)
# # x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
# x_train = x_train.to(device)
# x_train = x_train.to(torch.float)
# y_train = torch.tensor(np.array(train_y))
# # y_train = y_train.reshape(y_train.shape[0],1)
# y_train = y_train.to(device)
# y_train = y_train.to(torch.float)

# x_test = torch.tensor(test_x)
# # x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
# x_test = x_test.to(device)
# x_test = x_test.to(torch.float)
# y_test = torch.tensor(np.array(test_y))
# # y_test = y_test.reshape(y_test.shape[0],1)
# y_test = y_test.to(device)
# y_test = y_test.to(torch.float)

# model =  Net(114).to(device)
# k_fold(model,x_train,y_train,num_epochs = 128,learning_rate = 0.001,batch_size = 128)
# print(mean_squared_error(y_train.detach().cpu().numpy(),model(x_train).detach().cpu().numpy()))
# print(r2_score(y_train.detach().cpu().numpy(), model(x_train).detach().cpu().numpy()))
# print(mean_squared_error(y_test.detach().cpu().numpy(),model(x_test).detach().cpu().numpy()))
# print(r2_score(y_test.detach().cpu().numpy(), model(x_test).detach().cpu().numpy()))
# torch.save(model, 'model/net.pth') 

# #net
# train_x  = train_x.values.tolist()
# test_x = test_x.values.tolist()
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# x_train = torch.tensor(train_x)
# # x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
# x_train = x_train.to(device)
# x_train = x_train.to(torch.float)
# train_y = [[i] for i in np.array(train_y)]
# y_train = torch.tensor(train_y)
# # y_train = torch.tensor(np.array(train_y))
#
# # y_train = y_train.reshape(y_train.shape[0],1)
# y_train = y_train.to(device)
# y_train = y_train.to(torch.float)
#
# x_test = torch.tensor(test_x)
# # x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
# x_test = x_test.to(device)
# x_test = x_test.to(torch.float)
# # y_test = torch.tensor(np.array(test_y))
# test_y = [[i] for i in np.array(test_y)]
# y_test = torch.tensor(test_y)
# # y_test = y_test.reshape(y_test.shape[0],1)
# y_test = y_test.to(device)
# y_test = y_test.to(torch.float)
#
# model = Net(109).to(device)
# k_fold(model,x_train,y_train,num_epochs = 128,learning_rate = 0.001,batch_size = 128)
# model.eval()
# print(mean_squared_error(y_train.detach().cpu().numpy(),model(x_train).detach().cpu().numpy()))
# print(r2_score(y_train.detach().cpu().numpy(), model(x_train).detach().cpu().numpy()))
# print(mean_squared_error(y_test.detach().cpu().numpy(),model(x_test).detach().cpu().numpy()))
# print(r2_score(y_test.detach().cpu().numpy(), model(x_test).detach().cpu().numpy()))
# torch.save(model, 'model/net1.pth')

# #net
# train_x  = train_x.values.tolist()
# test_x = test_x.values.tolist()
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# x_train = torch.tensor(train_x)
# # x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
# x_train = x_train.to(device)
# x_train = x_train.to(torch.float)
# y_train = torch.tensor(np.array(train_y))
# # y_train = y_train.reshape(y_train.shape[0],1)
# y_train = y_train.to(device)
# y_train = y_train.to(torch.float)
#
# x_test = torch.tensor(test_x)
# # x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
# x_test = x_test.to(device)
# x_test = x_test.to(torch.float)
# y_test = torch.tensor(np.array(test_y))
# # y_test = y_test.reshape(y_test.shape[0],1)
# y_test = y_test.to(device)
# y_test = y_test.to(torch.float)
#
# model = Net(107).to(device)
# k_fold(model,x_train,y_train,num_epochs = 128,learning_rate = 0.001,batch_size = 128)
# print(mean_squared_error(y_train.detach().cpu().numpy(),model(x_train).detach().cpu().numpy()))
# print(r2_score(y_train.detach().cpu().numpy(), model(x_train).detach().cpu().numpy()))
# print(mean_squared_error(y_test.detach().cpu().numpy(),model(x_test).detach().cpu().numpy()))
# print(r2_score(y_test.detach().cpu().numpy(), model(x_test).detach().cpu().numpy()))
# torch.save(model, 'model/net2.pth')

# #net
# train_x  = train_x.values.tolist()
# test_x = test_x.values.tolist()
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# x_train = torch.tensor(train_x)
# # x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1])
# x_train = x_train.to(device)
# x_train = x_train.to(torch.float)
# y_train = torch.tensor(np.array(train_y))
# # y_train = y_train.reshape(y_train.shape[0],1)
# y_train = y_train.to(device)
# y_train = y_train.to(torch.float)
#
# x_test = torch.tensor(test_x)
# # x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
# x_test = x_test.to(device)
# x_test = x_test.to(torch.float)
# y_test = torch.tensor(np.array(test_y))
# # y_test = y_test.reshape(y_test.shape[0],1)
# y_test = y_test.to(device)
# y_test = y_test.to(torch.float)
#
# model =  Net(106).to(device)
# k_fold(model,x_train,y_train,num_epochs = 128,learning_rate = 0.001,batch_size = 128)
# print(mean_squared_error(y_train.detach().cpu().numpy(),model(x_train).detach().cpu().numpy()))
# print(r2_score(y_train.detach().cpu().numpy(), model(x_train).detach().cpu().numpy()))
# print(mean_squared_error(y_test.detach().cpu().numpy(),model(x_test).detach().cpu().numpy()))
# print(r2_score(y_test.detach().cpu().numpy(), model(x_test).detach().cpu().numpy()))
# torch.save(model, 'model/net3.pth')

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

model =  Net(103).to(device)
k_fold(model,x_train,y_train,num_epochs = 128,learning_rate = 0.001,batch_size = 128)
print(mean_squared_error(y_train.detach().cpu().numpy(),model(x_train).detach().cpu().numpy()))
print(r2_score(y_train.detach().cpu().numpy(), model(x_train).detach().cpu().numpy()))
print(mean_squared_error(y_test.detach().cpu().numpy(),model(x_test).detach().cpu().numpy()))
print(r2_score(y_test.detach().cpu().numpy(), model(x_test).detach().cpu().numpy()))
torch.save(model, 'model/net4.pth')