import torch
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import r2_score
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

class svr:
    def __init__(self):
        self.param_grid={"C": [0.1,0.2,0.3, 1,2,3,10,20],
                                "gamma": [1, 0.1, 0.01,0.001]}
        self.svr = SVR(kernel='rbf')
        self.grid_svr = GridSearchCV(self.svr,
                    param_grid=self.param_grid,
                    cv=10)
    def fit(self,x,y):
        self.grid_svr.fit(x,y)
    def best_estimator_(self):
        return self.grid_svr.best_estimator_

class RF:
    def __init__(self):
        self.param_grid = {
                            'n_estimators': [5,10,20,50,70,100],
                            'max_depth':[3,5,7,9,10,20],
                            'max_features':[0.6,0.7,1]
                            }
        self.rf = RandomForestRegressor()
        self.grid_rf = GridSearchCV(self.rf,param_grid=self.param_grid,cv=10)
    def fit(self,x,y):
        self.grid_rf.fit(x,y)
    def best_estimator_(self):
        return self.grid_rf.best_estimator_

class XGBR:
    def __init__(self):
        self.param_grid = {
        # 'n_estimators': [5, 10, 20, 50, 70, 100, 200,300,400,500],
        'eta': [0.4],
        'n_estimators': [20],
        # 'max_depth': [1,2,3, 4, 5, 6, 7, 8],
        'max_depth': [6],
        # 'max_delta_step': [1, 3, 5, 7],
        'max_delta_step': [6],
        'subsample': [0.6],
        # 'gamma': [0.0001],
        'colsample_bytree': [0.9],
        # 'colsample_bylevel': [0.3,0.5,0.7,0.8,1],
        # 'colsample_bynode': [0.3,0.5,0.7,0.8,1]
        }
        self.xgbr = XGBRegressor()
        self.grid_xgbr = GridSearchCV(self.xgbr,param_grid=self.param_grid,cv=10)
    def fit(self,x,y):
        self.grid_xgbr.fit(x,y)
    def best_estimator_(self):
        return self.grid_xgbr.best_estimator_

class Net(nn.Module):
    def __init__(self,input):
        super(Net,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input, 128),
            # nn.LeakyReLU(),
            nn.Linear(128, 256),
            # nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self,input):
        out = self.model(input)
        out = out.squeeze(-1)
        return out

def get_kfold_data(i,x,y):
    fold_size = len(x)//10;
    val_start = i * fold_size
    if i != 9:
        val_end = (i+1) * fold_size
        x_valid , y_valid = x[val_start:val_end],y[val_start:val_end]
        x_train = torch.cat((x[0:val_start],x[val_end:]),dim=0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
    else:
        x_valid, y_valid = x[val_start:], y[val_start:]
        x_train = x[0:val_start]
        y_train = y[0:val_start]
    return x_train, y_train, x_valid, y_valid

def train(model,x_train,y_train,x_val,y_val,BATCH_SIZE,learning_rate,TOTAL_EPOCHS):
    train_loader = DataLoader(TensorDataset(x_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), BATCH_SIZE, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_r2 = []
    val_r2 = []

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        for i,(x,y) in enumerate(train_loader):
            predict = model(x)
            loss = loss_fn(predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            # if (i+1)%10 == 0:
            #     print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
            #                                                         i + 1, len(x_train) // BATCH_SIZE,
            #                                                         loss.item()))
        r2 = r2_score(y_train.cpu().numpy(), model(x_train).cpu().detach().numpy())
        # print('Epoch: {}, Loss: {:.5f}, Training set r2:({:.3f})'.format(
        #     epoch + 1, loss.item(), r2))
        train_r2.append(r2)

        model.eval()
        val_loss = loss_fn(model(x_val),y_val)
        val_losses.append(val_loss.item())
        r2 = r2_score(y_val.cpu().numpy(), model(x_val).cpu().detach().numpy())
        val_r2.append(r2)

    return train_losses,val_losses,train_r2,val_r2

def k_fold(net,x_train,y_train,num_epochs = 3,learning_rate = 0.001,batch_size = 16):
    train_loss_sum=0.0
    valid_loss_sum = 0.0
    train_r2_sum= 0.0
    valid_r2_sum = 0.0
    for i in range(10):
        print('*'*25,'第', i + 1,'折','*'*25)
        data = get_kfold_data(i,x_train,y_train)
        train_loss, val_loss, train_r2, val_r2 = train(net, *data, batch_size, learning_rate, num_epochs)

        print('train_loss:{:.5f}, train_r2:{:.3f}'.format(train_loss[-1], train_r2[-1]))
        print('valid loss:{:.5f}, valid_r2:{:.3f}\n'.format(val_loss[-1], val_r2[-1]))

        train_loss_sum += train_loss[-1]
        valid_loss_sum += val_loss[-1]
        train_r2_sum += train_r2[-1]
        valid_r2_sum += val_r2[-1]

    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

    print('average train loss:{:.4f}, average train r2:{:.3f}'.format(train_loss_sum / 10, train_r2_sum / 10))
    print('average valid loss:{:.4f}, average valid r2:{:.3f}'.format(valid_loss_sum / 10, valid_r2_sum / 10))

    return