import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from cgan import Generator
import numpy as np
import pandas as pd
import argparse
from fun import filter,cal_similarity,get_des_sim
from rdkit import DataStructs
from rdkit.DataStructs import cDataStructs
from model import Net
from sklearn.metrics import mean_squared_error,r2_score
# from cgan import Generator,Discriminator

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=1, help="latent code")
parser.add_argument("--target", type=int, default=10, help="target for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.layer = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        a = F.leaky_relu(self.layer(x))
        a = F.softmax(a.view(a.size(0), -1), dim=1).view_as(a)
        x = x * a
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(in_channels, out_channels)
        self.layer2 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        y = F.leaky_relu(self.layer1(x))
        x = self.layer2(x)
        return F.leaky_relu(x + y)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.target

        # self.target_emb = nn.Embedding(opt.target,opt.target)
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128))

        self.main_block = nn.Sequential(
            # nn.BatchNorm1d(256, 0.8),
            # nn.LeakyReLU(),
            # nn.Linear(128,128),
            Attention(128, 128),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            ResidualBlock(128, 256),
            # nn.Linear(128,256),
            # nn.BatchNorm1d(512, 0.8),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            # ResidualBlock(256, 256),
            # nn.Linear(256, 512),
            nn.Linear(256, 100),
            # nn.Sigmoid()
            nn.ReLU()
        )

    def forward(self, noise, target):
        gen_input = torch.cat((noise, target), -1)
        out = self.l1(gen_input)
        out = self.main_block(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # self.target_embedding = nn.Embedding(opt.target, opt.target)
        # self.attention =
        self.model = nn.Sequential(
            nn.Linear((opt.target + 100), 128),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            Attention(128, 128),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Linear(128, 256),
            ResidualBlock(128, 256),
            # nn.LeakyReLU(),
            # nn.Linear(256, 256),
            # nn.LeakyReLU(),
            # # nn.ReLU(),
            # nn.Linear(256, 512),
            # ResidualBlock(256, 512),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, vec, target):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((vec, target), -1)
        validity = self.model(d_in)
        return validity
    
gen = Generator()
gen.load_state_dict(torch.load("model/gen1.pt"))

gan_test = pd.read_csv("conclude/test.csv")
gan_test = gan_test.drop("Unnamed: 0",axis=1)
fea = []
for i in range(100,115):
    fea.append(str(i))
x_test = gan_test.drop(fea,axis=1)
y_test = gan_test[['101','102','103','106','107','109','110','111','113','114']]

x_test = torch.tensor(x_test.values)
y_test = torch.tensor(y_test.values)
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

# z = Variable(LongTensor(np.random.uniform(0, 10,  100)))
# gen_target = Variable(FloatTensor(y_test[0].type(FloatTensor)))
# print(gen(z,gen_target))
net = torch.load("model/net1.pth")
net.eval()
gen.eval()
sim_sum = 0
netx=[]
for i in range(len(y_test)):
    z = Variable(LongTensor(np.random.uniform(0, 10,  100)))
    gen_target = Variable(FloatTensor(y_test[i].type(FloatTensor)))
    vec1 = gen(z,gen_target)
    print(vec1)
    prex=torch.cat([vec1,gen_target[:-1]],0)
    netx.append(net(prex))
    vec2 = Variable(FloatTensor(x_test[i].type(FloatTensor)))
    sim_sum += cal_similarity(vec1,vec2)
netx = torch.tensor(netx)
print(r2_score(netx.detach().cpu().numpy(),gan_test["114"]))
print(sim_sum/len(y_test))

des = filter()
m = list(np.random.uniform(0, 200,  3))
m.extend([0.5,4,30,90,8,3,20])
get_des_sim(gen,m,des)

# # #0,3,0.5,0,1,30,70,8,20
# # des1 = get_des_sim(gen,[0,3,0.5,0,1,30,70,8,20],des)
# # #0,4,0.5,0,1,30,70,8,20
# # des2 = get_des_sim(gen,[0,4,0.5,0,1,30,70,8,20],des)
# # #0,5,0.5,0,1,30,70,8,20
# # des3 = get_des_sim(gen,[0,5,0.5,0,1,30,70,8,20],des)
# # #0,7,0.5,0,1,30,70,8,20
# # des4 = get_des_sim(gen,[0,7,0.5,0,1,30,70,8,20],des)
# # #0,10,0.5,0,1,30,70,8,20
# # des5 = get_des_sim(gen,[0,10,0.5,0,1,30,70,8,20],des)

# ratio_map = {30:20,40:15,50:12}
# # col = ['pKa','E+','E-','Cl-','ratio','tem','time','mg','DES','ratio_DES']
# col = ['Cl-','ratio','tem','time','mg','DES','ratio_DES']
# ex = pd.DataFrame(columns=col)
# # for pka in [3,4,5,7,10]:
# #     for ep in [0.5,0.55]:
# #         for en in [0,-0.2]:
# #             for ratio in [30,40,50]:
# #                 for tem in [70,90]:
# #                     condition = [0,pka,ep,en,1,ratio,tem,8,ratio_map[ratio]]
# #                     item = get_des_sim(gen,condition,des).iloc[[0]]
# #                     condition.append(item["DES"].values[0])
# #                     condition.append(item["HBA:HBD\n比例"].values[0])
# #                     condition.pop(0)
# #                     ex.loc[len(ex)]=condition
#                     # line = {}
#                     # for i in range(len(col)):
#                     #     line[col[i]] = condition[i+1]
#                     # print(line)
#                     # ex = ex.append(line,ignore_index=True)
# for ratio in [30,40,50]:
#     for tem in [70,90]:
#         condition = [0,1,ratio,tem,8,ratio_map[ratio]]
#         item = get_des_sim(gen,condition,des).iloc[[0]]
#         condition.append(item["DES"].values[0])
#         condition.append(item["HBA:HBD比例"].values[0])
#         condition.pop(0)
#         ex.loc[len(ex)]=condition
# print(ex)
# ex.to_excel("sim1/ex.xlsx",index=False)