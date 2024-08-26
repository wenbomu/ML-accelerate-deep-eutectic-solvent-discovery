import argparse
import os
import numpy as np
import pandas as pd
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets
from torch.autograd import Variable
from fun import cal_similarity

import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
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
    def __init__(self,in_channels,out_channels):
        super(Attention,self).__init__()
        self.layer = nn.Linear(in_channels,out_channels)
    def forward(self,x):
        a = F.leaky_relu(self.layer(x))
        a = F.softmax(a.view(a.size(0),-1),dim=1).view_as(a)
        x = x*a
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(in_channels,out_channels)
        self.layer2 = nn.Linear(in_channels,out_channels)

    def forward(self, x):
        y = F.leaky_relu(self.layer1(x))
        x = self.layer2(x)
        return F.leaky_relu(x + y)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        input_dim = opt.latent_dim + opt.target
        
        # self.target_emb = nn.Embedding(opt.target,opt.target)
        self.l1 = nn.Sequential(nn.Linear(input_dim,128))

        self.main_block = nn.Sequential(
            # nn.BatchNorm1d(256, 0.8),
            # nn.LeakyReLU(),
            # nn.Linear(128,128),
            Attention(128,128),
            # nn.ReLU(),
            # nn.LeakyReLU(),
            ResidualBlock(128, 256),
            # nn.Linear(128,256),
            # nn.BatchNorm1d(512, 0.8),
            # nn.LeakyReLU(),
            # nn.ReLU(),
            # ResidualBlock(256, 256),
            # nn.Linear(256, 512),
            nn.Linear(256,100),
            # nn.Sigmoid()
            nn.ReLU()
        )

    def forward(self,noise,target):
        gen_input = torch.cat((noise,target),-1)
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
            Attention(128,128),
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
    
# Loss functions
# adversarial_loss = torch.nn.MSELoss()
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
# generator = generator.load_state_dict(torch.load("model/gen1.pt"))
discriminator = Discriminator()
# discriminator = discriminator.load_state_dict(torch.load("model/dic1.pt"))

batch_size = 128
gan_train = pd.read_csv("gan_train.csv")
# gan_train = pd.read_csv("conclude/train.csv")
# gan_test = pd.read_csv("conclude/test.csv")
# gan_train = gan_train.drop("Unnamed: 0",axis=1)
# gan_test = gan_test.drop("Unnamed: 0",axis=1)

# droplist = [str(i) for i in range(100,109)]
# x_train = gan_train.drop(droplist,axis=1)
# y_train = gan_train[droplist]


fea = []
for i in range(100,115):
    fea.append(str(i))

X = gan_train.drop(fea,axis=1)
y = gan_train[['101','102','103','106','107','109','110','111','113','114']]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# x_train = gan_train.drop(fea,axis=1)
# x_test = gan_test.drop(fea,axis=1)
# y_train = gan_train[['101','102','103','106','107','109','110','111','113','114']]
# y_test = gan_test[['101','102','103','106','107','109','110','111','113','114']]
# y_train = gan_train[['100','106','107','109','110','111','112','114']]
# y_test = gan_test[['100','106','107','109','110','111','112','114']]

# y_train['108'] = np.exp(y_train['108'])-1
x_train = torch.tensor(x_train.values)
y_train = torch.tensor(y_train.values)
# y_train = torch.tensor(np.array(y_train))

dataloader = torch.utils.data.DataLoader(
    TensorDataset(x_train, y_train),batch_size, shuffle=True
)

# Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)


FloatTensor = torch.FloatTensor 
LongTensor = torch.LongTensor

for epoch in range(opt.n_epochs):
    for i,(x,y) in enumerate(dataloader):

        batch_size = x.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_vec = Variable(x.type(FloatTensor))
        target = Variable(y.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(LongTensor(np.random.uniform(0, 10, (batch_size, opt.latent_dim))))
        gen_target = Variable(FloatTensor(np.random.uniform(0, 100, (batch_size, opt.target))))

        # Generate a batch of images
        gen_vec = generator(z, target)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_vec, target)
        g_loss = adversarial_loss(validity, valid)
        # g_loss = adversarial_loss(gen_vec, real_vec)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_vec, target)
        d_real_loss = adversarial_loss(validity_real, valid)
        d_real_loss.backward()

        # Loss for fake images
        # validity_fake = discriminator(gen_vec.detach(), gen_target)
        validity_fake = discriminator(gen_vec.detach(), target)
        d_fake_loss = adversarial_loss(validity_fake, fake)
        d_fake_loss.backward()

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss

        # d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
# l = [0,139.62,90.03,0,5363,1.23,0.597,0.092,0.505,4.79,1,50,30,0.83,0.186,1,5,1.17555831905736]
# gen_vec = []
torch.save(generator.state_dict(),"model/gen1.pt")
torch.save(discriminator.state_dict(),"model/dic1.pt")

x_test = torch.tensor(x_test.values)
y_test = torch.tensor(y_test.values)
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
# z = Variable(LongTensor(np.random.uniform(0, 10,  opt.latent_dim)))
# gen_target = Variable(FloatTensor(y_test[0].type(FloatTensor)))
# print(generator(z,gen_target))
generator.eval()
sim_sum = 0
for i in range(len(y_test)):
    z = Variable(LongTensor(np.random.uniform(0, 10,  100)))
    gen_target = Variable(FloatTensor(y_test[i].type(FloatTensor)))
    vec1 = generator(z,gen_target)
    vec2 = Variable(FloatTensor(x_test[i].type(FloatTensor)))
    sim_sum += cal_similarity(vec1,vec2)
print(sim_sum/len(y_test))
# for i,(x,y) in enumerate(dataloader):
#     batch_size = x.shape[0]
#     z = Variable(FloatTensor(np.random.uniform(0, 10, (batch_size, opt.latent_dim))))
#     gen_target = Variable(y.type(FloatTensor))
#     gen_vec = generator(z, gen_target)
#     # print(gen_vec[0])
#     for item in gen_vec:
#         cnt=0
#         for i in item:
#             if i>0.5:
#                 cnt+=1
#         print(cnt)