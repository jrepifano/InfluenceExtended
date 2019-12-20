#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:28:21 2019

@author: epifanoj

setup dnns for extended influence functions
"""
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()

        self.lin1 = nn.Linear(4,2)
        self.lin2 = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()

    def forward(self,x):

        out1 = self.selu(self.lin1(x))
        out2 = self.sigmoid(self.lin2(out1))

        return out2

# iris = load_iris()
# X = iris.data[:100,:]
# y = iris.target[:100]

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# input_size = X.shape[1]

# r = np.random.RandomState(seed=1234567890)

# X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=r)

X_train = np.load('x_train.npy')
Y_train = np.load('y_train.npy')
X_test = np.load('x_test.npy')
Y_test = np.load('y_test.npy')

train_X = Variable(torch.Tensor(X_train).float())
test_X = Variable(torch.Tensor(X_test).float())
train_y = Variable(torch.Tensor(Y_train).float()).view(-1,1)
test_y = Variable(torch.Tensor(Y_test).float()).view(-1,1)

# Loss and optimizer

model = net()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 10000
# Train the model
for epoch in range(num_epochs):
    correct = 0

    optimizer.zero_grad()
    # Forward pass
    outputs = model.forward(train_X)
    loss = criterion(outputs, train_y)
    # Backward and optimize
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch==num_epochs-1:
        outputs = (outputs>0.5).float()
        correct = (outputs == train_y).float().sum()
        predict_out = model(test_X)
        predict_y = torch.round(predict_out)
        print("Epoch {}/{}, Loss: {:.3f}, Training Acc: {:.3f}, Test Acc: {:.3f}".format(
            epoch+1,num_epochs, loss.data, correct/outputs.shape[0], accuracy_score(test_y.data, predict_y.data)))

torch.save(model,'selu_model.pt')
