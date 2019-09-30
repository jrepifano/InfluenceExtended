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
from torch.utils import data
from sklearn.model_selection import train_test_split
import influence

class dataset(data.Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,i):
        x_i = torch.tensor(self.X[i,:])
        y_i = torch.tensor(self.Y[i])
        y_i = y_i.float()
        x_i = x_i.float()
        return x_i, y_i
    
X1 = np.load('X1.npy')
X2 = np.load('X2.npy')
X = (np.vstack([X1,X2]))
y = np.hstack([np.zeros(50),np.ones(50)]).reshape((-1,1))

num_classes = 2
num_epochs = 1000
batch_size = round(len(X)*1)
learning_rate = 0.001
input_size = X.shape[1]

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=1)

train_dataset = dataset(X_train, Y_train)
test_dataset = dataset(X_test, Y_test)

train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
test_loader = data.DataLoader(test_dataset,batch_size=1,shuffle=False)

model = nn.Sequential(
        nn.Linear(input_size,1),
        nn.Sigmoid())

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    correct = 0
    for j, (in_data, labels) in enumerate(train_loader):  
        
        # Forward pass
        outputs = model(in_data)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    #Accuracy
    outputs = (outputs>0.5).float()
    correct = (outputs == labels).float().sum()
    print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss.data, correct/outputs.shape[0]))


#Influence start
outputs = model(in_data)
loss = criterion(outputs,labels)
J = torch.autograd.grad(loss,model.parameters(),create_graph=True)
H = torch.autograd.grad(torch.sum(J[0]),model.parameters())

    
    
#test_index = 0
#
#test_x = torch.tensor([X_test[test_index,:]]).float()
#test_x.requires_grad=True
#test_y = torch.tensor([Y_test[test_index]]).long()
#
#test_output = model(test_x)
#test_loss = criterion(test_output,test_y)
#
#
##define HVP subroutine - layer by layer influence?
#grads = torch.autograd.grad(test_loss,model.parameters(),create_graph=True) 

