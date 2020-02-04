#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:00:26 2020

@author: epifanoj
"""

import torch
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(28, 100)
        self.linear_2 = torch.nn.Linear(100,100)
        self.linear_3 = torch.nn.Linear(100,2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        pred = self.softmax(x)

        return pred
    
r = np.random.RandomState(seed=1234567890)

x_imputed = np.load('data/x_imputed.npy')
y = np.load('data/y.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(torch.cuda.is_available()):
    torch.cuda.set_device(1)
    print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

model = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

mlp_probs = np.array([])
test_labels = np.array([])
test_data = np.array([])

no_epochs = 2000
train_loss = list()
val_loss = list()
best_val_loss = 1

sm = SMOTE()

x_train,x_test,y_train,y_test = train_test_split(x_imputed,y,test_size=0.3333,random_state=r)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
    
x_train, y_train = sm.fit_resample(x_train, y_train)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

if device:
    model.to(device)
    print('Moved to GPU')

print_iter = 500

for epoch in range(no_epochs):
    total_train_loss = 0
    total_val_loss = 0

    model.train()
    # training

    image = (x_train).float().to(device)
    label = y_train.long().to(device)

    optimizer.zero_grad()
    
    pred = model(image)

    loss = criterion(pred, label)
    total_train_loss += loss.item()

    loss.backward()
    optimizer.step()

    train_loss.append(total_train_loss)

    # validation
    model.eval()
    total = 0

    image = (x_test).float().to(device)
    label = y_test.long().to(device)
    
    pred_test = model(image)

    loss = criterion(pred_test, label)
    total_val_loss += loss.item()

    val_loss.append(total_val_loss)
    if(epoch % print_iter == 0 or epoch == no_epochs-1):
        print('\nEpoch: {}/{}, Train Loss: {:.8f}, Test Loss: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss))
    
fig=plt.figure(figsize=(20, 10))
plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
plt.plot(np.arange(1, no_epochs+1), val_loss, label="Test loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Loss Plots")
plt.legend(loc='upper right')
plt.show()

start_time = time.time()
e = shap.DeepExplainer(model, x_train.float().to(device))
shap_values = e.shap_values(x_test.float().to(device))
shap.summary_plot(shap_values, x_train.float().to(device), plot_type="bar")

np.save('results/mlp_shap_values_smote.npy',shap_values)

elapsed_time = time.time()-start_time
# np.save('results/mlp_shap_time',elapsed_time)