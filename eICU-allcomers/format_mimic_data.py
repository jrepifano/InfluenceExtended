# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:10:32 2020

@author: Jake
"""

import pandas as pd
import numpy as np
import torch
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc, precision_recall_curve, average_precision_score

df = pd.read_csv('data/mimic.csv')
ds = pd.read_csv('data/nopf.csv')

x_eicu = ds.iloc[:,2:-1].to_numpy()
y_eicu = np.load('data/y.npy')

x_mimic = df.iloc[:,1:-1]
x_mimic.replace({'M':0,'F':1},inplace=True)
x_mimic = x_mimic.to_numpy()
y_mimic = df.iloc[:,-1].to_numpy()

imputer = IterativeImputer(random_state=1234567890,verbose=2)
x_eicu = imputer.fit_transform(x_eicu)
x_mimic = imputer.transform(x_mimic)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(20, 12)
        self.linear_2 = torch.nn.Linear(12,188)
        self.linear_3 = torch.nn.Linear(188,2)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.selu(x)
        x = self.linear_2(x)
        x = self.selu(x)
        x = self.linear_3(x)
        pred = self.softmax(x)
        return pred

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(torch.cuda.is_available()):
    print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

mlp_probs = np.array([])
test_labels = np.array([])
test_data = np.array([])

no_epochs = 503

scaler = StandardScaler()
x_train = scaler.fit_transform(x_eicu)
y_train = y_eicu

x_test = scaler.transform(x_mimic)
y_test = y_mimic

x_train,y_train = SMOTE().fit_resample(x_train,y_train)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
   
train_loss = []
val_loss = []

model = Model()

print_iter = 100

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5828, momentum=0.8222,weight_decay=0.001757, nesterov=True)

if device:
        model.to(device)
        print('Moved to GPU')
    
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

mlp_probs = np.vstack((mlp_probs,pred_test.detach().cpu().numpy())) if mlp_probs.size else pred_test.detach().cpu().numpy()
test_labels = np.hstack((test_labels,y_test)) if test_labels.size else y_test
test_data = np.vstack((test_data,x_test)) if test_data.size else x_test

fpr, tpr, thresholds = roc_curve(test_labels, mlp_probs[:,1], pos_label=1)
roc_auc=auc(fpr,tpr)
print(roc_auc)