#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:33:20 2020

@author: epifanoj
"""

import numpy as np
import torch
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc, precision_recall_curve, average_precision_score

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(28, 184)
        self.linear_2 = torch.nn.Linear(184,192)
        self.linear_3 = torch.nn.Linear(192,2)
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
    
r = np.random.RandomState(seed=1234567890)

x_imputed = np.load('data/x_imputed.npy')
y = np.load('data/y.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(torch.cuda.is_available()):
    print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

mlp_probs = np.array([])
test_labels = np.array([])
test_data = np.array([])

no_epochs = 810

kf = KFold(n_splits=5, shuffle=True, random_state=r)

print_iter = 100

for train_index, test_index in kf.split(x_imputed,y):

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_imputed[train_index])
    y_train = y[train_index]
    
    x_test = scaler.transform(x_imputed[test_index])
    y_test = y[test_index]
    
    x_train,y_train = SMOTE().fit_resample(x_train,y_train)
    
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
   
    train_loss = []
    val_loss = []
    
    model = Model()

    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.729, momentum=0.1902,weight_decay=0.009005, nesterov=True)
    
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
    
    # fig=plt.figure(figsize=(20, 10))
    # plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
    # plt.plot(np.arange(1, no_epochs+1), val_loss, label="Test loss")
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title("Loss Plots")
    # plt.legend(loc='upper right')
    # plt.show()


fpr, tpr, thresholds = roc_curve(test_labels, mlp_probs[:,1], pos_label=1)
roc_auc=auc(fpr,tpr)
print(roc_auc)

np.save('results/probs/smote/mlp_probs.npy',mlp_probs)
np.save('results/probs/smote/mlp_test_labels.npy',test_labels)
torch.save(model,'mlp.pt')