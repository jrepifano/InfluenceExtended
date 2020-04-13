# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:45:08 2020

@author: Jake
"""

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve,auc, precision_recall_curve, average_precision_score
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--layer_1', help='layer 1 size', type=int, default=50, required=True)
    parser.add_argument(
        '--layer_2', help='layer 2 size', type=int, default=50, required=True)
    parser.add_argument(
        '--activation', help='Sigmoid or SELU activations', type=int, default=0, required=True)   
    parser.add_argument(
        '--lr', help='Learning Rate', type=float, default=0.01, required=True)
    parser.add_argument(
        '--no_epoch', help='Number of Epochs', type=int, default=1000, required=True)
    args = parser.parse_args()
    return args

class Model(torch.nn.Module):
    def __init__(self, layer_1,layer_2,activation):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(20, layer_1)
        self.linear_2 = torch.nn.Linear(layer_1,layer_2)
        self.linear_3 = torch.nn.Linear(layer_2,2)
        self.selu = torch.nn.SELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.activation = torch.nn.Sigmoid() if(activation == 0) else torch.nn.SELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.activation(x)
        x = self.linear_3(x)
        pred = self.softmax(x)
        return pred
    
wandb.init()
cmd_args = parse_args()

x_imputed = np.load('data/x_imputed.npy')
y = np.load('data/y.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(torch.cuda.is_available()):
    print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

no_epochs = cmd_args.no_epoch
print_iter = 100

x_train, x_test, y_train, y_test = train_test_split(x_imputed, y, test_size=0.33)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train,y_train = SMOTE().fit_resample(x_train,y_train)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)
   
train_loss = []
val_loss = []

model = Model(layer_1=cmd_args.layer_1,layer_2=cmd_args.layer_2,activation=cmd_args.activation)
# Magic
wandb.watch(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=cmd_args.lr, momentum=0.9,weight_decay=0.01, nesterov=True)

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

mlp_probs = np.array([])
test_labels = np.array([])
test_data = np.array([])

mlp_probs = np.vstack((mlp_probs,pred_test.detach().cpu().numpy())) if mlp_probs.size else pred_test.detach().cpu().numpy()
test_labels = np.hstack((test_labels,y_test)) if test_labels.size else y_test

fpr, tpr, thresholds = roc_curve(test_labels, mlp_probs[:,1], pos_label=1)
roc_auc=auc(fpr,tpr)
wandb.log({'roc_auc':roc_auc})