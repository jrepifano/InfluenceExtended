# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 01:42:40 2020

@author: Jake
"""

import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.linear_1 = torch.nn.Linear(25088, 100)
#         # self.linear_2 = torch.nn.Linear(100,10)
#         self.selu = torch.nn.SELU()
#         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv_1(x)
#         x = self.selu(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.linear_1(x)
#         # x = self.selu(x)
#         # x = self.linear_2(x)
#         pred = self.softmax(x)

#         return pred

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=25, kernel_size=5, stride=2, padding=1)
        self.linear_1 = torch.nn.Linear(4225, 10)
        # self.linear_2 = torch.nn.Linear(100,10)
        self.selu = torch.nn.SELU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.selu(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_1(x)
        # x = self.selu(x)
        # x = self.linear_2(x)
        pred = self.softmax(x)

        return pred
    
model = torch.load('cnn_2.pt')

mnist_trainset = torch.load('mnist_trainset.pt')
mnist_testset = torch.load('mnist_testset.pt')
mnist_valset = torch.load('mnist_valset.pt')

train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', torch.cuda.get_device_name(torch.cuda.current_device()))

criterion = torch.nn.CrossEntropyLoss()
model.to(device)

total= 0
for itr, (image, label) in enumerate(test_dataloader):

        image = image.to(device)
        label = label.to(device)
        
        pred = model(image)

        pred = torch.nn.functional.softmax(pred, dim=1)
        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1

accuracy = total / len(mnist_testset)
print('Test-Set Accuracy: '+str(accuracy))