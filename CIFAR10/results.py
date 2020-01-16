# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:43:43 2020

@author: Jake
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

eqn_2_test_set = np.load('results/eqn_2-test_set.npy')
eqn_5_test_set = np.load('results/eqn_5-test_set.npy')

train_images = np.load('results/train_images.npy')
train_labels = np.load('results/train_labels.npy')

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

channel_1_scaler = MinMaxScaler()
channel_2_scaler = MinMaxScaler()
channel_3_scaler = MinMaxScaler()

channel_1 = eqn_5_test_set[:,0,:,:].reshape(-1,1024).transpose()
channel_2 = eqn_5_test_set[:,1,:,:].reshape(-1,1024).transpose()
channel_3 = eqn_5_test_set[:,2,:,:].reshape(-1,1024).transpose()

channel_1 = channel_1_scaler.fit_transform(channel_1).transpose().reshape(-1,1,32,32)
channel_2 = channel_2_scaler.fit_transform(channel_2).transpose().reshape(-1,1,32,32)
channel_3 = channel_3_scaler.fit_transform(channel_3).transpose().reshape(-1,1,32,32)

eqn_5_scaled = np.hstack((channel_1,channel_2,channel_3))

imshow(train_images[3126])
plt.imshow(np.transpose(eqn_5_scaled[3126], (1, 2, 0)))

cnn_rank = np.argsort(eqn_2_test_set[:,0])