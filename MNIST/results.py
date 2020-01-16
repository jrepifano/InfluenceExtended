# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:51:51 2020

@author: Jake
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

cnn_test_set_eqn_2 = np.load('results/cnn-eqn_2-test_set.npy')
mlp_test_set_eqn_2 = np.load('results/mlp-eqn_2-test_set.npy')

cnn_test_set_eqn_5 = np.load('results/cnn-eqn_5-test_set.npy')
mlp_test_set_eqn_5 = np.load('results/mlp-eqn_5-test_set.npy')

train_images = np.load('results/train_images.npy').reshape(-1,28,28)
train_labels = np.load('results/train_labels.npy')

# test_images = np.load('results/test_images.npy').reshape(-1,28,28)
# test_labels = np.load('results/test_labels.npy')

mlp_scaler = MinMaxScaler()
cnn_scaler = MinMaxScaler()

cnn_test_set_eqn_5 = cnn_test_set_eqn_5.reshape(-1,784).transpose()
cnn_test_set_eqn_5 = cnn_scaler.fit_transform(cnn_test_set_eqn_5).transpose().reshape(-1,28,28)

mlp_test_set_eqn_5 = mlp_test_set_eqn_5.reshape(-1,784).transpose()
mlp_test_set_eqn_5 = mlp_scaler.fit_transform(mlp_test_set_eqn_5).transpose().reshape(-1,28,28)

cnn_avg = np.mean(cnn_test_set_eqn_5,axis=0)
mlp_avg = np.mean(mlp_test_set_eqn_5,axis=0)

plt.imshow(cnn_avg)
plt.show()
plt.imshow(mlp_avg)
plt.show()

i = 50
plt.imshow(cnn_test_set_eqn_5[i])
plt.show()
plt.imshow(mlp_test_set_eqn_5[i])
plt.show()
plt.imshow(train_images[i])
plt.show()

# unique, counts = np.unique(test_labels,return_counts=True)

# cnn_rank = np.argsort(cnn_test_set_eqn_2[:,0])
# mlp_rank = np.argsort(mlp_test_set_eqn_2[:,0])

# top_10_cnn = train_labels[cnn_rank[:10]]
# top_10_mlp = train_labels[mlp_rank[:10]]