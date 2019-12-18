# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:22:28 2019

@author: Jake
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import binaryLogisticRegressionWithLBFGS
from sklearn.linear_model import LogisticRegression
from tensorflow.contrib.learn.python.learn.datasets import base

# x1 = np.random.multivariate_normal([2,2], [[1,0],[0,1]],100)
# x2 = np.random.multivariate_normal([7,2], [[1,0],[0,1]],100)
tf.reset_default_graph()                #reset graph so that I can re-run in the same python console


x1 = np.load('gaussian_1.npy')
x2 = np.load('gaussian_2.npy')

plt.scatter(x1[:,0],x1[:,1])
plt.scatter(x2[:,0],x2[:,1])
plt.show()

x_train = np.vstack((x1,x2))
y_train = np.hstack((np.zeros(100),np.ones(100)))


x3 = np.random.multivariate_normal([2,2], [[1,0],[0,1]],25)
x4 = np.random.multivariate_normal([7,2], [[1,0],[0,1]],25)

x_test = np.vstack((x3,x4))
y_test = np.hstack((np.zeros(25),np.ones(25)))

lr_train = dataset.DataSet(x_train, np.array((y_train + 1) / 2, dtype=int))
lr_validation = None
lr_test = dataset.DataSet(x_test, np.array((y_test + 1) / 2, dtype=int))
lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)
                                        #Put dataset into dataset object as definted by authors

num_classes = 2                         #Use default parameters for author class
input_dim = x_train.shape[1]
weight_decay = 0.01
batch_size = 10
initial_learning_rate = 0.01 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 50000
use_bias = True

                                        #Define author class with parameters
orig_model = binaryLogisticRegressionWithLBFGS.BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=lr_data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='test')

C = 1.0 / (80 * weight_decay) 
clf = LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False, 
            solver='lbfgs',
            warm_start=True, #True
            max_iter=max_lbfgs_iter).fit(x_train,y_train)

params = clf.coef_
w = np.reshape(clf.coef_.T, -1)         #Flatten weights
params_feed_dict = {}                   #Next 2 lines feed weights into model
params_feed_dict[orig_model.W_placeholder] = w
orig_model.sess.run(orig_model.set_params_op, feed_dict=params_feed_dict)

test_idx = np.arange(50)

influence_0 = orig_model.get_influence_on_test_loss(test_idx,range(len(x_train)), approx_type='lissa')
eqn_2 = influence_0
iPertLoss = orig_model.get_grad_of_influence_wrt_input(range(len(x_train)), test_idx, approx_type='lissa')
eqn_5 = iPertLoss

np.save('gaussian_test_set_eqn_2.npy',eqn_2)
np.save('gaussian_test_set_eqn_5.npy',eqn_5)