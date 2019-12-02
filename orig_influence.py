#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:58:30 2019

@author: epifanoj

Author vs me testing script
"""

import numpy as np
import tensorflow as tf
import dataset
import binaryLogisticRegressionWithLBFGS
from sklearn.linear_model import LogisticRegression
from tensorflow.contrib.learn.python.learn.datasets import base

tf.reset_default_graph()                #reset graph so that I can re-run in the same python console

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

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

#average loss over all of the test samples
test_idx = np.arange(20)

influence_0 = orig_model.get_influence_on_test_loss(test_idx,range(len(x_train)), approx_type='lissa')
iPertLoss = orig_model.get_grad_of_influence_wrt_input(range(len(x_train)), test_idx, approx_type='lissa')