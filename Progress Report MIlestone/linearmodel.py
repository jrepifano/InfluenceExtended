# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:38:57 2019

@author: epifanoj0

Linear model feature importance comparisons
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

x = np.vstack((x_train,x_test))
y = np.concatenate((y_train,y_test))

r = np.random.RandomState(seed=1234567890)

mi_score = mutual_info_classif(x,y)

clf = LogisticRegression(solver='lbfgs',random_state=r).fit(x_train,y_train)

print(clf.score(x_test,y_test))

coefs = clf.coef_

