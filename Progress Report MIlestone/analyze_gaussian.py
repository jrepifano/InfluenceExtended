# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:16:07 2019

@author: Jake
"""

import numpy as np
import matplotlib.pyplot as plt

x1 = np.load('gaussian_1.npy')
x2 = np.load('gaussian_2.npy')
eqn_2 = np.load('results/gaussian_test_eqn_2.npy')
eqn_5 = np.load('results/gaussian_test_eqn_5.npy')

# sort_class_1 = eqn_2[:,0].argsort()[::-1]
# sort_class_2 = eqn_2[:,1].argsort()[::-1]


x_train = np.vstack((x1,x2))
y_train = np.hstack((np.zeros(100),np.ones(100)))

x_test = np.array([[2,2],[7,2]])
y_test = np.array([0,1])

plt.scatter(x1[:,0],x1[:,1],label='Class 0')
plt.scatter(x2[:,0],x2[:,1],label='Class 1')
plt.scatter(x_test[:,0],x_test[:,1],label='Center Distribution Points')
plt.legend(loc='lower right',prop={'size':8})
plt.show()

most_infl_0 = np.argsort(eqn_2[:,0])[::-1][:10]
least_infl_0 = np.argsort(eqn_2[:,0])[:10]
most_infl_1 = np.argsort(eqn_2[:,1])[::-1][:10]
least_infl_1 = np.argsort(eqn_2[:,1])[:10]

most_0 = eqn_2[most_infl_0,0]
least_0 = eqn_2[least_infl_0,0]
most_1 = eqn_2[most_infl_1,1]
least_1 = eqn_2[least_infl_1,1]

xax = np.arange(10)
plt.scatter(xax,most_0,label='Positive Influence')
plt.scatter(xax,least_0,label='Negative Influence')
plt.legend(loc='lower right',prop={'size':8})
plt.xticks(xax)
plt.xlabel('Index')
plt.ylabel('Influence')
plt.title('Positive vs Negative Influential Points')
plt.show()

feat_scores = np.sum(eqn_5,axis=0)
