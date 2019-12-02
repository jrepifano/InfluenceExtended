# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:56:53 2019

@author: epifanoj0

Look at correlations between feature importance estimates...
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()

mi_scores = scaler1.fit_transform(np.load('mi_feat_scores.npy').reshape(-1,1))

scaler2 = StandardScaler()
log_reg_coefs = scaler2.fit_transform(np.load('log_reg_coefs.npy').reshape(-1,1))

eqn_2_control = np.load('eqn_2_control.npy')
eqn_5_control = np.load('eqn_5_control.npy')

eqn_2_extended_lin1 = np.load('eqn_2_extended_lin1.npy')
eqn_5_extended_lin1 = np.load('eqn_5_extended_lin1.npy')

eqn_2_extended_lin2 = np.load('eqn_2_extended_lin2.npy')
eqn_5_extended_lin2 = np.load('eqn_5_extended_lin2.npy')

network_eqn_2 = eqn_2_extended_lin1 + eqn_2_extended_lin2
network_eqn_5 = eqn_5_extended_lin1 + eqn_5_extended_lin2


scaler3 = StandardScaler()
influence_control_feat_importance = scaler3.fit_transform(np.sum(eqn_5_control,axis=0).reshape(-1,1))
scaler4 = StandardScaler()
influence_extended_feat_importance = scaler4.fit_transform(np.sum(network_eqn_5,axis=0).reshape(-1,1))

xax = np.arange(4)

plt.plot(xax,mi_scores.reshape(-1),label='Mutual Information Scores')
plt.plot(xax,log_reg_coefs.reshape(-1),label='Logistic Regression Coefs')
plt.plot(xax,influence_control_feat_importance.reshape(-1),label='Author Influence Function code via LogReg')
plt.plot(xax,influence_extended_feat_importance.reshape(-1),label='Influence Extended via NN')
plt.xlabel('feature number (0-3)')
plt.ylabel('Normalized Feature Score')
plt.title('Comparison of Measured Feature Importance')
plt.legend(loc='lower right',prop={'size':8})
plt.xticks(np.arange(4))
plt.show()
