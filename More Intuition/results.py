# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:15:08 2019

@author: Jake
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

coefs = np.load('results/log_reg_coefs.npy')

eqn_2_control = np.load('results/eqn_2_control.npy')
eqn_5_control = np.load('results/eqn_5_control.npy')

eqn_2_extended_lin1 = np.load('results/eqn_2_extended_lin1.npy')
eqn_2_extended_lin2 = np.load('results/eqn_2_extended_lin2.npy')
eqn_5_extended_lin1 = np.load('results/eqn_5_extended_lin1.npy')
eqn_5_extended_lin2 = np.load('results/eqn_5_extended_lin2.npy')

eqn_2_selu_lin1 = np.load('results/eqn_2_selu_lin1.npy')
eqn_2_selu_lin2 = np.load('results/eqn_2_selu_lin2.npy')
eqn_5_selu_lin1 = np.load('results/eqn_5_selu_lin1.npy')
eqn_5_selu_lin2 = np.load('results/eqn_5_selu_lin2.npy')

eqn_2_3layer_lin1 = np.load('results/eqn_2_3layer_lin1.npy')
eqn_2_3layer_lin2 = np.load('results/eqn_2_3layer_lin2.npy')
eqn_2_3layer_lin3 = np.load('results/eqn_2_3layer_lin3.npy')
eqn_5_3layer_lin1 = np.load('results/eqn_5_3layer_lin1.npy')
eqn_5_3layer_lin2 = np.load('results/eqn_5_3layer_lin2.npy')
eqn_5_3layer_lin3 = np.load('results/eqn_5_3layer_lin3.npy')

control_corr, control_p = pearsonr(np.sum(eqn_5_control,axis=0),coefs[0])
extended_lin1_corr, extended_lin1_p = pearsonr(np.sum(eqn_5_extended_lin1,axis=0),coefs[0])
extended_lin2_corr, extended_lin2_p = pearsonr(np.sum(eqn_5_extended_lin2,axis=0),coefs[0])
selu_lin1_corr, selu_lin1_p = pearsonr(np.sum(eqn_5_selu_lin1,axis=0),coefs[0])
selu_lin2_corr, selu_lin2_p = pearsonr(np.sum(eqn_5_selu_lin2,axis=0),coefs[0])
three_layer_lin1_corr, three_layer_lin1_p = pearsonr(np.sum(eqn_5_3layer_lin1,axis=0),coefs[0])
three_layer_lin2_corr, three_layer_lin2_p = pearsonr(np.sum(eqn_5_3layer_lin2,axis=0),coefs[0])
three_layer_lin3_corr, three_layer_lin3_p = pearsonr(np.sum(eqn_5_3layer_lin3,axis=0),coefs[0])