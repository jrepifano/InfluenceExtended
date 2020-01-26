# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:35:08 2020

@author: Jake
"""

import numpy as np
import save_plots

probs = np.load('results/probs/xgb_probs.npy')
test_labels = np.load('results/probs/test_labels_xgb.npy')

save_plots.save_plots('XGBoost',None,None,None,test_labels,probs[:,1])