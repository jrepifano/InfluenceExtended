# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:35:08 2020

@author: Jake
"""

import numpy as np
import pandas as pd
import save_plots

probs = np.load('results/probs/mlp_probs.npy')
test_labels = np.load('results/probs/mlp_test_labels.npy')

save_plots.save_plots('MLP No SMOTE',None,None,None,test_labels,probs[:,1])

# column_names = np.load('data/column_names.npy',allow_pickle=True)
# coefs = np.load('results/coefs.npy')
# coefs = np.mean(coefs,axis=0)
# logreg_top_feats = np.argsort(abs(coefs))
# survey_results = pd.read_csv('results/survey_results.csv')

# feat_names = survey_results.columns.values
# survey_top_feats = np.argsort(survey_results.values)
