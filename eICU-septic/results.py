# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:35:08 2020

@author: Jake
"""

import numpy as np
import pandas as pd
import save_plots

# noise = np.load('data/x_noise.pkl',allow_pickle=True)
# probs = np.load('results/probs/smote/xgb_prob.npy')
# test_labels = np.load('results/probs/smote/test_labels_xgb.npy')

# save_plots.save_plots('XGBoost SMOTE',None,None,None,test_labels,probs[:,1])

# survey_results = pd.read_csv('results/survey_results.csv')
# feat_names = survey_results.columns.values
# survey_top_feats = np.flip(np.argsort(survey_results.values))

column_names = np.load('data/column_names.npy',allow_pickle=True)
# coefs = np.load('results/coefs_smote.npy')
# coefs = np.mean(coefs,axis=0)
# logreg_top_feats = np.flip(np.argsort(abs(coefs)))

# mi_score = np.load('results/mi_scores.npy')
# mi_top_feats = np.flip(np.argsort(mi_score))

# xgb_shap = np.load('results/xgb_shap_smote.npy')
# xgb_top_feats = np.flip(np.argsort(abs(xgb_shap)))


mlp_shap = np.load('results/mlp_shap_values_smote.npy')
sums = np.sum(np.sum((mlp_shap[0],mlp_shap[1]),axis=1),axis=0)
mlp_shap_top_feats = np.flip(np.argsort(abs(sums)))

# eqn_2 = np.load('results/eqn_2-test_set.npy')
eqn_5 = np.load('results/eqn_5-test_set_cases-no_smote.npy')

infl_feat_importance = np.sum(eqn_5,axis=0)
infl_top_feats = np.flip(np.argsort(abs(infl_feat_importance)))

for i in range(10):
     print(column_names[mlp_shap_top_feats[i]])


# print(np.load('results/mlp_influence_time.npy')/60)