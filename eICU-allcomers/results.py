# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:35:08 2020

@author: Jake
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc, precision_recall_curve, average_precision_score

# lr_probs = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers/results/probs/smote/logreg_probs.npy')[:,1]
# lr_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers/results/probs/smote/logreg_test_labels.npy')
# xgb_probs = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers/results/probs/smote/xgb_probs.npy')[:,1]
# xgb_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers/results/probs/smote/xgb_test_labels.npy')
# mlp_probs = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers/results/probs/smote/mlp_probs.npy')[:,1]
# mlp_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers/results/probs/smote/mlp_test_labels.npy')

# lr_fpr, lr_tpr, thresholds = roc_curve(lr_labels, lr_probs, pos_label=1)
# lr_auc = auc(lr_fpr, lr_tpr)

# xgb_fpr, xgb_tpr, thresholds = roc_curve(xgb_labels, xgb_probs, pos_label=1)
# xgb_auc = auc(xgb_fpr, xgb_tpr)

# mlp_fpr, mlp_tpr, thresholds = roc_curve(mlp_labels, mlp_probs, pos_label=1)
# mlp_auc = auc(mlp_fpr, mlp_tpr)

# plt.figure()
# lw = 3
# plt.plot(lr_fpr, lr_tpr,
# 		lw=lw, label='Logistic Regression (area = %0.4f)'% lr_auc)
# plt.plot(xgb_fpr, xgb_tpr,
# 		lw=lw, label='XGBoost (area = %0.4f)'% xgb_auc)
# plt.plot(mlp_fpr, mlp_tpr,
# 		lw=lw, label='MLP (area = %0.4f)'% mlp_auc)
# plt.plot([0, 1], [0, 1], color='darkgrey', lw=lw, linestyle='--', label='Random Guess (area = 0.50)')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Specificity', fontsize=12)
# plt.ylabel('Sensitivity',fontsize=12)
# plt.title('Trained Models: All Patients',fontsize=14)
# plt.legend(loc="lower right")
# plt.grid()
# plt.show()

# noise = np.load('data/x_noise.pkl',allow_pickle=True)
# probs = np.load('results/probs/smote/mlp_probs.npy')[:,1]
# test_labels = np.load('results/probs/smote/mlp_test_labels.npy')

# save_plots.save_plots('Neural Network SMOTE',None,None,None,test_labels,probs[:,1])

# survey_results = pd.read_csv('results/survey_results.csv')
# feat_names = survey_results.columns.values
# survey_top_feats = np.flip(np.argsort(survey_results.values))

column_names = np.load('data/column_names.npy',allow_pickle=True)
# coefs = np.load('results/coefs_no_smote.npy')
# coefs = np.mean(coefs,axis=0)
# logreg_top_feats = np.flip(np.argsort(abs(coefs)))

# mi_score = np.load('results/mi_scores.npy')
# mi_top_feats = np.flip(np.argsort(mi_score))

# xgb_shap = np.load('results/xgb_shap_no_smote.npy')
# xgb_shap = np.sum(xgb_shap,axis=0)
# xgb_top_feats = np.flip(np.argsort(abs(xgb_shap)))


mlp_shap = np.load('results/mlp_shap_values_no_smote.npy')
sums = np.sum(np.sum((mlp_shap[0],mlp_shap[1]),axis=1),axis=0)
mlp_shap_top_feats = np.flip(np.argsort(abs(sums)))

# eqn_2 = np.load('results/eqn_2-test_set.npy')
# eqn_5 = np.load('results/eqn_5-test_set_smote.npy')

# infl_feat_importance = np.sum(eqn_5,axis=0)
# infl_top_feats = np.flip(np.argsort((infl_feat_importance)))

for i in range(11):
      # print(feat_names[survey_top_feats[0,i]])
        # print(column_names[mi_top_feats[i]])
        # print(column_names[logreg_top_feats[i]])
        # print(column_names[xgb_top_feats[i]])
       print(column_names[mlp_shap_top_feats[i]])
        # print(column_names[infl_top_feats[i]])


