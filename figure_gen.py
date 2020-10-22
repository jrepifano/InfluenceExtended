# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 07:21:52 2020

@author: Jake
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc, precision_recall_curve, average_precision_score


apache_probs =  np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic-apache/results/probs/smote/apache/logreg_probs.npy')[:,1]
apache_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic-apache/results/probs/smote/apache/logreg_test_labels.npy')
aps_probs = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic-apache/results/probs/smote/aps/logreg_probs.npy')[:,1]
aps_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic-apache/results/probs/smote/aps/logreg_test_labels.npy')
lr_probs = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic/results/probs/smote/logreg_probs.npy')[:,1]
lr_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic/results/probs/smote/logreg_test_labels.npy')
xgb_probs = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic/results/probs/smote/xgb_probs.npy')[:,1]
xgb_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic/results/probs/smote/xgb_test_labels.npy')
mlp_probs = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic/results/probs/smote/mlp_probs.npy')[:,1]
mlp_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-septic/results/probs/smote/mlp_test_labels.npy')

lr_fpr, lr_tpr, thresholds = roc_curve(lr_labels, lr_probs, pos_label=1)
lr_auc = auc(lr_fpr, lr_tpr)

xgb_fpr, xgb_tpr, thresholds = roc_curve(xgb_labels, xgb_probs, pos_label=1)
xgb_auc = auc(xgb_fpr, xgb_tpr)

mlp_fpr, mlp_tpr, thresholds = roc_curve(mlp_labels, mlp_probs, pos_label=1)
mlp_auc = auc(mlp_fpr, mlp_tpr)

plt.figure()
lw = 3
plt.plot(lr_fpr, lr_tpr,
		lw=lw, label='Logistic Regression (area = %0.4f)'% lr_auc)
plt.plot(xgb_fpr, xgb_tpr,
		lw=lw, label='XGBoost (area = %0.4f)'% xgb_auc)
plt.plot(mlp_fpr, mlp_tpr,
		lw=lw, label='Neural Net (area = %0.4f)'% mlp_auc)

fpr_aps, tpr_aps, thresholds = roc_curve(aps_labels, aps_probs, pos_label=1)
roc_auc_aps=auc(fpr_aps,tpr_aps)

fpr_apache, tpr_apache, thresholds = roc_curve(apache_labels, apache_probs, pos_label=1)
roc_auc_apache=auc(fpr_apache,tpr_apache)

lw = 3
plt.plot(fpr_aps, tpr_aps,
		lw=lw, label='SAPS (area = %0.4f)'%roc_auc_aps)
plt.plot(fpr_apache, tpr_apache,
		lw=lw, label='APACHE (area = %0.4f)'%roc_auc_apache)
plt.plot([0, 1], [0, 1], color='darkgrey', lw=lw, linestyle='--', label='Random Guess (area = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity', fontsize=12)
plt.ylabel('Sensitivity',fontsize=12)
plt.title('ROC: Septic Patients',fontsize=14)
plt.legend(loc="lower right")
plt.grid()
plt.show()