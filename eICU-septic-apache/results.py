# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:48:25 2020

@author: Jake
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc, precision_recall_curve, average_precision_score


apache_probs =  np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers-apache/results/probs/smote/apache/logreg_probs.npy')[:,1]
apache_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers-apache/results/probs/smote/apache/logreg_test_labels.npy')

aps_probs = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers-apache/results/probs/smote/aps/logreg_probs.npy')[:,1]
aps_labels = np.load('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers-apache/results/probs/smote/aps/logreg_test_labels.npy')

fpr_aps, tpr_aps, thresholds = roc_curve(aps_labels, aps_probs, pos_label=1)
roc_auc_aps=auc(fpr_aps,tpr_aps)

fpr_apache, tpr_apache, thresholds = roc_curve(apache_labels, apache_probs, pos_label=1)
roc_auc_apache=auc(fpr_apache,tpr_apache)

plt.figure()
lw = 3
plt.plot(fpr_aps, tpr_aps,
		lw=lw, label='APS ROC curve (area = %0.4f)'%roc_auc_aps)
plt.plot(fpr_apache, tpr_apache,
		lw=lw, label='APACHE IVa ROC curve (area = %0.4f)'%roc_auc_apache)
plt.plot([0, 1], [0, 1], color='darkgrey', lw=lw, linestyle='--', label='Random Guess (area = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Specificity', fontsize=12)
plt.ylabel('Sensitivity',fontsize=12)
plt.title('APACHE IV vs APS ROC: SMOTE',fontsize=14)
plt.legend(loc="lower right")
plt.grid()
# roc_title = 'APACHE IV vs APS' + ' ROC.pdf'
# pdf = PdfPages(os.path.join(roc_title))
# pdf.savefig(dpi=600, bbox_inches='tight', pad_inches=.15)
# pdf.close()
plt.show()