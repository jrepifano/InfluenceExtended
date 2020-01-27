# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 00:54:23 2020

@author: Jake
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, GridSearchCV
from imblearn.over_sampling import SMOTE
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer

r = np.random.RandomState(seed=1234567890)

x = np.load('data/x.npy')
y = np.load('data/y.npy')
x_imputed = np.load('data/x_imputed.npy')
x_scaled = np.load('data/x_scaled.npy')

feat_names = np.load('data/column_names.npy',allow_pickle=True)

## Corr matrix plot
# corr_matrix = pd.DataFrame(x).corr(method = "spearman").abs()
# # Draw the heatmap
# sns.set(font_scale = 1.0)
# f, ax = plt.subplots(figsize=(11, 9))
# sns.heatmap(corr_matrix, cmap= "YlGnBu", square=True, ax = ax)
# f.tight_layout()
# plt.savefig("results/correlation_matrix.png", dpi = 1080)

## Impute with MICE
imputer = IterativeImputer()
x_imputed = imputer.fit_transform(x)
np.save('data/x_imputed.npy',x_imputed)

## Standardize Data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_imputed)
np.save('data/x_scaled.npy',x_scaled)

## Mutual Information Scores
mi_score = mutual_info_classif(x_scaled,y)
np.save('results/mi_scores.npy',mi_score)

##Training Loop Starts
logReg_params = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100],'solver':['liblinear'],'random_state':[r]}
xgb_params = {'max_depth':[2,5,10],'learning_rate':[0.001,0.01,0.1],'n_estimators':[10,50,100],
              'min_child_weight':[0.1,0.2,0.5,1,5],'gamma':[0,0.1,1,5],'random_state':[r.randint(5,size=1)[0]]}

kf = KFold(n_splits=5)

logreg_probs = np.array([])
xgb_probs = np.array([])
coefs = np.array([])
test_labels = np.array([])
i = 1
print('Starting CV Loop')
for train_index,test_index in kf.split(x_scaled):
    # x_resampled,y_resampled = SMOTE().fit_resample(x_scaled[train_index],y[train_index])
    x_resampled = x_scaled[train_index]
    y_resampled = y[train_index]
    
    logReg = LogisticRegression()
    xg = xgboost.XGBClassifier()
    # print('Starting LogReg grid search')
    # clf1 = GridSearchCV(logReg,logReg_params,make_scorer(roc_auc_score),iid=False,cv=5,refit=True,n_jobs=-1).fit(x_resampled,y_resampled)
    print('Starting xgb grid search')
    clf2 = GridSearchCV(xg,xgb_params,make_scorer(roc_auc_score),iid=False,cv=5,refit=True,n_jobs=3).fit(x_resampled,y_resampled)

    # pd.DataFrame(clf1.cv_results_).to_csv('results/logreg_cv_results_'+str(i)+'.csv')
    pd.DataFrame(clf2.cv_results_).to_csv('results/xgb_cv_results_'+str(i)+'.csv')
    
    # probs_1 = clf1.predict_proba(x_scaled[test_index])
    probs_2 = clf2.predict_proba(x_scaled[test_index])
    
    # logreg_probs = np.vstack((logreg_probs,probs_1)) if logreg_probs.size else probs_1
    xgb_probs = np.vstack((xgb_probs,probs_2)) if xgb_probs.size else probs_2
    # coefs = np.vstack((coefs,clf1.best_estimator_.coef_)) if coefs.size else clf1.best_estimator_.coef_
    test_labels = np.hstack((test_labels,y[test_index])) if test_labels.size else y[test_index]
    
    print('Done Fold: '+str(i) +'/5')
    i+=1
    
# np.save('results/logreg_probs_no_resampling.npy',logreg_probs)
np.save('results/xgb_probs_no_resampling.npy',xgb_probs)
# np.save('results/coefs_no_resampling.npy',coefs)
np.save('results/probs/test_labels_xgb_no_resampling.npy',test_labels)