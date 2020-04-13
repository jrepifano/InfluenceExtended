# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:18:30 2020

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

df = pd.read_csv('E:/Documents/GitHub/InfluenceExtended/eICU-septic-apache/data/122.csv')
patient_id = df['patientunitstayid']

apach = pd.read_csv('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers-apache/data/apachePatientResult.csv')

apach = apach.loc[apach['patientunitstayid'].isin(patient_id)]
apach = apach.loc[apach['apacheversion']=='IVa']

aps = apach['acutephysiologyscore'].to_numpy().reshape((-1,1))
apache4 = apach['apachescore'].to_numpy().reshape((-1,1))

labels = apach['actualicumortality'].to_numpy()

y = []
for i in range(len(labels)):
    if labels[i] == 'ALIVE':
        y.append(0)
    else:
        y.append(1)
        
y = np.asarray(y)

logReg_params = {'penalty':['l1','l2'],'C':[0.01,0.1,1,10,100],'solver':['liblinear'],'random_state':[r]}

kf = KFold(n_splits=5)

logreg_probs = np.array([])
test_labels = np.array([])
i = 1

print('Starting CV Loop')
for train_index,test_index in kf.split(aps):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(apache4[train_index])
    y_train = y[train_index]
    
    x_test = scaler.transform(apache4[test_index])
    y_test = y[test_index]
    
    x_train,y_train = SMOTE().fit_resample(x_train,y_train)
    
    logReg = LogisticRegression()
    print('Starting LogReg grid search')
    clf1 = GridSearchCV(logReg,logReg_params,make_scorer(roc_auc_score),iid=False,cv=5,refit=True,n_jobs=-1).fit(x_train,y_train)

    pd.DataFrame(clf1.cv_results_).to_csv('results/tuning_params/smote/apache/logreg_cv_results_'+str(i)+'.csv')

    probs_1 = clf1.predict_proba(x_test)
    
    logreg_probs = np.vstack((logreg_probs,probs_1)) if logreg_probs.size else probs_1

    test_labels = np.hstack((test_labels,y_test)) if test_labels.size else y_test
    
    print('Done Fold: '+str(i) +'/5')
    i+=1
    
np.save('results/probs/smote/apache/logreg_probs.npy',logreg_probs)
np.save('results/probs/smote/apache/logreg_test_labels.npy',test_labels)