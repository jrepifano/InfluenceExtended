#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:04:59 2020

@author: epifanoj
"""

import numpy as np
import xgboost
import shap
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

r = np.random.RandomState(seed=1234567890)

x_imputed = np.load('data/x_imputed.npy')
y = np.load('data/y.npy')

scaler = StandardScaler()
x_scaled =  scaler.fit_transform(x_imputed)

sm = SMOTE()

x_resampled, y_resampled = sm.fit_resample(x_scaled,y)


feat_names = np.load('data/column_names.npy',allow_pickle=True)

model = xgboost.train({"gamma":0,"learning_rate": 0.01,"max_depth":2,"min_child_weight":0.1,
                       "n_estimators":50,"random_state":2}, xgboost.DMatrix(x_resampled, label=y_resampled), 100)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_scaled)

shap.summary_plot(shap_values, x_scaled, plot_type="bar")