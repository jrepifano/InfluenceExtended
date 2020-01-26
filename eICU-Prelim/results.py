# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 01:25:01 2020

@author: Jake
"""

import numpy as np

coefs = np.load('results/logreg_coefs.npy')
eqn_2 = np.load('results/eqn_2-test_set.npy')
eqn_5 = np.load('results/eqn_5-test_set.npy')
feat_names = np.load('data/feat_names.npy',allow_pickle=True)

eqn_5_sum = np.sum(eqn_5,axis=0)

eqn_5_sort = np.argsort(eqn_5_sum)
coefs_sort = np.argsort(abs(coefs))[0]

top_5_infl = np.flip(eqn_5_sort[-15:])
top_5_coefs = np.flip(coefs_sort[-15:])

print(feat_names[top_5_infl])
print(feat_names[top_5_coefs])



# Influence Functions top 15 features:

# ['INR_max' 'BICARBONATE_min' 'SODIUM_max' 'WBC_max' 'POTASSIUM_min'
#  'PLATELET_min' 'alt' 'day1pao2' 'CHLORIDE_min' 'CREATININE_max' 'age'
#  'calcium' 'BUN_max' 'ALBUMIN_min' 'POTASSIUM_max']

# Logistic Regression top 15 features:
# ['BICARBONATE_min' 'CHLORIDE_min' 'LACTATE_max' 'WBC_min' 'WBC_max'
#  'HEMATOCRIT_min' 'BUN_min' 'BUN_max' 'SODIUM_max' 'CHLORIDE_max'
#  'BILIRUBIN_max' 'PT_max' 'BANDS_max' 'SODIUM_min' 'age']
