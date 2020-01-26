# -*- coding: utf-8 -*-
"""
@author: epifanoj0
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/master.csv')

data = df.iloc[:,:29]
labels = df.iloc[:,30]

feat_names = data.columns

np_data = data.values

np.save('data/data.npy',np_data)
np.save('data/feat_names.npy',feat_names)


y = []
for i in range(len(labels)):
    if labels[i] == 'ALIVE':
        y.append(0)
    else:
        y.append(1)

y = np.asarray(y)

np.save('data/labels.npy',y)