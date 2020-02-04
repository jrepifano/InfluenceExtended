# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:59:26 2020

@author: Jake
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/122.csv')

data = df.iloc[:,1:33]
labels = df.iloc[:,34]

counter = np.zeros((data.shape[1]))
for i in range(len(data)):
    for j in range(data.shape[1]):
        if(data.iloc[i,j]<0 or pd.isnull(data.iloc[i,j])):
            counter[j]+=1
            
normalized_counter = counter/len(data)

col_to_drop = np.where(normalized_counter>0.5)[0]

print('Dropping Features with more than 50% missing: '+str(data.columns[col_to_drop]))

data = data.drop(data.columns[col_to_drop],axis=1)

y = []
for i in range(len(labels)):
    if labels[i] == 'ALIVE':
        y.append(0)
    else:
        y.append(1)
        
y = np.asarray(y)

np.save('data/column_names.npy',data.columns.values,allow_pickle=True)
np.save('data/x.npy',data.values)
np.save('data/y.npy',y)
