# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 07:41:01 2020

@author: Jake
"""

import pandas as pd
import numpy as np

df = pd.read_csv('E:/Documents/GitHub/InfluenceExtended/eICU-septic-apache/data/122.csv')
patient_id = df['patientunitstayid']

apach = pd.read_csv('E:/Documents/GitHub/InfluenceExtended/eICU-allcomers-apache/data/apachePatientResult.csv')

apach = apach.loc[apach['patientunitstayid'].isin(patient_id)]
apach = apach.loc[apach['apacheversion']=='IVa']