# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 01:01:10 2020

@author: pdesa
"""

#DATOS CATEGÓRICOS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar dataset
dataset = pd.read_csv('Data.csv')

#Separar variables dependientes de independientes
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Variables categóricas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
onehotencoder = make_column_transformer((OneHotEncoder(), [0]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)

##########################################################################################

#Variables ordinales
from sklearn import preprocessing
labelencoder_y = preprocessing.LabelEncoder()
y = labelencoder_y.fit_transform(y)