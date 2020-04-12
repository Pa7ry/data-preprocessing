# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:18:03 2020

@author: pdesa
"""


#PLANTILLA DE PREPROCESADO DE DATOS

#Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar dataset
dataset = pd.read_csv('Data.csv')
#Crea una variable con las filas y columnas indicadas
#empieza:acaba // : --> selecciona de principio a fin
#Las variables de matrices van en mayus y las de vectores en min
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
