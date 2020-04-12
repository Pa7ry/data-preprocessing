# -*- coding: utf-8 -*-

#Regresión lineal simple

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar dataset
dataset = pd.read_csv('Salary_Data.csv')
# Crear variable con las filas y columnas indicadas
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Crear modelo regresión lineal
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjuto de test
y_pred = regression.predict(X_test)
y_pred_train = regression.predict(X_train)


# Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, c="blueviolet")
plt.plot(X_train, y_pred_train, c="magenta")
plt.title("Sueldo vs Experiencia (conjunto de entrenamiento)")
plt.ylabel('Sueldo ($)')
plt.xlabel('Experiencia (años)')
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test, y_test, c="blueviolet")
plt.plot(X_train, y_pred_train, c="magenta")
plt.title("Sueldo vs Experiencia (conjunto de test)")
plt.ylabel('Sueldo ($)')
plt.xlabel('Experiencia (años)')
plt.show()