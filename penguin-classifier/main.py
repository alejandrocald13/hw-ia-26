import numpy as np
import pandas as pd

# Matrices de peso (obtenidas en .R)

WI = np.array([
    [  2.192217 ,   0.5246864],
    [ -2.954160 ,  -8.9487519],
    [ 20.892754 ,   3.6770797],
    [ -7.864390 ,   0.4837516],
    [-18.692350 ,   2.4763967]
])

W2 = np.array([
    [-4.948473 ,  -5.843387 , -3.4558859],
    [ 2.395414 ,  10.015232 ,  0.5416288],
    [ 6.795956 , -10.269666 ,  6.4977414]
])

W3 = np.array([
    [ -29.657471,   -4.32322,    27.70592],
    [  64.555982,   -7.82514,  -122.33129],
    [ -36.054764,  102.40149,  -270.47453],
    [   3.362089,  -50.67397,    49.39689]
])

# Valores obtenidos en R
mins = np.array([32.1, 13.1, 176.0, 2700.0])
maxs = np.array([59.6, 21.5, 231.0, 6050.0])

def f_act(X):
    X = np.clip(X, -500, 500)
    return 1 / (1 + np.exp(-X))

# Cargar el mismo test que generó R
penguins = pd.read_csv("test_penguins.csv")

xcols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
especie = np.array(["Adelie", "Chinstrap", "Gentoo"])

X = penguins[xcols].copy()
X = (X - mins) / (maxs - mins)
X.insert(0, 'bias', 1)

prediccion = []

for index, fila in X.iterrows():
    fila = fila.to_numpy(dtype=float)

    capa1 = f_act(fila.dot(WI))
    capa1 = np.insert(capa1, 0, 1)

    capa2 = f_act(capa1.dot(W2))
    capa2 = np.insert(capa2, 0, 1)

    salida = f_act(capa2.dot(W3))
    prediccion.append(especie[np.argmax(salida)])

penguins['prediccion'] = prediccion

print(penguins.head())
print()

erroneas = penguins[penguins['species'] != penguins['prediccion']]
print(erroneas.head())
print()

eficiencia = (1 - len(erroneas) / len(penguins)) * 100
print(eficiencia)