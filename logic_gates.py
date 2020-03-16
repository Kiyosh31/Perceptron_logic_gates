# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:10:41 2020

@author: david
"""

import numpy as np
from matplotlib import pyplot as plt

class Neurona:
    def __init__(self, dim, eta):
        self.n = dim
        self.eta = eta
        # pesos sinapticos
        self.w = -1 + 2 * np.random.random((dim, 1))
        # sesgo o bias
        self.b = -1 + 2 * np.random.random()
        
    def predict(self, x):
        # producto punto
        y = np.dot(self.w.transpose(), x) + self.b
        if y > 0:
            return 1
        else:
            return 0
        
    def train(self, X, y, epochs = 50):
        # X -> entradas
        # Y -> valores deseados
        n, m = X.shape
        for i in range(epochs):
            for j in range(m):
                y_pred = self.predict(X[:,j])
                self.w += self.eta*(y[j] - y_pred) * X[:,j].reshape(-1,1)
                self.b += self.eta*(y[j] - y_pred)
                




logic_and = Neurona(2, 0.1)
X = np.array([
        [0, 1, 0, 1], 
        [0, 0, 1, 1]
        ])
y = np.array([0, 0, 0, 1])
logic_and.train(X, y, 30)

w = logic_and.w
b = logic_and.b

plt.title("Resultado Compuerta AND")
plt.grid(True)
plt.plot(X[0,0], X[1,0], 'or')
plt.plot(X[0,1], X[1,1], 'or')
plt.plot(X[0,2], X[1,2], 'or')
plt.plot(X[0,3], X[1,3], '^b')
plt.plot([-1, 3],[(-w[0]/w[1]) * (-1) - (b/w[1]), (-w[0]/w[1]) * (3) - (b/w[1])], '-m')
plt.xlim(-1, 3)
plt.ylim(-1, 3)
plt.show()

print("Salida compuerta AND")
for i in range(4):
    print(logic_and.predict(X[:,i]))
    
    
    
    
    

logic_or = Neurona(2, 0.1)
y = np.array([0, 1, 1, 1])
logic_or.train(X, y, 30)

w = logic_or.w
b = logic_or.b

plt.title("Resultado Compuerta OR")
plt.grid(True)
plt.plot(X[0,0], X[1,0], 'or')
plt.plot(X[0,1], X[1,1], '^b')
plt.plot(X[0,2], X[1,2], '^b')
plt.plot(X[0,3], X[1,3], '^b')
plt.plot([-2,3],[(-w[0]/w[1])*(-2)-(b/w[1]),(-w[0]/w[1])*(3)-(b/w[1])],'-m')
plt.xlim(-2, 3)
plt.ylim(-2, 3)
plt.show()

print("Salida compuerta OR")
for i in range(4):
    print(logic_or.predict(X[:,i]))
    




logic_xor = Neurona(2, 0.1)
y = np.array([0, 1, 1, 0])
logic_xor.train(X, y, 30)

w = logic_xor.w
b = logic_xor.b

plt.title("Resultado Compuerta XOR")
plt.grid(True)
plt.plot(X[0,0], X[1,0], 'or')
plt.plot(X[0,1], X[1,1], '^b')
plt.plot(X[0,2], X[1,2], '^b')
plt.plot(X[0,3], X[1,3], 'or')
plt.plot([-2,3],[(-w[0]/w[1])*(-2)-(b/w[1]),(-w[0]/w[1])*(3)-(b/w[1])],'-m')
plt.xlim(-2, 3)
plt.ylim(-2, 3)
plt.show()

print("Salida compuerta XOR")
for i in range(4):
    print(logic_xor.predict(X[:,i]))