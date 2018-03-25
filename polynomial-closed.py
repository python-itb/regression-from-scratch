#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 22:19:07 2018

@author: amajidsinar
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-white')
np.random.seed(69)

house = datasets.load_boston()
dataset = house.data
#one feature
X = dataset[:,5]
Y = house.target

sorted_X = np.argsort(X)

def normalize(array):
    return (array - np.mean(array)) / np.std(array)

#plt.scatter(X,Y)
#plt.title('boston house pricing')
#plt.xlabel('number of room')
#plt.ylabel('price')
#
#X = normalize(X)
#
#plt.scatter(X,Y)
#plt.title('boston house pricing')
#plt.xlabel('number of room')
#plt.ylabel('price')

degree = 8
lamb = 0.002

from numpy.linalg import inv

order = np.arange(degree).reshape(degree,1,1)
X = (X**order).reshape(degree,X.shape[0]).T

w = np.dot(inv(np.dot(X.T,(X+lamb))),np.dot(X.T,Y))

x = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 1000000)
x = (x**order).reshape(degree,1000000).T

y = np.dot(x,w)

plt.plot(x[:,1], y, color='#008080', label='Gradient Descent')
# Ploting Scatter Points
plt.scatter(X[:,1], Y, c='#ef5423', label='Scatter Plot')
#plt.xlim(-4,4)
#plt.ylim(-2,3)

plt.title('boston house pricing')
plt.xlabel('number of room')
plt.ylabel('price')
plt.legend()
plt.show()

n = X.shape[0]
cost = 1/(2*n)*np.sum((Y-np.dot(X,w))**2)
print(cost)
