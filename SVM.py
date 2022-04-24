# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:18:53 2022
SVM
@author: CarlosDHernandez

encontrqr matriz , G h A B P q
"""
import cvxopt#instalar
import cvxopt.solver
import numpy as np 

class SVM():
    def __int__(self, kernal= "linear", C=1, gamma=1, degree=1):
        self.C = float(c)
        self.gamma = float(gamma)
        self.d = int(degree)
        
        if kernel == "linear":
            self.kernel = self.linear
        elif kernel == "poly":
            self.kernel == self.polymonial
        elif kernel == "gaussian":
            self.kernel == self.gaussian
        else:
            raise NameError("No existe")
        
        
    def linear(self, x1, x2):
        return np.dot(x1,x2)
    
    def polymonial(self, x1,x2):
        return (np.dot(x1,x2) + 1) ** self.d
    
    def gaussian (self, x1, x2):
        return np.exp(- self.gamma * np.linalg.norm(x1 - x2) ** 2)
    
    def fir(self, X, y):
        n, n_features = X.shape
        
        k = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    K[i,j] = self.kernel(X[i], X[j])
 #optimizar con cvxopt
         P = cvxopt.matrix(np.outer(y,y) * K)
         q = cvxopt.matrix(np.ones(n)*-1)
         A = cvxopt.matrix(y, (1,n))
         b = cvxopt.matrix(y, (0.0)
                           
        if self.C is None or selfC == 0:
            G = cvxopt.matrix(np.identity(n) *  -1)
            h = cvxopt.matrix(np.zeros(n))
        else: #soft
            temp1 = np.identity(n) * -1
            temp2 = np.identity(n)
            G = cvxopt.matrix(np.vstack((temp1,temp2)))
            temp1 = np.zeros(n)
            temp2 = np.one(n) * self.C
            h = cvxopt.matrix(np.vstack((temp1,temp2)))
            
        cvxopt.solvers.options["show_progress"] = False
        solution = cvxopt.solvers.qp(P,q,G,h,A,b)
        lamb = np.ravel(solution["x"])
        #lect vectores soporte
        sv = lamb > 1e-5
        ind = np.arange(len(lamb))[sv]
        self.lamb = lamb[sv]
        #guardar vec soportes 
        self.sv = X[sv]
        self.sv_y = y[sv]
        
        self.b = 0
        for i in range(len(self.lamb)):
            self.b += self.sv_y [i]
            self.b = np.sum(self.lamb* self.sv_y * K[ind[i], sv])
            self.b = self.b / len(self.lamb)
            
    def project(self, X) : 
         y_pred = np.zeros(len(X))
         for i in range(len(X)):
             s = 0
             for a, sv_y sv in zip(self.lamb, self.sv_y, self.sv):
                 s += a * sv_y * self.kernel(X[i], sv)
             y_pred[i] = s
        return y_pred + self.b 
    
    def predict (self, X): 
        return np.sign(self.project(X))
                
            

#Experimentos
"""
en SVM_test en default es linealmente no separable
pero al utilizar duros con C= 0.0

luego con c = 1.0
c se busc a en escala logartimica en escalas de 10 
utilizando una c de 10.0 el maegen se hace ,mas pequeño es decir mas pequeño 
entre mas crece 
sobre entrenar

con c 0.01 margen mas grande 

ahora con poly con c = 0.0 y en degree = 1 lo importante aqui es el degree
con degree = 2 , 3 , 7
c 1 degree 3

gassian c = 0.0, sin degree gamma es importante  mator que 0
gamma 10
gamma puede ser muy pequeño , 0.0001

"""


