# List of activation functions  
import numpy as np 

def sigmoid(x):
    return 1/(1+np.exp(-1))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


def binarystep(x):
    return np.heaviside(x,1)

def linear(x):
    return x 

def tanh(x): 
    return np.tanh(x)

def relu(x):
    x1=[]
    for i in x:
        if i<0:
            x1.append(0)
        else:
            x1.append(i)

    return x1

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
