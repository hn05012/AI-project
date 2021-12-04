import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from utility_functions import *         


def train_BNN(train_X, train_Y, epoch, lr):
    np.random.seed(75)
    weights = np.random.rand(9,1)
    x = weights
    bias = np.random.rand(1)
    errors = []
    iterations = []

    for i in range(epoch):
        iterations.append(i)
        inputs = train_X
        XW = np.dot(inputs, weights)+ bias
        z = sigmoid(XW)
        error = z - train_Y                          # predicted - actual output
        errors.append(float(error.sum())/len(error))      # average error
        dcost = error
        dpred = sigmoid_derivative(z)
        z_del = dcost * dpred
        inputs = train_X.T                                    # transpose of the matrix
        weights = weights - lr*np.dot(inputs, z_del)
        for num in z_del:
            bias = bias - lr*num
    w_b = {'weight':weights, 'bias':bias, 'initial weight':x, 'error':errors, 'iterations':iterations}
    return w_b

                     
def test_BNN(test_X, test_Y, weights, bias):
    results = {'actual':[], 'predicted':[], 'accuracy':0}
    
    for i in range(len(test_X)):                                # prediction
        p = sigmoid(np.dot(test_X[i], weights) + bias)          # predicted value, np array
        p = float(np.extract(True, p))
        if p <= 0.5:
            p = 0
        else:
            p = 1
        results['predicted'].append(p)
    for i in range(len(test_Y)):
        y = int(np.extract(True, test_Y[i]))
        results['actual'].append(y)
    if len(results['actual']) == len(results['predicted']):
        correct = 0
        for i in range(len(results['actual'])):
            if results['actual'][i] == results['predicted'][i]:
                correct+=1
        results['accuracy'] = (correct/len(results['actual']))*100
    return results


def predict_BNN(record, w, b):
    result = sigmoid(np.dot(record, w) + b)
    result = float(np.extract(True, result))
    if result <= 0:
        result = 0
    else:
        result = 1
    return result




