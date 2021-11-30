import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import random 
from activation_function import * 

# input layer for dataset 
# column headings 
independent_variable = [] 

dataset = pd.read_csv('updated_dataset.csv')

#print('target')
target = pd.DataFrame(dataset, columns= ['Class'])
#print(target)

#print('predictor')
predictor = pd.DataFrame(dataset, columns= ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',	'Marginal Adhesion','Single Epithelial Cell Size',	'Bare Nuclei',	'Bland Chromatin',	'Normal Nucleoli',	'Mitoses'])
#print(predictor)
predictor = predictor.apply(pd.to_numeric)
target = target.apply(pd.to_numeric)

input_set = np.array(predictor.values)
labels = np.array(target.values)
labels = labels.reshape(len(predictor.values),1) #to convert labels to vector

np.random.seed(42)
weights = np.random.rand(9,1)
bias = np.random.rand(1)
lr = 0.05 #learning rate
errors = [] 
for epoch in range(10):
    inputs = input_set
    XW = np.dot(inputs, weights)+ bias
    z = sigmoid(XW)
    error = z - labels
    errors.append(error) 
    # print(error.sum())
    dcost = error
    dpred = sigmoid_derivative(z)
    z_del = dcost * dpred
    inputs = input_set.T
    # print(weights)
    # print('\n')
    weights = weights - lr*np.dot(inputs, z_del)
    # print(weights)
    # print('\n')
    for num in z_del:
        bias = bias - lr*num

single_pt = np.array([5,1,1,1,2,1,3,1,1])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)


#iterations = np.array([0,999])


#plt.plot(errors,iterations)
#plt.show() 


