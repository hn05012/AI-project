import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import random 
# List of activation functions  

def sigmoid(x):
    return 1/1+np.exp(-1)

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

# input layer for dataset 
# column headings 
independent_variable = [] 
# sales_time = pd.read_csv('sales_time_data.csv')
# sales_time['Date'] = pd.to_datetime(sales_time['Date'])
# sales_time['Date']=sales_time['Date'].map(dt.datetime.toordinal)

# target = ['Sales']
# predictor = ['Date']

# X = sales_time[predictor].values
# Y = sales_time[target].values   

dataset = pd.read_csv(r'C:\Users\ABC\Documents\fall 2021\AI\Project\updated_dataset.csv')

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
for epoch in range(1000):
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
    weights = weights - lr*np.dot(inputs, z_del)
    
    for num in z_del:
        bias = bias - lr*num

single_pt = np.array([5,1,1,1,2,1,3,1,1])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)


#iterations = np.array([0,999])


#plt.plot(errors,iterations)
#plt.show() 


