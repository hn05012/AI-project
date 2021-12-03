import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import random 
from utility_functions import * 


dataset = pd.read_csv('updated_dataset.csv')

predictor = pd.DataFrame(dataset, columns= ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',	'Marginal Adhesion','Single Epithelial Cell Size',	'Bare Nuclei',	'Bland Chromatin',	'Normal Nucleoli',	'Mitoses'])

target = pd.DataFrame(dataset, columns= ['Class'])


# dividing dataset into training and testing data
train_X = predictor.sample(frac = 0.8)
test_X = predictor.drop(train_X.index)

train_Y = target.sample(frac = 0.8)
test_Y = target.drop(train_X.index)

# converting data values to integer and storing in numpy array
train_X = train_X.apply(pd.to_numeric)
train_X = np.array(train_X.values)
test_X = test_X.apply(pd.to_numeric)
test_X = np.array(test_X.values)

train_Y = train_Y.apply(pd.to_numeric)
training_labels = np.array(train_Y.values)
training_labels = training_labels.reshape(len(train_Y.values),1)                    # reshape converts it to a vector

test_Y = test_Y.apply(pd.to_numeric)
testing_labels = np.array(test_Y.values)
testing_labels = testing_labels.reshape(len(test_Y.values),1)                    


def train(train_X, train_Y, epoch, lr):
    np.random.seed(0)
    weights = np.random.rand(9,1)
    x = weights
    bias = np.random.rand(1)

    for i in range(epoch):
        inputs = train_X
        XW = np.dot(inputs, weights)+ bias
        z = sigmoid(XW)
        error = z - train_Y                          # predicted - actual output
        dcost = error
        dpred = sigmoid_derivative(z)
        z_del = dcost * dpred
        inputs = train_X.T                                    # transpose of the matrix
        weights = weights - lr*np.dot(inputs, z_del)
        for num in z_del:
            bias = bias - lr*num
    w_b = {'weight':weights, 'bias':bias, 'initial weight':x}
    return w_b

                     
def test(test_X, test_Y, weights, bias):
    results = {'actual':[], 'predicted':[], 'accuracy':0}
    
    for i in range(len(test_X)):
        p = sigmoid(np.dot(test_X[i], weights) + bias)
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





w_b = train(train_X, training_labels, 20000, 0.55)
weight = w_b['weight']
bias = w_b['bias']
results = test(test_X, testing_labels, weight, bias)

x = results['actual']
y = results['predicted']
print(x)
print(y)
accuracy = results['accuracy']

plt.scatter(x,y, color= "green", marker= "*", s=30)
plt.xlabel('actual output values')
plt.ylabel('predicted output values')
plt.title("Accuracy " + str(accuracy) + "%")
plt.show()




# single_pt = np.array([5,1,1,1,2,1,3,1,1])
# result = sigmoid(np.dot(single_pt, w_b['weight']) + w_b['bias'])
# print(len(result))



