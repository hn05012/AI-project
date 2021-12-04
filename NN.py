

from os import error
import numpy as np
import pandas as pd
import random
from utility_functions import * 
import matplotlib.pyplot as plt 


input_features = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',	'Marginal Adhesion','Single Epithelial Cell Size',	'Bare Nuclei',	'Bland Chromatin',	'Normal Nucleoli',	'Mitoses']

# number of hidden layers = 1

# number of hidden nodes 2/3 times the input nodes + output nodes
# hidden_nodes = math.floor((2/3)*len(input_features) + len(target_class))

# number of hidden nodes < 2 times input nodes
# hidden_nodes = 2*len(input_features) - 1

# input nodes < hidden nodes < output nodes 
# random.seed(42)
# hidden_nodes = random.randint(len(target_class), len(input_features), )
hidden_nodes = 12

weight_hidden = []

for i in range(len(input_features)):
    x = []
    for i in range(hidden_nodes):
        random.seed(random.randint(0,100))
        x.append(round(random.uniform(0,1), 2))
    weight_hidden.append(x)
    x = []
weight_hidden = np.array(weight_hidden)


# there are 7 hidden nodes and 1 output node
# so there will be a total of 7 weights for the output layer

weight_output = []
for i in range(hidden_nodes):
    random.seed(random.randint(0,100))
    weight_output.append([round(random.uniform(0,1), 2)])
weight_output = np.array(weight_output)


# training
def train_NN(train_X, training_labels, weight_hidden, weight_output, epoch):
    lr = 0.01
    w = {'w_hidden': weight_hidden, 'w_output':weight_output,'error':[], 'iterations':[]} 
    for i in range(epoch):
        # input for hidden layer
        input_hidden = np.dot(train_X, weight_hidden)
        
        # output from hidden layer
        output_hidden = sigmoid(input_hidden)

        # input for output layer
        input_op = np.dot(output_hidden, weight_output)

        #output from output layer
        output_op = sigmoid(input_op)

        error_out = ((1/2)*(np.power( (output_op - training_labels), 2 )))          # MSE = sum (1/2*(p - a))
        w['error'].append(float(error_out.sum())/len(error_out))        
        w['iterations'].append(i)

        # backpropogation
        # derivates for phase1 
        derror_douto = output_op - training_labels                        #predicted values - actual output values  
        douto_dino = sigmoid_derivative(input_op)
        dino_dwo = output_hidden

        derror_dwo = np.dot(dino_dwo.T, derror_douto*douto_dino)         # dino_dwo.T takes the transpose of the matrix   

        # derivates for phase 2
        derror_dino = derror_douto*douto_dino
        dino_douth = weight_output
        derror_douth = np.dot(derror_dino, dino_douth.T)
        douth_dinh = sigmoid_derivative(input_hidden)
        dinh_dwh = train_X
        derror_wh = np.dot(dinh_dwh.T, douth_dinh*derror_douth)

        #update weights
        weight_hidden -= lr*derror_wh
        weight_output -= lr*derror_dwo
        # bias -= lr * db

    w['w_hidden'] = weight_hidden
    w['w_output'] = weight_output
    return w





def test_NN(test_X, testing_labels, weight_hidden, weight_output):
    results = {'actual':[], 'predicted':[], 'accuracy':0}
    for i in range(len(test_X)):                                # prediction
        record = test_X[i]
        r1 = np.dot(record, weight_hidden)
        r2 = sigmoid(r1)
        r3 = np.dot(r2, weight_output)
        r4 = sigmoid(r3)
        r4 = float(np.extract(True, r4))
        if r4 <= 0.5:
            r4 = 0
        else:
            r4 = 1
        results['predicted'].append(r4)
    for i in range(len(testing_labels)):                    # actual values
        y = int(np.extract(True, testing_labels[i]))
        results['actual'].append(y)
    if len(results['actual']) == len(results['predicted']):     # comparison and accuracy
        correct = 0
        for i in range(len(results['actual'])):
            if results['actual'][i] == results['predicted'][i]:
                correct+=1
        results['accuracy'] = (correct/len(results['actual']))*100
    return results

def predict_NN(record, w_h, w_o):
    r1 = np.dot(record, w_h)
    r2 = sigmoid(r1)
    r3 = np.dot(r2, w_o)
    r4 = sigmoid(r3)
    r4 = float(np.extract(True, r4))
    if r4 <= 0.5:
        r4 = 0
    else:
        r4 = 1
    return r4



