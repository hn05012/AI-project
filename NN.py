

from os import error
import numpy as np
import pandas as pd
import math
import random
from utility_functions import * 


input_features = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',	'Marginal Adhesion','Single Epithelial Cell Size',	'Bare Nuclei',	'Bland Chromatin',	'Normal Nucleoli',	'Mitoses']
input_data = pd.DataFrame(pd.read_csv("updated_dataset.csv"), columns = input_features)
input_data = input_data.apply(pd.to_numeric)

predictor_values = input_data.values

target_class = ["Class"]
target_dataframe = pd.DataFrame(pd.read_csv("updated_dataset.csv"), columns=target_class)
target_data = target_dataframe.apply(pd.to_numeric)

target_data= np.array(target_data)
target_data.reshape(len(target_data), 1)            # converts output data to vector

# number of hidden layers = 1
# number of hidden nodes 2/3 times the input nodes + output nodes
# hidden_nodes = math.floor((2/3)*len(input_features) + len(target_class))

# number of hidden nodes < 2 times input nodes
# hidden_nodes = 2*len(input_features) - 1

# input nodes < hidden nodes < output nodes 
# random.seed(42)
# hidden_nodes = random.randint(len(target_class), len(input_features), )
hidden_nodes = 3

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

# a low learning rate achieves minimal error rate
lr = 0.8


# training
def train(weight_hidden, weight_output):
    for epoch in range(10000):
        # input for hidden layer
        input_hidden = np.dot(predictor_values, weight_hidden)
        
        # output from hidden layer
        output_hidden = sigmoid(input_hidden)

        # input for output layer
        input_op = np.dot(output_hidden, weight_output)

        #output from output layer
        output_op = sigmoid(input_op)

        error_out = ((1/2)*(np.power( (output_op - target_data), 2 )))

        # derivates for phase1 
        derror_douto = output_op - target_data          #predicted values - actual output values  
        douto_dino = sigmoid(input_op)
        dino_dwo = output_hidden

        derror_dwo = np.dot(dino_dwo.T, derror_douto*douto_dino)         # dino_dwo.T takes the transpose of the matrix   

        # derivates for phase 2
        derror_dino = derror_douto*douto_dino
        dino_douth = weight_output
        derror_douth = np.dot(derror_dino, dino_douth.T)
        douth_dinh = sigmoid(input_hidden)
        dinh_dwh = predictor_values
        derror_wh = np.dot(dinh_dwh.T, douth_dinh*derror_douth)

        #update weights
        weight_hidden -= lr*derror_wh
        weight_output -= lr*derror_dwo


train(weight_hidden, weight_output)
# prediction
record = np.array([7,6,4,4,10,5,1,1,1])
r1 = np.dot(record, weight_hidden)
r2 = sigmoid(r1)
r3 = np.dot(r2, weight_output)
r4 = sigmoid(r3)
print(r4)