import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from math import pow
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from NN import train_NN, test_NN, predict_NN, weight_hidden, weight_output
from BasicNN import train_BNN, test_BNN, predict_BNN



root= tk.Tk()
root.title("Breast Cancer Evaluation") 
root.geometry('800x600')
root.configure(bg='Black')

basicnn = False
nn = True



input_features = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',	'Marginal Adhesion','Single Epithelial Cell Size',	'Bare Nuclei',	'Bland Chromatin',	'Normal Nucleoli',	'Mitoses']
input_data = pd.DataFrame(pd.read_csv("updated_dataset.csv"), columns = input_features)

train_X = input_data.sample(frac = 0.6)
test_X = input_data.drop(train_X.index)
train_X = train_X.apply(pd.to_numeric)
train_X = np.array(train_X.values)
test_X = test_X.apply(pd.to_numeric)
test_X = np.array(test_X.values)


target_class = ["Class"]
target_dataframe = pd.DataFrame(pd.read_csv("updated_dataset.csv"), columns=target_class)

train_Y = target_dataframe.sample(frac = 0.6)
test_Y = target_dataframe.drop(train_Y.index)

train_Y = train_Y.apply(pd.to_numeric)
training_labels = np.array(train_Y.values)
training_labels.reshape(len(train_Y.values), 1)

test_Y = test_Y.apply(pd.to_numeric)
testing_labels = np.array(test_Y.values)
testing_labels.reshape(len(test_Y.values), 1)

# function call to NN train
if nn:
    weights = train_NN(train_X, training_labels, weight_hidden, weight_output, 1000)
    w_h = weights['w_hidden']
    w_o = weights['w_output']
    err = weights['error']
    itr = weights['iterations']
    # function call to NN test
    results = test_NN(test_X, testing_labels, w_h, w_o)    
    x = results['actual']
    y = results['predicted']
    accuracy = results['accuracy']


    clump_thickness=tk.IntVar()
    Uniformity_of_size=tk.IntVar()
    uniformity_of_shape = tk.IntVar()
    marginal_adhesion = tk.IntVar()
    epithelial_cell = tk.IntVar()
    bare_nuclei = tk.IntVar()
    chromotin = IntVar()
    nucleoli = IntVar()
    mitosis = IntVar()



    #This function takes input from the network and makes a scatter plot to show the accuracy
    def view_accuracy_graph():
        data3 = {'Actual Output Values': x,
                'Predicted Values': y}
                
        df3 = pd.DataFrame(data3,columns=['Actual Output Values','Predicted Values'])
        figure3 = plt.Figure(figsize=(5,50), dpi=100, facecolor='black')
        ax3 = figure3.add_subplot(111)
        ax3.scatter(df3['Actual Output Values'],df3['Predicted Values'], color = 'r', s = 0.1)
        scatter3 = FigureCanvasTkAgg(figure3, root) 
        scatter3.get_tk_widget().place(x=50, y=10, relheight=0.8)
        ax3.set_facecolor('black')
        ax3.xaxis.label.set_color('white')      
        ax3.yaxis.label.set_color('white')          
        ax3.tick_params(axis='x', colors='white')    
        ax3.tick_params(axis='y', colors='white')  
        ax3.spines['left'].set_color('white')        
        ax3.spines['bottom'].set_color('white') 
        ax3.spines['right'].set_color('white')        
        ax3.spines['top'].set_color('white')
        ax3.legend(['predicted']) 
        ax3.set_xlabel('actual')
        ax3.set_ylabel('predicted')
        ax3.set_title('Accuracy: ' + str(accuracy) + ' %', color= 'white')


    #This function takes input from the network and makes a scatter plot to show the errors
    def display_error_graph():
        data2 = {'Number of Iterations': itr,
                    'Error': err}
        df2 = pd.DataFrame(data2,columns=['Number of Iterations','Error'])
        figure2 = plt.Figure(figsize=(5,50), dpi=100, facecolor='black')
        ax2 = figure2.add_subplot(111)
        ax2.scatter(df2['Number of Iterations'],df2['Error'], color = 'white', s = 0.75)
        scatter3 = FigureCanvasTkAgg(figure2, root) 
        scatter3.get_tk_widget().place(x=900, y=10, relheight=0.8)
        ax2.set_facecolor('black')
        ax2.xaxis.label.set_color('white')        
        ax2.yaxis.label.set_color('white')          
        ax2.tick_params(axis='x', colors='white')    
        ax2.tick_params(axis='y', colors='white')  
        ax2.spines['left'].set_color('white')        
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white')        
        ax2.spines['bottom'].set_color('white')
        ax2.legend(['Error']) 
        ax2.set_xlabel('Number of Iterations')
        ax2.set_ylabel('Error')
        ax2.set_title('Number of Iterations Vs. Error', color= 'white')


    #This function predicts whether a cell is cancerous or not
    def predict_function():
        record = np.array([int(clump_entry.get()), int(size_entry.get()), int(shape_entry.get()), int(adhesion_entry.get()), int(epithelial_entry.get()),int(nuclei_entry.get()), int(chromotin_entry.get()) , int(nucleoli.get()), int(mitosis_entry.get())])
        p = predict_NN(record, w_h, w_o)
        if p<=0.5:
            label = Label(root, text = 'Cancer cell is Benign', bg= 'Black', fg='White', padx = 5, pady = 5, font = 'calibre')
        else:
            label = Label(root, text = "Cancer cell is Malignant", bg= 'Black', fg='White', padx = 5, pady = 5, font = 'calibre')
        label.pack()

    #All the labels and enteries are made here 
    clump_label = tk.Label(root, text = 'Clump Thickness : ', font=('calibre',10, 'bold'), bg= 'Black', fg='White')
    clump_label.pack()

    clump_entry = tk.Entry(root,textvariable = clump_thickness, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    clump_entry.pack(padx= 5, pady=5) 


    size_label = tk.Label(root, text = 'Uniformity of Size : ', font = ('calibre',10,'bold'), bg= 'Black', fg='White')
    size_label.pack()

    size_entry=tk.Entry(root, textvariable = Uniformity_of_size, font = ('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    size_entry.pack(padx= 5, pady=5)


    shape_label = tk.Label(root, text = 'Uniformity of Shape : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    shape_label.pack()

    shape_entry = tk.Entry(root,textvariable = uniformity_of_shape , font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    shape_entry.pack(padx= 5, pady=5)


    adhesion_label = tk.Label(root, text = 'Marginal Adhesion : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    adhesion_label.pack()

    adhesion_entry = tk.Entry(root,textvariable = marginal_adhesion , font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    adhesion_entry.pack(padx= 5, pady=5)


    epithelial_label = tk.Label(root, text = 'Epithelial Cell : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    epithelial_label.pack()

    epithelial_entry = tk.Entry(root,textvariable = epithelial_cell, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    epithelial_entry.pack(padx= 5, pady=5)


    nuclei_label = tk.Label(root, text = 'Bare Nuclei : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    nuclei_label.pack()

    nuclei_entry = tk.Entry(root,textvariable = bare_nuclei, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    nuclei_entry.pack(padx= 5, pady = 5)


    chromotin_label = tk.Label(root, text = 'Chromotin : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    chromotin_label.pack()

    chromotin_entry = tk.Entry(root,textvariable = chromotin, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    chromotin_entry.pack(padx= 5, pady=5)

    nucleoli_label = tk.Label(root, text = 'Nucleoili : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    nucleoli_label.pack()

    nucleoli_entry = tk.Entry(root,textvariable = nucleoli, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    nucleoli_entry.pack(padx= 5, pady=5)

    mitosis_label = tk.Label(root, text = 'Mitosis : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    mitosis_label.pack()

    mitosis_entry = tk.Entry(root,textvariable = mitosis, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    mitosis_entry.pack(padx= 5, pady=5)


    #These 3 buttons plot the graphs and make the prediction based on the input
    button=tk.Button(root,text = 'Predict',bg= 'Black', fg='White', command = predict_function)
    button.pack()

    button=tk.Button(root,text = 'Error Function',bg= 'Black', fg='White', command = display_error_graph)
    button.pack()
    
    button=tk.Button(root,text = 'View Accuracy',bg= 'Black', fg='White', command = view_accuracy_graph)
    button.pack()
        
    root.mainloop()


elif basicnn:
    weights = train_BNN(train_X, training_labels, 2000, 0.01)
    w = weights['weight']
    b = weights['bias']
    err = weights['error']
    itr = weights['iterations']
    
    # function call to NN test
    results = test_BNN(test_X, testing_labels, w, b)    
    x = results['actual']
    y = results['predicted']
    accuracy = results['accuracy']


    clump_thickness=tk.IntVar()
    Uniformity_of_size=tk.IntVar()
    uniformity_of_shape = tk.IntVar()
    marginal_adhesion = tk.IntVar()
    epithelial_cell = tk.IntVar()
    bare_nuclei = tk.IntVar()
    chromotin = IntVar()
    nucleoli = IntVar()
    mitosis = IntVar()



    #This function takes input from the network and makes a scatter plot to show the accuracy
    def view_accuracy_graph():
        data3 = {'Actual Output Values': x,
                'Predicted Values': y}
                
        df3 = pd.DataFrame(data3,columns=['Actual Output Values','Predicted Values'])
        figure3 = plt.Figure(figsize=(5,50), dpi=100, facecolor='black')
        ax3 = figure3.add_subplot(111)
        ax3.scatter(df3['Actual Output Values'],df3['Predicted Values'], color = 'r', s = 0.1)
        scatter3 = FigureCanvasTkAgg(figure3, root) 
        scatter3.get_tk_widget().place(x=50, y=10, relheight=0.8)
        ax3.set_facecolor('black')
        ax3.xaxis.label.set_color('white')      
        ax3.yaxis.label.set_color('white')          
        ax3.tick_params(axis='x', colors='white')    
        ax3.tick_params(axis='y', colors='white')  
        ax3.spines['left'].set_color('white')        
        ax3.spines['bottom'].set_color('white') 
        ax3.spines['right'].set_color('white')        
        ax3.spines['top'].set_color('white')
        ax3.legend(['predicted']) 
        ax3.set_xlabel('actual')
        ax3.set_ylabel('predicted')
        ax3.set_title('Accuracy: ' + str(accuracy) + ' %', color= 'white')


    #This function takes input from the network and makes a scatter plot to show the errors
    def display_error_graph():
        data2 = {'Number of Iterations': itr,
                    'Error': err}
        df2 = pd.DataFrame(data2,columns=['Number of Iterations','Error'])
        figure2 = plt.Figure(figsize=(5,50), dpi=100, facecolor='black')
        ax2 = figure2.add_subplot(111)
        ax2.scatter(df2['Number of Iterations'],df2['Error'], color = 'white', s = 0.75)
        scatter3 = FigureCanvasTkAgg(figure2, root) 
        scatter3.get_tk_widget().place(x=900, y=10, relheight=0.8)
        ax2.set_facecolor('black')
        ax2.xaxis.label.set_color('white')        
        ax2.yaxis.label.set_color('white')          
        ax2.tick_params(axis='x', colors='white')    
        ax2.tick_params(axis='y', colors='white')  
        ax2.spines['left'].set_color('white')        
        ax2.spines['top'].set_color('white')
        ax2.spines['right'].set_color('white')        
        ax2.spines['bottom'].set_color('white')
        ax2.legend(['Error']) 
        ax2.set_xlabel('Number of Iterations')
        ax2.set_ylabel('Error')
        ax2.set_title('Number of Iterations Vs. Error', color= 'white')


    #This function predicts whether a cell is cancerous or not
    def predict_function():
        record = np.array([int(clump_entry.get()), int(size_entry.get()), int(shape_entry.get()), int(adhesion_entry.get()), int(epithelial_entry.get()),int(nuclei_entry.get()), int(chromotin_entry.get()) , int(nucleoli.get()), int(mitosis_entry.get())])
        p = predict_BNN(record, w, b)
        if p<=0.5:
            label = Label(root, text = 'Cancer cell is Benign', bg= 'Black', fg='White', padx = 5, pady = 5, font = 'calibre')
        else:
            label = Label(root, text = "Cancer cell is Malignant", bg= 'Black', fg='White', padx = 5, pady = 5, font = 'calibre')
        label.pack()

    #All the labels and enteries are made here 
    clump_label = tk.Label(root, text = 'Clump Thickness : ', font=('calibre',10, 'bold'), bg= 'Black', fg='White')
    clump_label.pack()

    clump_entry = tk.Entry(root,textvariable = clump_thickness, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    clump_entry.pack(padx= 5, pady=5) 


    size_label = tk.Label(root, text = 'Uniformity of Size : ', font = ('calibre',10,'bold'), bg= 'Black', fg='White')
    size_label.pack()

    size_entry=tk.Entry(root, textvariable = Uniformity_of_size, font = ('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    size_entry.pack(padx= 5, pady=5)


    shape_label = tk.Label(root, text = 'Uniformity of Shape : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    shape_label.pack()

    shape_entry = tk.Entry(root,textvariable = uniformity_of_shape , font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    shape_entry.pack(padx= 5, pady=5)


    adhesion_label = tk.Label(root, text = 'Marginal Adhesion : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    adhesion_label.pack()

    adhesion_entry = tk.Entry(root,textvariable = marginal_adhesion , font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    adhesion_entry.pack(padx= 5, pady=5)


    epithelial_label = tk.Label(root, text = 'Epithelial Cell : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    epithelial_label.pack()

    epithelial_entry = tk.Entry(root,textvariable = epithelial_cell, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    epithelial_entry.pack(padx= 5, pady=5)


    nuclei_label = tk.Label(root, text = 'Bare Nuclei : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    nuclei_label.pack()

    nuclei_entry = tk.Entry(root,textvariable = bare_nuclei, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    nuclei_entry.pack(padx= 5, pady = 5)


    chromotin_label = tk.Label(root, text = 'Chromotin : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    chromotin_label.pack()

    chromotin_entry = tk.Entry(root,textvariable = chromotin, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    chromotin_entry.pack(padx= 5, pady=5)

    nucleoli_label = tk.Label(root, text = 'Nucleoili : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    nucleoli_label.pack()

    nucleoli_entry = tk.Entry(root,textvariable = nucleoli, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    nucleoli_entry.pack(padx= 5, pady=5)

    mitosis_label = tk.Label(root, text = 'Mitosis : ', font = ('calibre',10,'bold'),bg= 'Black', fg='White')
    mitosis_label.pack()

    mitosis_entry = tk.Entry(root,textvariable = mitosis, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
    mitosis_entry.pack(padx= 5, pady=5)


    #These 3 buttons plot the graphs and make the prediction based on the input
    button=tk.Button(root,text = 'Predict',bg= 'Black', fg='White', command = predict_function)
    button.pack()

    button=tk.Button(root,text = 'Error Function',bg= 'Black', fg='White', command = display_error_graph)
    button.pack()
    
    button=tk.Button(root,text = 'View Accuracy',bg= 'Black', fg='White', command = view_accuracy_graph)
    button.pack()
        
    root.mainloop()
 
