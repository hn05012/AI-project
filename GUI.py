import tkinter as tk
import random
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import math
from math import pow
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from PIL import Image, ImageTk


root= tk.Tk()
clump_thickness=tk.IntVar()
Uniformity_of_size=tk.IntVar()
uniformity_of_shape = tk.IntVar()
marginal_adhesion = tk.IntVar()
epithelial_cell = tk.IntVar()
bare_nuclei = tk.IntVar()
chromotin = tk.IntVar()
nucleoli = tk.IntVar()
mitosis = tk.IntVar()


# img = PhotoImage(file="C:\\Users\\Ali Adnan\\Pictures\\picz.png")
# label = Label(root, image=img)
# label.place(x=0, y=0)



def submit():
    addition = int(clump_entry.get()) + int(size_entry.get()) + int(shape_entry.get())+ int(adhesion_entry.get()) + int(epithelial_entry.get())+ int(nuclei_entry.get())+ int(chromotin_entry.get()) + int(nucleoli.get())+ int(mitosis_entry.get())
    label = Label(root, text = addition, font = 'calibre')
    label.grid(row = 5, column = 6)
clump_label = tk.Label(root, text = 'Clump Thickness : ', font=('Al Bayan',10, 'bold'))
clump_label.grid(row=0, column=0)


clump_entry = tk.Entry(root,textvariable = clump_thickness, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
clump_entry.grid(row=1, column=0)

size_label = tk.Label(root, text = 'Uniformity of Size : ', font = ('calibre',10,'bold'))
size_label.grid(row=0, column=2)
# creating a entry for password
size_entry=tk.Entry(root, textvariable = Uniformity_of_size, font = ('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
size_entry.grid(row=1, column=2)

shape_label = tk.Label(root, text = 'Uniformity of Shape : ', font = ('calibre',10,'bold'))
shape_label.grid(row=0, column=3)

shape_entry = tk.Entry(root,textvariable = uniformity_of_shape , font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
shape_entry.grid(row=1, column=3)


adhesion_label = tk.Label(root, text = 'Marginal Adhesion : ', font = ('calibre',10,'bold'))
adhesion_label.grid(row=0, column=4)

adhesion_entry = tk.Entry(root,textvariable = marginal_adhesion , font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
adhesion_entry.grid(row=1, column=4)
epithelial_label = tk.Label(root, text = 'Epithelial Cell : ', font = ('calibre',10,'bold'))
epithelial_label.grid(row=0, column=5)

epithelial_entry = tk.Entry(root,textvariable = epithelial_cell, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
epithelial_entry.grid(row=1, column=5)


nuclei_label = tk.Label(root, text = 'Bare Nuclei : ', font = ('calibre',10,'bold'))
nuclei_label.grid(row=0, column=6)

nuclei_entry = tk.Entry(root,textvariable = bare_nuclei, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
nuclei_entry.grid(row=1, column=6)

chromotin_label = tk.Label(root, text = 'Chromotin : ', font = ('calibre',10,'bold'))
chromotin_label.grid(row=0, column=7)

chromotin_entry = tk.Entry(root,textvariable = chromotin, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
chromotin_entry.grid(row=1, column=7)

nucleoli_label = tk.Label(root, text = 'Nucleoili : ', font = ('calibre',10,'bold'))
nucleoli_label.grid(row=0, column=8)

nucleoli_entry = tk.Entry(root,textvariable = nucleoli, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
nucleoli_entry.grid(row=1, column=8)

mitosis_label = tk.Label(root, text = 'Mitosis : ', font = ('calibre',10,'bold'))
mitosis_label.grid(row=0, column=9)

mitosis_entry = tk.Entry(root,textvariable = mitosis, font=('calibre',10,'normal'), fg = 'Red', bg = 'Black', borderwidth= 5)
mitosis_entry.grid(row=1, column=9)

class_label = tk.Label(root, text = 'Class : ', font = ('calibre',10,'bold'))
class_label.grid(row=0, column=10)


button=tk.Button(root,text = 'Predict..', command = submit)
button.grid(row=4, column=6)




data2 = {'Number of Iterations': [1,2,3,4,5,6,7,8,9,10],
         'Error': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
        }
df2 = DataFrame(data2,columns=['Number of Iterations','Error'])
 
root.geometry('800x600')
  
figure2 = plt.Figure(figsize=(5,4), dpi=100)
ax2 = figure2.add_subplot(111)
line2 = FigureCanvasTkAgg(figure2, root)
line2.get_tk_widget().grid(row=160, column=1)
df2 = df2[['Number of Iterations','Error']].groupby('Number of Iterations').sum()
df2.plot(kind='line', legend=True, ax=ax2, color='r',marker='o', fontsize=10)
ax2.set_title('Number of Iterations Vs. Error')


root.mainloop()
