# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:51:49 2024

@author: ozturklab
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kneefinder import KneeFinder

data_url= r"C:\Users\ozturklab\Desktop\Chao_Liu_Nanowire\EASA\1209\10mv.txt"
df = pd.read_csv(data_url, sep='\t')


x = df['V'].to_numpy()
y = df['I'].to_numpy()


start = 10*40
end = 10*60
x_ = x[start:end]
y_ = y[start:end]


plt.figure()
plt.plot (x_ , y_)
plt.xlabel ('Voltage (V)')
plt.ylabel ('Current (A)')
plt.show ()


knee_finder = KneeFinder(x_, y_)
knee_x, knee_y = knee_finder.find_knee()


plt.plot(x_, y_)
plt.plot(knee_x, knee_y, 'ro')
plt.show()