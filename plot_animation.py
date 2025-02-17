# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import keyboard
import pandas as pd

data_url= r"C:\Users\ozturklab\Desktop\Chao_Liu_Nanowire\Cell\11-13-2024\in and out.txt"
df = pd.read_csv(data_url, delimiter = "\t")

time = df['T'].to_numpy() 
data = df_y = df['V'].to_numpy()

fig, ax = plt.subplots()


# Create a line object, which will be updated during the animation
line, = ax.plot([], [], lw=2)

# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# Animation function: called sequentially to update the data
def animate(i):
    window_size = 1
    if time[i] > window_size:
        ax.set_xlim(time[i] - window_size, time[i])
        
    else:
        ax.set_xlim(0, window_size)  

    
    x = time[:i]
    y = data[:i]
    line.set_data(x, y)
    return line,

# Create the animation
ani = FuncAnimation(fig, animate, init_func=init, frames=len(time), interval=1, blit=True)
ani.save("inadnout.mp4", writer="ffmpeg")
# Display the animation
#plt.show()