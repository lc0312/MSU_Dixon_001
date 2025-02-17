# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:15:14 2024

@author: ozturklab
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import signal
from scipy.fft import  rfft, irfft, rfftfreq

data_url= r"C:\Users\ozturklab\\Downloads\153114_18122024.txt"
df = pd.read_csv(data_url, sep=" ")



plt.rcParams['text.usetex'] = True



x = df['x']
y = df['y']

plt.figure()
plt.plot (x, y)
plt.show ()

sample_rate = 6.7

fourier = rfft(y)

N = y.size
xf = rfftfreq(N, 1 / sample_rate)


plt.figure()
plt.plot(xf, np.abs(fourier))
plt.show()


'''
cut_f_signal = fourier.copy()

cut_f_signal[(xf>=56)] = 0
cut_f_signal[(xf<=4)] = 0


cut_signal = irfft(cut_f_signal)
'''

'''
plt.rcParams['text.usetex'] = True
plt.figure()

plt.plot(x, -cut_signal, color='black')

plt.title('Fast Fourier Transformed  ECG',color='black', size='14')
plt.xlabel ('Time (s)')
plt.ylabel ('Voltage (V)')
plt.minorticks_on()
plt.show ()
'''

'''
sos = signal.butter(3,[4,56], btype='bandpass', fs=sample_rate, output='sos')
filtered = signal.sosfilt(sos, cut_signal)
'''
'''
plt.rcParams['text.usetex'] = True
plt.figure()
plt.plot (x, -filtered, color='black')

plt.title('Detected ECG',color='black', size='14')
plt.xlabel ('Time (s)')
plt.ylabel ('Voltage (V)')
plt.minorticks_on()
plt.show ()
'''

'''
plt.rcParams['text.usetex'] = True
#plt.rcParams['figure.dpi'] = 240
plt.figure()

PSD_filtered_0 = signal.savgol_filter(-filtered,81, 3, mode='nearest')

plt.plot (x, PSD_filtered_0,color='black')

plt.title('Detected ECG',color='black', size='14')
plt.xlabel ('Time (s)')
plt.ylabel ('Voltage (V)')
plt.minorticks_on()
plt.show ()
'''