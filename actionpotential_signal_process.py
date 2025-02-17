# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 11:51:49 2024

@author: ozturklab
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import  rfft, irfft, rfftfreq

data_url= r"C:\Users\ozturklab\Desktop\Chao_Liu_Nanowire\Cell\9-11-2024 JHU Tissue test\0002txt.txt"
df = pd.read_csv(data_url, sep='\t')

x = df['Time (s)'].to_numpy()
MSU = df['Trace #3L (mV)'].to_numpy()
JHU = df['Trace #3R (mV)'].to_numpy()

start = 0
end = 2000
x_U = x[start:end]
y_MSU = MSU[start:end]
y_JHU = -JHU[start:end]


plt.rcParams['text.usetex'] = True

sample_rate = 1E+4


fourier_MSU = rfft(y_MSU)
fourier_JHU = rfft(y_JHU)

N = y_MSU.size
xf = rfftfreq(N, 1 / sample_rate)

plt.figure()
plt.title('FFT',color='black', size='14')
plt.plot(xf, np.abs(fourier_JHU), label='JHU')
plt.plot(xf, np.abs(fourier_MSU), label='MSU')
plt.legend()
plt.show()



sos = signal.iirfilter(N=2, Wn=[59, 61], btype='bandstop', ftype='butter', output='sos', fs=sample_rate)
filtered = signal.sosfiltfilt(sos, y_MSU)

plt.rcParams['text.usetex'] = True
#plt.rcParams['figure.figsize'] = [3.5, 2.625]
#plt.rcParams['figure.dpi'] = 300
plt.figure()

plt.plot (x_U, y_MSU,':', label='Raw MSU', color='#E4BF8C')
plt.plot (x_U, filtered, label='Filtered MSU', color='#F47937')
plt.plot(x_U, y_JHU, label='JHU', color='#68ACE5')

plt.title('Filter ',color='black', size='14')
plt.xlabel ('Time (s)')
plt.ylabel ('Voltage (mV)')
plt.minorticks_on()
plt.legend()
plt.show ()