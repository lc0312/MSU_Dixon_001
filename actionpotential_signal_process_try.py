# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:02:35 2024

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
MSU = df['Trace #2L (mV)'].to_numpy()
JHU = df['Trace #2R (mV)'].to_numpy()

start = 0
end = 5000
x_U = x[start:end]
y_MSU = MSU[start:end]
y_JHU = -JHU[start:end]

plt.rcParams['text.usetex'] = True

sample_rate = 1E+4

'''
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
'''

fs = sample_rate  # Sampling frequency (Hz)
f0 = 60  # Frequency to be removed (60Hz)

# Compute passband and stopband frequencies for buttord
wp = [3/(fs / 2), 3000 / (fs / 2)]  # Passband frequencies (normalized)
ws = [62/(fs / 2), 58 / (fs / 2)]  # Stopband frequencies (normalized)
gpass = 1E-3  # Maximum loss in the passband (dB)
gstop = 45  # Minimum attenuation in the stopband (dB)

# Determine the order and natural frequency of the filter
N, Wn = signal.buttord(wp, ws, gpass, gstop, fs=sample_rate)

# Create the band-stop filter
b, a = signal.butter(N, Wn, btype='bandstop')

# Apply the filter to the noisy signal
filtered1 = signal.filtfilt(b, a, y_MSU)

sos = signal.iirfilter(N=4, Wn=[59, 61], btype='bandstop', ftype='bessel', output='sos', fs=sample_rate)
filtered = signal.sosfiltfilt(sos, y_MSU)


'''
plt.plot (x_U, y_MSU,':', label='Raw MSU',color='#F47937')
plt.plot (x_U, filtered, label='Filtered MSU', color='#F47937')
plt.plot (x_U, filtered1, label=str(gstop)+ 'Hz')


plt.title('Filter ',color='black', size='14')
plt.xlabel ('Time (s)')
plt.ylabel ('Voltage (mV)')
plt.minorticks_on()
plt.legend()
plt.show ()
'''


fig, axs = plt.subplots(1, 3)

# Plot data on each subplot
axs[0, 0].plot(x_U, y_MSU,color='#F47937')
axs[0, 0].set_title('Raw MSU')

axs[0, 1].plot(x_U, filtered, color='#F47937')
axs[0, 1].set_title('Filtered MSU')

axs[0, 2].plot(x_U,y_JHU, color='#002D72')
axs[0, 2].set_title('JHU')


# Adjust layout to prevent overlapping
plt.tight_layout()
'''
plt.xlabel ('Time (s)')
plt.ylabel ('Voltage (mV)')
plt.minorticks_on()
plt.legend()
'''
plt.show ()