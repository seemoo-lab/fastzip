# Power thresholds (in dB) for acc_v, acc_h and gyrW
P_THR_ACC_V = -30.4
P_THR_ACC_H = -28
P_THR_GYR = -36
P_THR_BAR = -12

# SNR thresholds, ensure that signal varies sufficiently
SNR_THR_ACC_V = 1.08
SNR_THR_ACC_H = 1.32
SNR_THR_GYR = 0.82
SNR_THR_BAR = 1.2

# Peak height is defined as fraction of 25% from the height of a maximum peak
PEAK_HEIGHT = 0.25

# Distance between two successive peaks in seconds
PEAK_DISTANCE = 0.25

# Number of peaks threshold for acc_v and acc_h
NP_THR_ACC_V = 11
NP_THR_ACC_H = 14

# Smoothing factor of EWMA filter for acceleration components
EWMA_ACC_V = 0.16
EWMA_ACC_H = 0.2

# Range threshold for bar, i.e., range = max - min after scaling 
RNG_THR_BAR = 1.08
