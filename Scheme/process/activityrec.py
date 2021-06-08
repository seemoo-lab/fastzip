import numpy as np
from scipy.signal import find_peaks
from process.procdata import ewma_filter
from const.activityconst import *


def compute_sig_power(sig):
    # Check if sig is non-zero
    if len(sig) == 0:
        print('compute_sig_power: signal must have non-zero length!')
        return
    
    return 10 * np.log10(np.sum(sig ** 2) / len(sig))


def get_acc_peaks(sig, fs):
    # Find a peak threshold: height computed as a fraction of the proximity to the max peak
    peak_height = np.mean(sorted(sig)[-9:]) * PEAK_HEIGHT

    # Search peaks using height and distance parameters
    peaks, _ = find_peaks(sig, height=peak_height, distance=fs * PEAK_DISTANCE)
    
    return peak_height, peaks


def compute_snr(sig):
    # Check if sig is non-zero
    if len(sig) == 0:
        print('compute_snr: signal must have non-zero length!')
        return
    
    # We take abs as such SNR computation requires non-negative values, 
    # see https://en.wikipedia.org/wiki/Signal-to-noise_ratio#Alternative_definition
    return np.mean(abs(sig)) / np.std(abs(sig))


def do_activty_recongition(sig, sensor_type, fs):
    # Activity recognition metrics
    power = 0
    snr = 0
    n_peaks = 0
    
    # Make a copy of a signal to be working on
    ar_sig = np.copy(sig)
    
    # Compute signal power (similar for acc, gyr and bar)
    power = compute_sig_power(ar_sig)
    
    # For acc we need an extra n_peaks estimation; acc needs to be smoothed for n_peaks and snr estimation 
    if sensor_type == 'acc_v' or sensor_type == 'acc_h':
        # Set the alpha for EWMA filter
        if sensor_type == 'acc_v':
            alpha = EWMA_ACC_V
        else:
            alpha = EWMA_ACC_H
      
        # Smooth acc further with EWMA filter
        ar_sig = ewma_filter(abs(ar_sig), alpha)
        
        # Get number of prominent peaks
        n_peaks = len(get_acc_peaks(ar_sig, fs)[1])
    
    # Compute singal's SNR (works similar for acc, gyr and bar)
    snr = compute_snr(ar_sig)
    
    return power, snr, n_peaks
