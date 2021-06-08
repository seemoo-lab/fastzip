import numpy as np
from scipy.signal import butter, sosfilt, cheby2, medfilt, savgol_filter
from scipy.ndimage import gaussian_filter
from math import ceil
import sys
from common.helper import check_if_pos_numbers
from const.globconst import *

# Supported filter types
FILTER_TYPES = ['butter', 'cheby2']

# Filter pass types
PASS_TYPES =  ['lowpass', 'highpass', 'bandpass', 'bandstop']

PROC_FILTERS = ['med', 'gaus', 'savgol']

# G unit
G_UNIT = 9.80665


# Convert pressure data (hPa) to altidtude (meters)
def convert_meters(press):
    # Return altitude in meters
    alt = np.zeros(len(press))

    # Iterate over press np.array
    for i in range(0, len(press)):
        # Convert pressure value to height in meters according to formula:
        # https://www.weather.gov/media/epz/wxcalc/pressureAltitude.pdf
        alt[i] = (1 - (press[i] / 1013.25) ** 0.190284) * 145336.45 * 0.3048

    return alt


# Exponentially weighted moving average filter
def ewma_filter(data, alpha=0.15):
    # Initialize out_data np.array
    ewma_data = np.zeros(len(data))

    # Set the first out_data value to the one in in_data
    ewma_data[0] = data[0]

    # Apply the EWMA filter to data (np.array)
    for i in range(1, len(ewma_data)):
        # Formula: currentHeight = a*sensorHeight + (1-a)*prevHeight
        # (source: "Using Mobile Phone Barometer for Low-Power
        # Transportation Context Detection")
        ewma_data[i] = alpha * data[i] + (1 - alpha) * ewma_data[i - 1]

    return ewma_data


def remove_noise(data, apply_to):
    # Array to store filtered acc data
    rn_data = np.zeros(data.shape)
    
    if apply_to == 'all':
        # Filter depending on dimensions
        if len(data.shape) == 1:
            # Apply first Savitzky–Golay and then Gaussian filters to data
            rn_data = gaussian_filter(savgol_filter(data, 3, 2), sigma=1.4)
            
        elif len(data.shape) > 1:
            for i in range(0, data.shape[1]):
                # Apply first Savitzky–Golay and then Gaussian filters to each column
                rn_data[:, i] = gaussian_filter(savgol_filter(data[:, i], 3, 2), sigma=1.4)
        else:
            print('remove_noise: data must have a non-zero dimension!')
            sys.exit(0)
        
    elif apply_to == 'chunk':
        # Here we only deal with 1D data, no need to check the dimensions
        rn_data = gaussian_filter(savgol_filter(data, 5, 3), sigma=1.4)
        
    else:
        print('remove_noise: "apply_to" parameter can only be "all" or "chunk"!')
        sys.exit(0)
        
    return rn_data


def decompose_acc(data, fs, sw_len=5, sw_step=1, convert_flag=True):
    # Convert sliding window length and step from seconds to # of samples
    sw_len = sw_len * fs
    sw_step = sw_step * fs

    # Array of resulting vertical and horizontal acc components
    dec_acc = np.zeros((len(data), 2))

    # Iterate over the data: i depends on the sw_len (with ceil we cover all samples)
    for i in range(0, ceil(len(data) / sw_len)):
        # Get a submatrix of length sw_len
        sw_frame = data[i * sw_len:(i + 1) * sw_len, :].copy()

        # Check if the number of columns is valid
        if sw_frame.shape[1] != 3:
            print('decompose_acc: provided "data" should be three dimensional!')
            sys.exit(0)

        # Gravity estimation over each column X, Y and Z
        g_est = np.array([np.mean(sw_frame, axis=0)])

        # Remove gravity from sw_frame
        sw_frame = sw_frame - g_est

        # Convert acc values from m/s^2 to G units
        if convert_flag:
            sw_frame = sw_frame / G_UNIT
            g_est = g_est / G_UNIT

        # Compute the vertical component as a dot product between
        # the acc without gravity and gravity estimation
        v_acc = np.dot(sw_frame, g_est.T)

        # Norm is computed for each row (axis=1) in the resulting matrix
        h_acc = np.linalg.norm(sw_frame - v_acc * g_est, axis=1)

        # Update dec_acc
        dec_acc[i * sw_len:(i + 1) * sw_len, 0] = v_acc.T
        dec_acc[i * sw_len:(i + 1) * sw_len, 1] = h_acc.T

    return dec_acc


def process_data(data, sensor_type, fs):
    # Store processed sensor data
    proc_data = {}
    
    # Iterate over data
    for k, v in sorted(data.items()):
        # Sensor processing pipelines
        if sensor_type == 'acc':
            # 1) extract gravity from X, Y, Z axes, 2) decompose acc to horizontal and vertical components
            # in the world reference frame, and 3) filter data
            proc_data[k] = remove_noise(decompose_acc(v.iloc[:, 0:len(v.columns) - 1].values, fs), 'all')
        elif sensor_type == 'gyrW':
            # 1) Get only the Z-axis (pointig towards Sky) of gyr data -> already world coordinates 2) filter data
            proc_data[k] = remove_noise(v['Z'].values, 'all')
        elif sensor_type == 'bar':
            # 1) convert data from pressure (hPa) to altidtude (meters), 2) filter data
            proc_data[k] = remove_noise(convert_meters(v.iloc[:, 0:len(v.columns) - 1].values), 'all')
        elif sensor_type == 'unmag':
            print('process_data: no processing for "unmag" modality is implemented, no data is returned!')
            return
        else:
            print('process_data: allowed sensor types are "acc", "gyrW", "bar" and "unmag"!')
            sys.exit(0)

    return proc_data
