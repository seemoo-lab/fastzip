from glob import glob
import numpy as np
import pandas as pd
import os
import sys
import re
from datetime import datetime
from const.globconst import *

# Constants
# Acc, gyr, mag fields: 3-axis values + timestamp
AXIS_FIELDS = ['X', 'Y', 'Z', 'TS']

# Bar, hum, lux, tmp fields: a single value + timestamp
VALUE_FIELDS = ['V1', 'TS']


# List input sensor date files
def get_data_files(root_path, sensor_type, mode):
    # Store sensor data files
    file_list = []

    # File extension
    file_ext = ''
    if mode == 'new':
        # For non-compressed files use '.txt'
        file_ext = '.txt.gz'

    # Sanity checks on 'root_path'
    # 1. Check if 'root_path' exists
    if not os.path.exists(root_path):
        print('get_data_files: root_path="%s" does not exist!' % root_path)
        sys.exit(0)

    #  2. Stupid Windows can do that
    if root_path[-1] == '\\':
        root_path = root_path[0:-1] + '/'

    # 3. Add slash to 'root_path' in case it is missing
    if root_path[-1] != '/':
        root_path = root_path + '/'

    # Check if provided 'sensor_type' is supported
    if sensor_type in SENSOR_TYPES:
        path = root_path + 'Sensor-*/sensors' + '/' + sensor_type + 'Data' + file_ext
    else:
        print('get_data_files: sensor_type="%s" is unknown, use: '
              '"acc", "unacc", "lacc", "laccW", "grav", "bar", "gyr", "ungyr", '
              '"hum", "lux", "mag", "unmag" or "tmp"!' % sensor_type)
        sys.exit(0)

    # Iterate over matched files
    for data_file in glob(path, recursive=True):
        # Add a full filename to the list
        file_list.append(data_file)

    # Sort 'file_list'
    file_list.sort()

    return file_list


# Read sensor data to Pandas dataframe
def load_data_file(filename, n_data_points):
    # Read the first row in a file
    cols = pd.read_csv(filename, delim_whitespace=True, nrows=1).columns

    if n_data_points == 3:
        # Check if we have min number of columns: 4 (X, Y, Z, TS)
        if len(cols) < 4:
            print('load_data_file: file=%s contains only %d columns, should be at least %d!' %
                  (filename, len(cols), len(AXIS_FIELDS)))
            sys.exit(0)

        # ToDo: switch back to np.float32 for memory efficiency
        # Construct column_types dict for explicit data types when loading data
        # column_types = dict(zip(AXIS_FIELDS[:-1], [np.float32, np.float32, np.float32]))
        column_types = dict(zip(AXIS_FIELDS[:-1], [float, float, float]))

        # If we have headers we load AXIS_FIELDS, otherwise load columns with AXIS_FIELDS names
        if set(cols).intersection(AXIS_FIELDS):
            # Case: data collected with Android smartphones (NEW)
            # Data format for acc, gyr, mag: 3-axis values + timestamp
            df = pd.read_csv(filename, delim_whitespace=True, usecols=AXIS_FIELDS, dtype=column_types)
        else:
            # Case: data collected with SensorTags (OLD)
            # Data format for acc, gyr, mag: 3-axis values + timestamp
            df = pd.read_csv(filename, delim_whitespace=True, names=AXIS_FIELDS, dtype=column_types)
    else:
        # Check if we have min number of columns: 2 (V1, TS)
        if len(cols) < 2:
            print('load_data_file: file=%s contains only %d columns, should be at least %d!' %
                  (filename, len(cols), len(VALUE_FIELDS)))
            sys.exit(0)

        # ToDo: switch back to np.float32 for memory efficiency
        # Construct column_types dict for explicit data types when loading data
        # column_types = dict(zip(VALUE_FIELDS[:-1], [np.float32]))
        column_types = dict(zip(VALUE_FIELDS[:-1], [float]))

        # If we have headers we load VALUE_FIELDS, otherwise load columns with VALUE_FIELDS names
        if set(cols).intersection(VALUE_FIELDS):
            # Case: data collected with Android smartphones (NEW)
            # Data format for bar, (hum, lux, tmp): single value + timestamp
            df = pd.read_csv(filename, delim_whitespace=True, usecols=VALUE_FIELDS, dtype=column_types)
        else:
            # Case: data collected with SensorTags (OLD)
            # Data format for bar, hum, lux, tmp: single value + timestamp
            df = pd.read_csv(filename, delim_whitespace=True, names=VALUE_FIELDS, dtype=column_types)

    # Replace 'T' in a timestamp with a space
    df['TS'].replace('T', ' ', regex=True, inplace=True)

    return df


# Extract sensor number, e.g. 01 from file name, e.g. .../Sensor-01/sensors/...
def get_sensor_num(filename):
    # Search for a match in the filename
    # (take different slashes into account: / or \)
    match = re.search(r'Sensor-(.*)(?:/|\\)sensors', filename)

    # If there is no match - exit
    if not match:
        print('get_sensor_num: No match for the sensor number in %s, exiting...' % filename)
        sys.exit(0)

    return match.group(1)


# Read data files
def read_data_files(root_path, sensor_type, mode='new'):

    # Dictionary to store sensor data
    data = {}

    # Get list of files containing 'sensor_type' data
    file_list = get_data_files(root_path, sensor_type, mode)

    # Check if 'file_list' is not empty
    if not file_list:
        print('read_data_files: File list for "%s" is empty, exiting...' % sensor_type)
        sys.exit(0)

    # Number of data points is 1 for 'bar', 'hum', 'lux', 'tmp'
    n_data_points = 1

    # If we have 'acc', 'gyr' or 'mag' set a number of data points to 3,
    if sensor_type == 'acc' or sensor_type == 'unacc' or sensor_type == 'lacc' or sensor_type == 'laccW' or \
            sensor_type == 'grav' or sensor_type == 'gyr' or sensor_type == 'gyrW' or sensor_type == 'ungyr' or \
            sensor_type == 'mag' or sensor_type == 'unmag':
        n_data_points = 3

    # Iterate over 'file_list' and load each file into data dictionary
    for file in file_list:
        # Add each loaded file (Pandas dataframe) to the dictionary
        data[get_sensor_num(file)] = load_data_file(file, n_data_points)

    return data


# Align data based on a timestamp
def align_data_ts(data):
    # Lists to store start and stop timestamps
    start_ts = []
    stop_ts = []

    # Iterate over dict of pandas dataframes
    for k, v in sorted(data.items()):
        # print('%s: %s %s' % (k, v['timestamp'].iloc[0], v['timestamp'].iloc[-1]))
        # Get the first timestamp in the dataframe
        start_ts.append(v['TS'].iloc[0])
        # Get the last timestamp in the dataframe
        stop_ts.append(v['TS'].iloc[-1])

    # Find the latest start and earliest stop timestamps
    max_start = max(start_ts)
    min_stop = min(stop_ts)

    # Iterate over dict of pandas dataframes
    for k, v in sorted(data.items()):
        # Query pandas dataframe to find timestamps corresponding to
        # 'max_start' without milliseconds (format: YYYY-MM-DD HH:MM:SS)
        start_ts = v[v['TS'].str.match(max_start.split('.')[0])]

        # Get only 'timestamp' field from the query result and convert it to list
        start_ts = start_ts['TS'].tolist()

        # Sort the list
        start_ts.sort()

        # List to store differences between 'max_start' and timestamps from 'start_ts'
        ts_diff = []

        # Iterate over 'start_ts' list
        for ts in start_ts:
            # Store the abs difference between 'ts' and 'max_start'
            ts_diff.append(abs(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f').timestamp() -
                               datetime.strptime(max_start, '%Y-%m-%d %H:%M:%S.%f').timestamp()))

        # Find the timestamp closest to 'max_start'
        closest_start_ts = start_ts[ts_diff.index(min(ts_diff))]

        # Get Pandas dataframe index of the closest timestamp to 'max_start'
        start_idx = v.index[v['TS'] == closest_start_ts].tolist()[0]

        # Query pandas dataframe to find timestamps corresponding to
        # 'min_stop' without milliseconds (format: YYYY-MM-DD HH:MM:SS)
        stop_ts = v[v['TS'].str.match(min_stop.split('.')[0])]

        # Get only 'timestamp' field from the query result and convert it to list
        stop_ts = stop_ts['TS'].tolist()

        # Sort the list
        stop_ts.sort()

        # List to store differences between 'min_stop' and timestamps from 'stop_ts'
        ts_diff = []

        # Iterate over 'stop_ts' list
        for ts in stop_ts:
            # Store the abs difference between 'ts' and 'min_stop'
            ts_diff.append(abs(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f').timestamp() -
                               datetime.strptime(min_stop, '%Y-%m-%d %H:%M:%S.%f').timestamp()))

        # Find the timestamp closest to 'min_stop'
        closest_stop_ts = stop_ts[ts_diff.index(min(ts_diff))]

        # Get Pandas dataframe index of the closest timestamp to 'min_stop'
        stop_idx = v.index[v['TS'] == closest_stop_ts].tolist()[0]

        # Update data dictionary with a new truncated version of Pandas dataframe
        data[k] = v.truncate(before=start_idx, after=stop_idx)

        # Reset index in the dataframe to start from 0
        data[k].reset_index(drop=True, inplace=True)

    return data


# Wrapper to load and align data in one go
def load_and_align_data(root_path, sensor_type):
    return align_data_ts(read_data_files(root_path, sensor_type))
