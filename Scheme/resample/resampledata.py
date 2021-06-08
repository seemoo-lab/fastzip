import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial
from const.globconst import * 


def find_num_samples(timestamps, idx, cur_second, fs):
    # We assume that we have a normal second, containing fs samples
    n_samples = fs

    # Check if we are still in the current second or already in the next second
    if timestamps[idx + n_samples - 1].split('.')[0] == cur_second:
        # Iterate forward
        for i in range(idx + n_samples, len(timestamps)):
            # Check if current second in the i-th timestamp
            if cur_second in timestamps[i]:
                # Increment number of samples
                n_samples += 1
            else:
                break
    else:
        # Iterate backwards
        for i in range(idx + n_samples - 1, idx - 1, -1):
            # Check if current second in the i-th timestamp
            if cur_second not in timestamps[i]:
                # Decrement number of samples
                n_samples += -1
            else:
                break

    return n_samples


# Convert data from pandas data frame to the dict of the following structure: 'idx': (X, Y, Z, TS),
# we are also returning timestamps
def from_df_to_dict(df):
    # Case for acc, gyr, mag
    if len(df.columns) == 4:
        return dict(zip(np.arange(len(df)), list(zip(df['X'], df['Y'], df['Z'], df['TS'])))), df['TS'].values.tolist()

    # Case for bar
    elif len(df.columns) == 2:
        return dict(zip(np.arange(len(df)), list(zip(df['V1'], df['TS'])))), df['TS'].values.tolist()

    else:
        print('from_df_to_dict: only 1 (bar) or 3 (acc, gyr, mag) column input data formats are suppored!')
        return

    # Convert data from pandas data frame to the dict of the following structure: 'TS': (X, Y, Z)


def from_dict_to_df(data_dict):
    # Lists to store individual axis values and timestamps
    x = []
    y = []
    z = []
    ts = []

    # Get data values
    values = list(data_dict.values())

    # Check if we have acc, gyr, mag or bar
    if len(values[0]) == 4:

        # Pack data values to individual lists
        for val in values:
            x.append(val[0])
            y.append(val[1])
            z.append(val[2])
            ts.append(val[3])

        return pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'TS': ts}, columns=['X', 'Y', 'Z', 'TS'])

    elif len(values[0]) == 2:

        # Pack data values to individual lists
        for val in values:
            x.append(val[0])
            ts.append(val[1])

        return pd.DataFrame({'V1': x, 'TS': ts}, columns=['V1', 'TS'])
    else:
        print('from_dict_to_df: only 1 (bar) or 3 (acc, gyr, mag) column input data formats are suppored!')
        return


def generate_samples(data, sample_timestamps):
    # Newly generated samples
    gen_samples = {}

    # Iterate over timestamps of the samples between which a new sample will be inserted
    for sample_ts in sample_timestamps:

        # Sanity check: input indices must be successive
        if abs(sample_ts[0] - sample_ts[1]) != 1:
            print('generate_samples: indices %d and %d are not successive, exiting...')
            sys.exit(0)

        # Get samples corresponding to timestamps in the tuple
        sample1 = data[sample_ts[0]]
        sample2 = data[sample_ts[1]]

        # Add new samples to a dictionary
        if len(sample1) == 2:
            gen_samples[np.mean([sample_ts[0], sample_ts[1]])] = (np.mean([sample1[0], sample2[0]]),
                                                                  get_new_timestamp(sample2[1], sample1[1]))

        elif len(sample1) == 4:
            gen_samples[np.mean([sample_ts[0], sample_ts[1]])] = (np.mean([sample1[0], sample2[0]]),
                                                                  np.mean([sample1[1], sample2[1]]),
                                                                  np.mean([sample1[2], sample2[2]]),
                                                                  get_new_timestamp(sample2[3], sample1[3]))
        else:
            print('generate_samples: Only one or three dimensional data is allowed!')

    return gen_samples


def get_new_timestamp(ts2, ts1):
    # Find delta between two sample timestamps
    dt_later = datetime.strptime(ts2, '%Y-%m-%d %H:%M:%S.%f')
    t_delta = dt_later - datetime.strptime(ts1, '%Y-%m-%d %H:%M:%S.%f')

    # Sanity check, maybe we have some weird samples
    if t_delta.seconds != 0:
        print('get_new_timestamp: Difference between timestamps "%s" and "%s" should be below one second, exiting...' %
              (ts1, ts2))
        sys.exit(0)

    return (dt_later - timedelta(microseconds=int(t_delta.microseconds / 2))).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def get_timestamp_seconds(timestamps):
    # List containing only timestamp seconds
    ts_seconds = []

    # Construct the list of timestamps without milliseconds
    for ts in timestamps:
        ts_seconds.append(ts.split('.')[0])

    # Get seconds only list by removing duplicates
    ts_seconds = list(set(ts_seconds))

    # Sort the list just in case
    ts_seconds.sort()

    return ts_seconds


def process_samples_per_second(data, fs=100, sampling_strat={}, sensors=[]):
    # Try-catch clause for debugging, prints out errors in multiprocessing
    try:
        # Extract a key-value pair from the data tuple: (key, pandas dataframe)
        k = data[0]
        v = data[1]

        # If the subset of sensors is provided use it, otherwise ignore
        if not sensors:
            sensors = [k]

        # Check if we need to consider only a subset of sensors
        if k in sensors:

            # Convert pandas dataframe to dict
            v, timestamps = from_df_to_dict(v)

            # List containing seconds of timestamps only
            ts_seconds = get_timestamp_seconds(timestamps)

            # Get timestamps for the first and last measurements without milliseconds (format: YYYY-MM-DD HH:MM:SS)
            ts = datetime.strptime(timestamps[0].split('.')[0], '%Y-%m-%d %H:%M:%S').timestamp()
            ts_end = datetime.strptime(timestamps[-1].split('.')[0], '%Y-%m-%d %H:%M:%S').timestamp()

            # Dictionary storing number of samples per second
            samples_per_sec = {}

            # List of sample indices to be dropped
            downsample_idx = []

            # List containing samples to be inserted in the data
            upsample = []

            # Index corresponding to the first sample of a second
            idx = 0

            # Iterate over timestamps
            while ts <= ts_end:

                # Convert a timestamp of the current second to date string (format: YYYY-MM-DD HH:MM:SS)
                cur_second = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

                # We assume the worst case: a second is missing
                n_samples = 0

                # Get a number of samples in the current second
                if len(timestamps[idx:]) < fs:
                    # Case corresponding to the end of recording (not enough samples)
                    n_samples = len(timestamps[idx:])
                else:
                    # Only check the number of samples in a second if it is present
                    if cur_second in ts_seconds:
                        # Case corresponding to the normal second (enough samples)
                        n_samples = find_num_samples(timestamps, idx, cur_second, fs)

                # Check if we need to up or downsample or just find number of samples in each second
                if sampling_strat:

                    # Check if the sensor needs to be up or downsampled
                    if k not in sampling_strat:
                        return k, from_dict_to_df(v)

                    if sampling_strat[k][0] == 'up':

                        if n_samples == fs - 1 and sampling_strat[k][1][0] == 1:
                            # Generate a sample between 1st and 2nd samples of the current second
                            upsample.append(generate_samples(v, [(idx, idx + 1)]))

                        if n_samples == fs - 2 and sampling_strat[k][1][1] == 1:
                            # Generate a sample between 1st and 2nd samples of the current second,
                            # and between the penultimate and last samples of the current second
                            upsample.append(generate_samples(v, [(idx, idx + 1),
                                                                 (idx + n_samples - 2, idx + n_samples - 1)]))

                        if n_samples == fs - 3 and sampling_strat[k][1][2] == 1:
                            # Insert a sample between 1st and 2nd samples of the current second,
                            # between the middle sample of sec and its preceeding sample
                            # and between the penultimate and last samples of the current second
                            upsample.append(generate_samples(v, [(idx, idx + 1),
                                                                 (idx + int(n_samples / 2) - 1,
                                                                  idx + int(n_samples / 2)),
                                                                 (idx + n_samples - 2, idx + n_samples - 1)]))

                    elif sampling_strat[k][0] == 'down':

                        if n_samples == fs + 1 and sampling_strat[k][1][0] == 1:
                            # Save index of the 1st sample of the second
                            downsample_idx.append(idx)

                        if n_samples == fs + 2 and sampling_strat[k][1][1] == 1:
                            # Save indices of 1st and last samples of the second
                            downsample_idx.append(idx)
                            downsample_idx.append(idx + n_samples - 1)

                        if n_samples == fs + 3 and sampling_strat[k][1][2] == 1:
                            # Save indices of 1st, middle and last samples of the second
                            downsample_idx.append(idx)
                            downsample_idx.append(idx + int(n_samples / 2))
                            downsample_idx.append(idx + n_samples - 1)
                    else:
                        print('process_samples_per_second: Only "up" or "down" sampling is possible, exiting...')
                        sys.exit(0)
                else:
                    # Store number of samples per second
                    samples_per_sec[cur_second] = n_samples

                # Increment time by one second
                ts += 1

                # Update number of samples
                idx += n_samples

            # Return cases
            if sampling_strat:

                # Check if we need to downsample
                if downsample_idx:

                    # Downsample data
                    for d_idx in downsample_idx:
                        if d_idx in v:
                            del v[d_idx]

                    # Get back to pandas dataframe and return it together with a key
                    return k, from_dict_to_df(v)
                else:
                    # Upsample data
                    for ups in upsample:
                        for k1, v1 in ups.items():
                            v[k1] = v1

                    # Get back to pandas dataframe and return it together with a key
                    return k, from_dict_to_df(dict(sorted(v.items())))
            else:
                # Return number of samples per second for a sensor
                return k, dict(sorted(samples_per_sec.items()))

        return

    except Exception as e:
        print(e)


def process_samples(data, fs, sampling_strat={}, sensors=[]):
    # List to store output of the parallel processing
    sample_distributions = []

    # Check if sensors has been set, otherwise use all the sensors
    if not sensors:
        sensors = list(data.keys())

    # Initiate a pool of workers, use all available cores
    pool = Pool(processes=cpu_count(), maxtasksperchild=1)

    # Use partial to pass static parameters
    func = partial(process_samples_per_second, fs=fs, sampling_strat=sampling_strat, sensors=sensors)

    # Let the workers do their job, we convert data dict to list of tuples
    processed_samples = pool.map(func, list(data.items()))

    # Wait for processes to terminate
    pool.close()
    pool.join()

    # Convert the results back to dictionary, remove None values in case of subset and return them
    return dict([dist for dist in processed_samples if dist is not None])


def save_resampled_data(resampled_data, sensor_type, root_path):
    # Sanity checks on 'root_path'
    # 1. Check if 'root_path' exists
    if not os.path.exists(root_path):
        print('save_resampled_data: root_path="%s" does not exist!' % root_path)
        exit(0)

    #  2. Stupid Windows can do that
    if root_path[-1] == '\\':
        root_path = root_path[0:-1] + '/'

    # 3. Add slash to 'root_path' in case it is missing
    if root_path[-1] != '/':
        root_path = root_path + '/'

    # Check if sensor type is supported
    if sensor_type not in SENSOR_TYPES:
        print('save_resampled_data: "%s" sensor is not suppored, exiting...')
        sys.exit(0)

    # Iterate over resampled data
    for k, v in sorted(resampled_data.items()):
        # Construct save path
        save_path = root_path + 'Sensor-' + k + '/sensors/'

        # Create folder to save files
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save pandas dataframe to a file and gzip compress it
        v.to_csv(save_path + sensor_type + 'Data.txt.gz', sep=' ', float_format='%g', header=True, index=False,
                 compression='gzip')