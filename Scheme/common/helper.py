import numpy as np
from json import dumps
from json import loads
from glob import glob
import gzip
import os
import re
import math
from datetime import datetime, timedelta
from process.normdata import normalize_signal
from const.globconst import *
from const.activityconst import *
from const.fpsconst import *


def get_sensors_len(data, sensors, fs=0):
    # List to store sensors lengths
    sensors_len = []
    
    # Iterate over sensors
    for k,v in sorted(data.items()):
        
        # Consider only relevant sensors
        if k not in sensors:
            continue
            
        # If sampling rate is set return length in seconds, otherwise return number of samples
        if fs:
            sensors_len.append(int(len(v) / fs))
        else:
            sensors_len.append(len(v))

    return sensors_len


def save_json_file(data, filepath, filename, compress=False):
    # Check if a file path exists
    if not os.path.exists(filepath):
        print('save_json_file: %s is not a valid path!' % filepath)
        return
    
    # Check if we need to gzip json file or not
    if compress:
        # Save data to JSON file and gzip it
        with gzip.open(filepath + '/' + filename + '.json.gz', 'wt', encoding="utf-8") as f:
             f.write(dumps(data, indent=4, sort_keys=True)) # change sort_keys to True
    else:
        # Save data to a JSON file
        with open(filepath + '/' + filename + '.json', 'w') as f:
            f.write(dumps(data, indent=4, sort_keys=True)) # change sort_keys to True

        
def check_tframe_boundaries(tframe):
    # Check if tframe is of valid type
    if not ((isinstance(tframe, tuple) or isinstance(tframe, list)) and len(tframe) == 2):
        print('check_tframe_boundaries: %s must be either a list or tuple of length 2!' % (tframe,))
        return
    
    # Check boundaries
    if isinstance(tframe[0], int) and isinstance(tframe[1], int):
        if tframe[1] < 0:
            if tframe[1] != -1:
                print('check_tframe_boundaries: provide x2 = -1 to get the full signal: [0, len(sig))')
                return
            
            if not tframe[0] >= 0:
                print('check_tframe_boundaries: if x2 == -1, then x1(=%d) must be >= 0!' % (tframe[0],))
                return
        elif tframe[1] == 0:
            print('check_tframe_boundaries: second value in tuple/list cannot be zero!')
            return
        else:
            if tframe[0] < 0:
                print('check_tframe_boundaries: x1(=%d) must be > 0!' % (tframe[0],))
                return
            
            if not tframe[1] > tframe[0]:
                print('check_tframe_boundaries: x2(=%d) must be > x1(=%d)!' % (tframe[1], tframe[0],))
                return
    else:
        print('check_tframe_boundaries: %s must only contain integers!' % (tframe,))
        return
    
    return 1


def get_std_and_range(sig):
    # Check if sig is non-zero
    if len(sig) == 0:
        print('get_std_and_range: signal must have non-zero length!')
        return
    
    return np.std(sig), np.amax(sig) - np.amin(sig)


def choose_bar_norm(sig):
    # Check if sig is non-zero
    if len(sig) == 0:
        print('choose_bar_norm: signal must have non-zero length!')
        return
    
    # Normalize signal by subtracting mean
    mean_norm = normalize_signal(sig, 'meansub')
    
    # Perform Z-score normalization
    zscore_norm = normalize_signal(sig, 'zscore')
    
    # Get the range of meansub normalized bar
    _, rng1 = get_std_and_range(mean_norm)
    
    # Get the range of Z-score nomralized bar
    _, rng2 = get_std_and_range(zscore_norm)
    
    if rng1 > rng2:
        return mean_norm
    else:
        return zscore_norm


def compute_sig_diff(sig1, sig2, distance_type='rmse'):
    # Check if signals have non-zero length
    if len(sig1) != 0:
        # Check if signals have similar length
        if len(sig1) != len(sig2):
            print('compute_sig_diff: two signals must have equal length!')
            return -1
    else:
        print('compute_sig_diff: signals must have non-zero length!')
        return -1
    
    # Check if distance type is valid
    if distance_type == 'rmse':
        # Compute root mean square error
        return np.sqrt(np.sum((sig1 - sig2) ** 2) / len(sig1))
        
    elif distance_type == 'euclid':
        # Compute Euclidian distance
        return np.linalg.norm(sig1 - sig2) ** 2 / (np.linalg.norm(sig1) * np.linalg.norm(sig2))
        
    else:
        print('compute_sig_diff: distance type can only be "rmse" or "euclid"!')
        return -1

    return -1


def compute_hamming_dist(bin_fp1, bin_fp2):
    # Check if two fingerprints have the same length
    if len(bin_fp1) != len(bin_fp2):
        print('compute_hamming_dist: binary fingerprints must have the same length!')
        return
    
    # Hamming distance
    ham_dist = 0

    # Iterate over fingerprints
    for i in range(0, len(bin_fp1)):
        # If mismatch increase the distance
        if bin_fp1[i] != bin_fp2[i]:
            ham_dist += 1
    
    # Similarity between two fingerprints in percent
    fp_sim = (len(bin_fp1) - ham_dist) / len(bin_fp1) * 100
    
    return ham_dist, fp_sim


def del_dict_elem(in_dict, drop_keys):
    # Drop elements from dict 
    for dk in drop_keys:
        if dk in in_dict:
            del in_dict[dk]
    
    return in_dict


def ecdf(x):
    ''' Return empirical cdf of x '''
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def get_fusion_plot_settings(plot_setup):
    # Line and markers settings
    line_marker_settings = {'acc_v': ['-.', 'x'], 'acc_h': ['-', 'D'], 'gyrW': ['--', '^'], 'bar': [[1, 1], 'h']}
    
    # Color settings
    color_settings = {'city': '#ff7f0e', 'country': '#2ca02c', 'highway': '#d62728', 'parking': '#9467bd', 'full': '#1f77b4'}
    
    # Iterate over plot setup
    for k,v in plot_setup.items():
        line_marker_settings[k].append(color_settings[v])
        
    return list(line_marker_settings.values())
    

def get_proximate_chunks(idx, ts, vic=9):
    # Get starting point
    start = idx - vic
    
    if start < 0:
        start = 0
    
    # Get ending point
    end = idx + vic + 1
    
    if end > len(ts) - 1:
        end = len(ts) - 1
    
    return ts[start:end]


def repack_padv_results(in_res, sensor_types):
    # Dictionary to store output results
    out_res = {}
    
    # Initialize helper dicts
    error_rate = {st: [] for st in sensor_types}
    adv_st_sim = {st: [] for st in sensor_types}
    
    # Iterate over input results
    for k,v in in_res.items():
        # Iterate over sensor types
        for st in sensor_types: 
            error_rate[st].append(v[st]['res'])
            adv_st_sim[st].append(v[st]['match'])
    
    # Iterate over sensor types: populate out_dict
    for st in sensor_types:
        # Get similarity threshold
        if st == 'acc_v':
            sim_thr = SIM_THR_ACC_V
        elif st == 'acc_h':
            sim_thr = SIM_THR_ACC_H
        elif st == 'gyrW':
            sim_thr = SIM_THR_GYR
        elif st == 'bar':
            sim_thr = SIM_THR_BAR
        
        out_res[st] = {'matched_far': np.count_nonzero(np.array(adv_st_sim[st]) >= sim_thr) / len(adv_st_sim[st]), 
                       'far': np.count_nonzero(np.array(error_rate[st]) == 1) / len(error_rate[st])}

    # Add length data
    out_res['far_chunks'] = len(in_res)
    
    return out_res
    

def reverse_str(s):
    return s.split('_')[1] + '-' + s.split('_')[0]


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def compute_avg_error_rate(error_rate):
    # List to store error rates
    avg_error_rate = []
    
    # Get error_rate keys
    sensor_pairs = list(error_rate.keys())
    
    # We need to compute error rates for CAR1 and CAR2 separately
    if len(sensor_pairs) > 10:
        # Adjust sensor pairs (CAR2), otherwise CAR1
        sensor_pairs = sensor_pairs[10:]
    
    # Iterate over error rates
    for sp in sensor_pairs:
        # Check we are dealing with TAR or FAR
        if 'tar' in error_rate[sp]:
            avg_error_rate.append(error_rate[sp]['tar'])
        elif 'far' in error_rate[sp]:
            avg_error_rate.append(error_rate[sp]['far'])
    
    # Remove 'n/a' elements from the list
    avg_error_rate = [er for er in avg_error_rate if er != 'n/a']

    return np.mean(np.array(avg_error_rate))
 
    
def get_error_rate_grid(error_rate):
    # Get error_rate keys
    sensor_pairs = list(error_rate.keys())
    
    # Create error rate grid
    error_rate_grid = np.full((5,5), -100.0)
    
    # We need to compute error rates for CAR1 and CAR2 separately
    if len(sensor_pairs) > 10:
        # Adjust sensor pairs (CAR2), otherwise CAR1
        sensor_pairs = sensor_pairs[10:]
    
    # Labels storing sensor names
    labels = []
    
    # Iterate over error rates
    for sp in sensor_pairs:
        # Indices
        ids = sp.split('_')
        
        # Add labels
        labels.append(ids[0])
        
        # Add values to error rate grid
        if 'tar' in error_rate[sp]:
            # Check if we have 'n/a' instances
            if isinstance(error_rate[sp]['tar'], str):
                error_rate_grid[(int(ids[0]) - 1) % 5, (int(ids[1]) - 1) % 5] = np.nan
            else:
                error_rate_grid[(int(ids[0]) - 1) % 5, (int(ids[1]) - 1) % 5] = error_rate[sp]['tar']
                
        elif 'far' in error_rate[sp]:
            # Check if we have 'n/a' instances
            if isinstance(error_rate[sp]['tar'], str):
                error_rate_grid[(int(ids[0]) - 1) % 5, (int(ids[1]) - 1) % 5] = np.nan
            else:
                error_rate_grid[(int(ids[0]) - 1) % 5, (int(ids[1]) - 1) % 5] = error_rate[sp]['far']
    
    # Remove duplicates from labels
    labels = list(set(labels))
    
    # Sort labels
    labels.sort()
    
    # One sensor might be missing
    if set(labels).issubset(CAR1):
        labels = CAR1
    elif set(labels).issubset(CAR2):
        labels = CAR2
    else:
        print('get_error_rate_grid: inconsistent labels "%s", investiagte!' % (labels,))
        return
    
    # Make the grid symmetric
    i_lower = np.tril_indices(5, -1)
    error_rate_grid[i_lower] = error_rate_grid.T[i_lower]  
    
    # Convert the grid from percentages to abs values
    error_rate_grid = error_rate_grid / 100
    
    return error_rate_grid, labels


def get_cached_error_rates(filepath, rpwc=False, powerful=False):
    # List subdirectories under provided filepath + sensor_type
    st_subdir = glob(filepath + '/*/')
    
    # Sort st_subdir
    st_subdir.sort()
    
    # Dictionaries to store results for plotting (we do this per car)
    car1_plot = {}
    car2_plot = {}
    
    # Iterate over subdirectories
    for st_sd in st_subdir:
        # Get sensor type
        sensor_type = st_sd.split(filepath)[1].split('/')[1]
        
        # Read json files under st_sd
        json_files = glob(st_sd + '*.json', recursive=True)
        
        # Sort json_files
        json_files.sort()
        
        # Case for replay with compensation
        if rpwc:
            # Subscenarios
            subs = ['full', 'city', 'country', 'highway', 'parking']
            
            # Dictionaries to store error rates per sensor type
            st_err_car1 = {s: [] for s in subs}
            st_err_car2 = {s: [] for s in subs}
           
            # Iterate over replay setups: sensors from the 2nd replayed to the 1st and vice versa
            for replay_setup in ['car1-2', 'car2-1']:
                # Adjust path for car1-2 and car2-1 cases
                if replay_setup in json_files[0]:
                    json_file = json_files[0]
                else:
                    # If json_file does not match with the car we need another file
                    diff_setup = ['car1-2', 'car2-1']
                    diff_setup.remove(replay_setup)
                    
                    json_file = json_files[0].replace(diff_setup[0], replay_setup)
                    
                # Check if we are dealing with sim or diff experiments
                if DIFF_NON_ADV in json_file or DIFF_ADV in json_file:
                    if DIFF_NON_ADV in json_file:
                        parking_file = json_file.replace(DIFF_NON_ADV, 'diff-park')
                    else:
                        parking_file = json_file.replace(DIFF_ADV, 'diff-park')

                    # Load two json files
                    with open(json_file, 'r') as f:
                        results = loads(f.read())['results']

                    with open(parking_file, 'r') as f:
                        park_res = loads(f.read())['results']
                    
                    # Merge contents of loaded files
                    for k,v in sorted(results.items()):
                        results[k].update(park_res[k])
                    
                else:
                    # Load json file
                    with open(json_file, 'r') as f:
                        results = loads(f.read())['results']
                
                # Iterate over results and populate st_err_carX dicts
                for k,v in sorted(results.items()):
                    # Get sensor pair
                    sensors = k.split('_')

                    # Iterate over subscenarios
                    for s in subs:
                        # Store error rates for car1 and car2                          
                        if sensors[0] in CAR1:
                            # Store FAR
                            if isinstance(v[s]['far'], float):
                                st_err_car1[s].append(v[s]['far'])
                                
                        elif sensors[0] in CAR2:
                            # Store FAR
                            if isinstance(v[s]['far'], float):
                                st_err_car2[s].append(v[s]['far'])
                        
                        else:
                            # Must never get here but just in case
                            print('plot_error_rates: invalid sensor number, the only following numbers are possible: %s!' % 
                                  (list(itertools.chain.from_iterable([CAR1, CAR2])),)) 
                            return

            # Compute average FAR
            for s in subs:
                # Check if list is non-empty
                if st_err_car1[s]:
                    st_err_car1[s] = np.array(st_err_car1[s]) / 100
                else:
                    st_err_car1[s] = 0

                # Check if list is non-empty
                if st_err_car2[s]:
                    st_err_car2[s] = np.array(st_err_car2[s]) / 100
                else:
                    st_err_car2[s] = 0

            # Add results to carX_plot dicts
            car1_plot[sensor_type] = st_err_car1
            car2_plot[sensor_type] = st_err_car2
            
            continue
        
        # Dictionaries to store error rates per sensor type
        st_err_car1 = {}
        st_err_car2 = {}

        # Iterate over json files
        for json_file in json_files:
            # Get file name 
            regex = re.escape(st_sd) + r'(.*).json'
            match = re.search(regex, json_file)

            # If there is no match - exit
            if not match:
                print('get_cached_error_rates: no match for the file name %s using regex %s!' % (json_file, regex))
                return

            # Current file name
            filename = match.group(1)
            
            # Load json file
            with open(json_file, 'r') as f:
                results = loads(f.read())['results']
                
            # Lists to store error rates
            err_car1 = []
            err_car2 = []
            
            # Iterate over results
            for k,v in sorted(results.items()):
                # Get sensor pair
                sensors = k.split('_')
                
                # Store error rates for car1 and car2                          
                if sensors[0] in CAR1:
                    # Check if we are dealing with the powerful adversary case
                    if powerful:
                        err_car1.append(v)
                        continue
                    
                    # Check if we are dealing with benign case, adversarial case or pairing time
                    if 'tar' in v:
                        # Store TAR
                        if isinstance(v['tar'], float):
                            err_car1.append(v['tar'])
                    
                    elif 'far' in v:
                        # Store FAR
                        if isinstance(v['far'], float):
                            err_car1.append(v['far'])
                            
                    elif 'pairing_time_sec' in v:
                        # Store pairing time
                        if isinstance(v['pairing_time_sec'], float):
                            if not np.isnan(v['pairing_time_sec']):
                                err_car1.append(v['pairing_time_sec'] * 100)
                            
                elif sensors[0] in CAR2:
                    # Check if we are dealing with the powerful adversary case
                    if powerful:
                        err_car2.append(v)
                        continue
                    
                    # Check if we are dealing with benign case, adversarial case or pairing time
                    if 'tar' in v:
                        # Store TAR
                        if isinstance(v['tar'], float):
                            err_car2.append(v['tar'])
                    
                    elif 'far' in v:
                        # Store FAR
                        if isinstance(v['far'], float):
                            err_car2.append(v['far'])
                            
                    elif 'pairing_time_sec' in v:
                        # Store pairing time
                        if isinstance(v['pairing_time_sec'], float):
                            if not np.isnan(v['pairing_time_sec']):
                                err_car2.append(v['pairing_time_sec'] * 100)
                
                else:
                    # Must never get here but just in case
                    print('plot_error_rates: invalid sensor number, the only following numbers are possible: %s!' % 
                          (list(itertools.chain.from_iterable([CAR1, CAR2])),)) 
                    return
            
            # Add error rates to st_err_carX dictionaries
            if powerful:
                st_err_car1[filename] = err_car1
                st_err_car2[filename] = err_car2
            else:
                # Check if the list is non-empty
                if err_car1:
                    st_err_car1[filename] = np.array(err_car1) / 100
                else:
                    st_err_car1[filename] = 0

                # Check if the list is non-empty
                if err_car2: 
                    st_err_car2[filename] = np.array(err_car2) / 100
                else:
                    st_err_car2[filename] = 0
        
        # Store modality error rates in carX_plot
        car1_plot[sensor_type] = st_err_car1
        car2_plot[sensor_type] = st_err_car2
        
    return car1_plot, car2_plot


def process_powerful_adv(plot_dict, plot_setup):
    # Output dicts for plotting results
    car_plot_indiv = {}
    car_plot_fused = {}
    
    # Initialize car_plot_indiv
    for k,v in sorted(plot_setup.items()):
        car_plot_indiv[k] = {v: {}}
        
    # Initialize car_plot_fused
    for fc in FUSION_CONFIG:
        f_key = ''
        
        # Iterate over sensor types
        for st in fc:
            if f_key:
                f_key += '-' + st
            else:
                f_key += st
         
        # Initialize dict
        car_plot_fused[f_key] = {scen: [] for scen in list(plot_setup.values())}

    # Iterate over plot setup
    for k,v in sorted(plot_setup.items()):
        # Iterate over loaded json data
        for k1,v1 in sorted(plot_dict.items()):
            
            # Check if k is k1
            if k in k1:
                # Store matched and resulting FARs
                matched_far = []
                fars = []

                # Iterate over error rates
                for far in v1[v]:
                    matched_far.append(far[k]['matched_far'])
                    fars.append(far[k]['far'])

                # Key is the mean FAR
                car_plot_indiv[k][v][np.mean(np.array(matched_far))] = np.array(matched_far)
                car_plot_fused[k1][v] = np.array(fars)

        # Pick higest possible FAR
        car_plot_indiv[k][v] = car_plot_indiv[k][v][max(list(car_plot_indiv[k][v].keys()))]

    return car_plot_indiv, car_plot_fused
        

def compute_percentile(sig, upper_centile, lower_centile):
    # Check if sig is non-zero
    if len(sig) == 0:
        print('compute_percentile: signal must have non-zero length!')
        return
    
    # Check that percentiels are provided as floats between 0 and 1
    if isinstance(upper_centile, float) and isinstance(lower_centile, float):
        if not 0 < upper_centile < 1:
            print('compute_percentile: upper percentile must be a float between 0 and 1!')
            return
            
        if not 0 < lower_centile < 1:
            print('compute_percentile: lower percentile must be a float between 0 and 1!')
            return
    else:
        print('compute_percentile: upper and lower percentiles must be provided as floats between 0 and 1!')
        return
    
    # Compute emperical CDF of the signal
    sig_cdf = ecdf(sig)
    
    # Take first existing value above the upper percentile margin
    upper_limit = sig_cdf[0][np.where(sig_cdf[1] > upper_centile)[0][0]]

    # Take last existing value below the lower percentile margin
    lower_limit = sig_cdf[0][np.where(sig_cdf[1] < lower_centile)[0][-1]]
    
    return upper_limit, lower_limit
    

def sig_ratio_by_mean(sig):
    # Check if sig is non-zero
    if len(sig) == 0:
        print('sig_ratio_by_mean: signal must have non-zero length!')
        return
    
    # Compute signal's mean
    mean = np.mean(sig)
    
    # Compute empirical density of a signal
    cdf = ecdf(sig)
    
    # Take first existing value above the 95th percentile margin
    upper_limit = cdf[0][np.where(cdf[1] > 0.95)[0][0]]
        
    # Take last existing value below the 5th percentile margin
    lower_limit = cdf[0][np.where(cdf[1] < 0.05)[0][-1]]
    
    # Counters for samples located above or below the mean 
    above_mean_count = 0
    below_mean_count = 0
    
    # Iterate over a signal and count samples located above or below mean, within a range of upper and lower limits
    for s in sig:
        if mean < s < upper_limit:
            above_mean_count += 1
        
        if lower_limit < s < mean:
            below_mean_count += 1
    
    # Compute above and below ratios in percent
    ratio_above = above_mean_count / (above_mean_count + below_mean_count) * 100
    ratio_below = below_mean_count / (above_mean_count + below_mean_count) * 100
    
    return mean, ratio_above, ratio_below
    

def idx_to_str(idx):
    # Make it 01, 02, etc.
    if idx < 10:
        return '0' + str(idx)

    return str(idx)


def pad_with_zeros(data, pad_len):
    # Check if we have 1D or 2D array
    if len(data.shape) > 1:
        # Initialzie padded data
        pad_data = np.zeros((len(data) + pad_len, data.shape[1]))
    else:
        # Initialzie padded data
        pad_data = np.zeros(len(data) + pad_len)
    
    # Index
    idx = 0
    
    # Copy original data to pad_data
    for d in data:
        pad_data[idx] = d
        
        idx += 1
        
    return pad_data


def check_if_pos_numbers(data):
    # Iterate over data
    for d in data:
        # Check if we have other elements other than int or float numbers
        if isinstance(d, int) or isinstance(d, float):
            # Check if int or float number is positive
            if d <= 0:
                return False
        else:
            return False

    return True


def fps_str_to_list(fps_str, fp_len):
    # List of binary fingerprints to be returned
    fps_list = []
    
    # Check that fps_str contains equal fingerprints
    if len(fps_str) % fp_len != 0:
        print('fps_str_to_list: input string must contain fingerprints of equal length = %d bits!' % fp_len)
        return
    
    # Iterate over binary string and split it back to individual fingerprints stored in the list
    for i in range(0, int(len(fps_str) / fp_len)):
        fps_list.append(fps_str[i * fp_len:(i + 1) * fp_len])
        
    return fps_list


def get_ewma_alpha(sensor_type):
    # Set the EWMA alpha param
    if sensor_type == 'acc_v':
        return EWMA_ACC_V
    elif sensor_type == 'acc_h':
        return EWMA_ACC_H
    
    return 0


def load_gz_json_file(gz_json_file):
    # Load *.JSON.GZ file and extract results data
    with gzip.open(gz_json_file, 'rt') as f:
        results = loads(f.read())
        metadata = results['metadata']
        results = results['results']

    return results, metadata


def parse_timestamp(timestamp):
    # Result key must be a string
    if isinstance(timestamp, str):
        # Split timestamp and seconds
        parsed_ts = timestamp.split(',')
        
        # Check if the split is valid
        if len(parsed_ts) != 2:
            print('parse_timestamp: timestamp splitted by "," must have two elements!')
            return
            
        parsed_ts = parsed_ts[1].split('->')
        
        # Check if the split is valid
        if len(parsed_ts) != 2:
            print('parse_timestamp: timestamp splitted by "->" must have two elements!')
            return
        
        return int(parsed_ts[0]), int(parsed_ts[1])
        
    else:
        print('parse_timestamp: timestamp must be a string!')
        return
    
    return
 
    
def construct_bar_ts(in_ts, ts_level=0):
    # Check which level of timestamps we are dealing with 0 or 1
    if ts_level == 0:
        # Get datetime string shifted by 10 seconds back from in_ts
        dt_str = (datetime.strptime(in_ts.split(',')[0], '%Y-%m-%d %H:%M:%S') - timedelta(seconds=10)).strftime('%Y-%m-%d %H:%M:%S')

        # Parse time in seconds of the chunks
        parsed_ts = parse_timestamp(in_ts)
        
        return dt_str + ',' + str(parsed_ts[0] - 10) + '->' + str(parsed_ts[1])
    
    elif ts_level == 1:
        # Parse time in seconds of the chunks
        parsed_ts = parse_timestamp(in_ts)

        return in_ts.split(',')[0] + ',' + str(parsed_ts[0]) + '->' + str(parsed_ts[1] + 10)
    else:
        print('construct_bar_ts: Timestamp level can only be 0 or 1!')
        return
    

def get_common_timestamps(results, ts_level=0):
    # Check if results if a dict of non-zero len
    if not (isinstance(results, dict) and len(results) > 0):
        print('get_common_timestamps: provided "results" must be a non-empty dictionary!')
        return
    
    # Get sensor types
    sensor_types = list(results.keys())
    
    # If bar is present drop it from sensor_types
    if 'bar' in sensor_types:
        sensor_types.remove('bar')
        
    # Sort sensor types
    sensor_types.sort()
    
    # Lengths of sensor type results
    st_lens = []
    
    # Iterate over results
    for k,v in sorted(results.items()):
        if k != 'bar':
            st_lens.append(len(v))

    # Sensor type containing most data
    max_st = sensor_types[st_lens.index(max(st_lens))]
    
    # Remove from sensor types sensor type containing most data
    del sensor_types[st_lens.index(max(st_lens))]
    
    # Store common timestamps
    common_ts = []
    
    # Iterate over results (using sensor type containing most data)
    for k, v in sorted(results[max_st].items()):
        # Index to track the number of modalities containing a specific timestamp
        ts_idx = 0
        
        # Check if timestamp 'k' exists in other modalities
        for st in sensor_types:
            if k in results[st]:
                ts_idx += 1
        
        # Add timestamp to common timestamps if it exists in all modalities
        if ts_idx == len(sensor_types):
            common_ts.append(k)
    
    # Get copy of common_ts
    common_ts_cpy = common_ts.copy()
    
    # Do further refinement of common_ts considering bar
    if 'bar' in results:
        # Get list of bar timestamps
        bar_ts = list(results['bar'].keys())
        
        # Iterate over common_ts
        for cts in common_ts:
            # Check if cts is contained in the bar data
            if not construct_bar_ts(cts, ts_level) in bar_ts:
                common_ts_cpy.remove(cts)
        
        # Update common_ts
        common_ts = common_ts_cpy
    
    return common_ts   


def process_json_file(json_file, action='', scenario=None, fuse=False):
    # 'Try' clause is used for catching errors
    try:
        # Get experiment name
        if action == 'keys' and 'powerful' in json_file:
            exp = json_file.split('powerful')[1].split('/')[1]
        else:
            exp = json_file.split(action)[1].split('/')[1]

        # Get file name 
        regex = re.escape(exp) + r'(?:/|\\)(.*)\.json.gz'
        match = re.search(regex, json_file)

        # If there is no match - exit
        if not match:
            print('process_json_file: no match for the file name %s using regex %s!' % (json_file, regex))
            return

        # Current file name
        filename = match.group(1).split('/')[-1]
        
        # Load file
        results, metadata = load_gz_json_file(json_file)
        
        # Extract data corresponding to specified scenario
        if scenario is not None:

            # Get sensor number
            sensor_num = filename.split('_')[0]

            # Check if sensor number is valid
            if sensor_num not in CAR1 + CAR2:
                print('process_json_file: invalid sensor number %s, should be one of the following:' % 
                      (sensor_num, CAR1 + CAR2))
                return

            # Select correct timeframe based on experiment and scenario
            if exp == SIM_NON_ADV:
                if scenario == 'city':
                    if sensor_num in CAR1:
                        tf = (SIM_CAR1_CITY, )
                    else:
                        tf = (SIM_CAR2_CITY, )

                elif scenario == 'country':
                    if sensor_num in CAR1:
                        tf = (SIM_CAR1_COUNTRY, )
                    else:
                        tf = (SIM_CAR2_COUNTRY, )

                elif scenario == 'highway':
                    if sensor_num in CAR1:
                        tf = (SIM_CAR1_HIGHWAY, )
                    else:
                        tf = (SIM_CAR2_HIGHWAY, )

                elif scenario == 'parking':
                    if sensor_num in CAR1:
                        tf = (SIM_PARKING1, )
                    else:
                        tf = (SIM_PARKING1, )

            elif exp == SIM_ADV:
                if scenario == 'city':
                    if sensor_num in CAR1:
                        tf = (SIM_CAR12_CITY1, SIM_CAR12_CITY2)
                    else:
                        tf = (SIM_CAR12_CITY1, SIM_CAR12_CITY2)

                elif scenario == 'country':
                    if sensor_num in CAR1:
                        tf = (SIM_CAR12_COUNTRY, )
                    else:
                        tf = (SIM_CAR12_COUNTRY, )

                elif scenario == 'highway':
                    if sensor_num in CAR1:
                        tf = (SIM_CAR12_HIGHWAY, )
                    else:
                        tf = (SIM_CAR12_HIGHWAY, )

                elif scenario == 'parking':
                    if sensor_num in CAR1:
                        tf = (SIM_PARKING2, )
                    else:
                        tf = (SIM_PARKING2, )

            elif exp == DIFF_PARK:
                if scenario == 'city':
                    tf = ([-1, -2], )
                elif scenario == 'country':
                    tf = ([-1, -2], )
                elif scenario == 'highway':
                    tf = ([-1, -2], )
                elif scenario == 'parking':
                    tf = (DIFF_PARKING1, DIFF_PARKING2)

            elif exp == DIFF_NON_ADV:
                if scenario == 'city':
                    if sensor_num in CAR1:
                        tf = (DIFF_CAR1_CITY, )
                    else:
                        tf = (DIFF_CAR2_CITY, )

                elif scenario == 'country':
                    if sensor_num in CAR1:
                        tf = (DIFF_CAR1_COUNTRY, )
                    else:
                        tf = (DIFF_CAR2_COUNTRY, )

                elif scenario == 'highway':
                    if sensor_num in CAR1:
                        tf = (DIFF_CAR1_HIGHWAY, )
                    else:
                        tf = (DIFF_CAR2_HIGHWAY, )
                        
                elif scenario == 'parking':
                    tf = ([-1, -2], )

            elif exp == DIFF_ADV:
                if scenario == 'city':
                    if sensor_num in CAR1:
                        tf = (DIFF_CAR12_CITY1, DIFF_CAR12_CITY2)
                    else:
                        tf = (DIFF_CAR12_CITY1, DIFF_CAR12_CITY2)

                elif scenario == 'country':
                    if sensor_num in CAR1:
                        tf = (DIFF_CAR12_COUNTRY, )
                    else:
                        tf = (DIFF_CAR12_COUNTRY, )

                elif scenario == 'highway':
                    if sensor_num in CAR1:
                        tf = (DIFF_CAR12_HIGHWAY, )
                    else:
                        tf = (DIFF_CAR12_HIGHWAY, )
                        
                elif scenario == 'parking':
                    tf = ([-1, -2], )
            
            else:
                print('process_json_file: unknown experiment name %s, exiting...' % (exp,))
                return

            # Replace -1 for the end of recording with the actual time in seconds
            if len(tf) == 1:
                if tf[0][1] == -1:
                    tf[0][1] = int(sorted(results.keys())[-1].split(',')[1].split('->')[1])

            elif len(tf) == 2:
                if tf[0][1] == -1:
                    tf[0][1] = int(sorted(results.keys())[-1].split(',')[1].split('->')[1])

                if tf[1][1] == -1:
                    tf[1][1] = int(sorted(results.keys())[-1].split(',')[1].split('->')[1])
            
            # Adjsut metadata
            if not 'powerful' in json_file:
                metadata = get_scenario_chunk_stat(json_file, action, tf, metadata)
            
            # Update results 
            if action == 'keys':
                results = get_scenario_data(results, tf, True)
                sensor_type = json_file.split(exp)[1].split('/')[1]
                
                return results, sensor_type, metadata
            else:
                results = get_scenario_data(results, tf)
        
        # Check if we need to return the data in the case of sensor fusion
        if fuse:
            # Get sensor type
            if action == 'benign' or action == 'replay':
                sensor_type = json_file.split(exp)[1].split('/')[1]
                
            elif action == 'baseline':
                # Get sensor type
                if json_file.count(exp) > 1:
                    sensor_type = json_file.split(exp)[json_file.count(exp)].split('/')[1]
                else:
                    sensor_type = json_file.split(exp)[1].split('/')[3]
                    
            elif action == 'keys':
                # Get sensor type
                sensor_type = json_file.split(exp)[1].split('/')[1]
                
                # Repack results
                res = {}
                
                for k,v in sorted(results.items()):
                    res[k] = v['01']['fp']
                
                return res, sensor_type, metadata
            
            return sensor_type, results, filename, metadata
        
        # Dictionary to store similarity results
        sim = {}
        
        # Iterate over results
        for k,v in sorted(results.items()):
            # Store similarity of chunks
            if action == 'benign':
                sim[k] = v['sim']

            elif action == 'baseline' or action == 'replay':
                # List to store similarity for a chunk over many chunks
                chunk_sim = []

                # Check if we have '01' or 'x01'
                if filename.split('_')[1] in v:
                    key = filename.split('_')[1]
                else:
                    key = 'x' + filename.split('_')[1]

                # Iterate over chunks
                for k1,v1 in sorted(v[key].items()):
                    # Add similarity to the list
                    chunk_sim.append(v1['sim'])

                # Convert list to numpy array
                sim[k] = np.array(chunk_sim)
        
        return sim, metadata, filename
        
    except Exception as e:
        print(e)


def get_scenario_chunk_stat(json_file, action, tf, metadata):
    # Sensor type
    sensor_type = ''
    
    # Find sensor type
    for st in ALLOWED_SENSORS:
        if st in json_file:
            sensor_type = st
            break
    
    # Get experiment name
    exp = json_file.split(action)[1].split('/')[1]

    # Get file name 
    regex = re.escape(exp) + r'(?:/|\\)(.*)\.json.gz'
    match = re.search(regex, json_file)

    # If there is no match - exit
    if not match:
        print('process_json_file: no match for the file name %s using regex %s!' % (json_file, regex))
        return

    # Current file name
    filename = match.group(1).split('/')[-1]

    # Replace action in the json file
    json_file = json_file.replace(action, 'AR') 
    
    # Adjust json_file path
    if 'car1' in json_file or 'car2' in json_file:
        json_file = re.sub(r'(?:/|\\)car[12]', '', json_file)
        
    elif 'silent' in json_file:
        json_file = re.sub(r'(?:/|\\)silent(.*)' + re.escape(json_file.split('silent')[1].split('/')[1]), '', json_file)
        json_file = re.sub(r'(?:/|\\)' + re.escape(filename.split('_')[0]), '', json_file, 1)
        
    elif 'moving' in json_file:
        json_file = re.sub(r'(?:/|\\)moving(.*)' + re.escape(json_file.split('moving')[1].split('/')[1]), '', json_file)
        json_file = re.sub(r'(?:/|\\)' + re.escape(filename.split('_')[0]), '', json_file, 1)
    
    # Small tweak for the replay case
    if action == 'replay':
        json_file = re.sub(r'(?:/|\\)' + re.escape(filename.split('_')[0]), '', json_file, 1)

    # Set defaults
    n_peaks_thr = 0
        
    # Activity recongition parameters
    if sensor_type == 'acc_v':
        p_thr = P_THR_ACC_V
        n_peaks_thr = NP_THR_ACC_V
        snr_thr = SNR_THR_ACC_V
    elif sensor_type == 'acc_h':
        p_thr = P_THR_ACC_H
        n_peaks_thr = NP_THR_ACC_H
        snr_thr = SNR_THR_ACC_H
    elif sensor_type == 'gyrW':
        p_thr = P_THR_GYR
        snr_thr = SNR_THR_GYR
    elif sensor_type == 'bar':
        p_thr = P_THR_BAR
        snr_thr = SNR_THR_BAR
    
    # Sensor idx
    s_idx = 0
    
    # Get sensors to iterate over
    sensors = filename.split('_')
    
    # Depending on the action we need to either iterate over two sensors or one
    if action == 'baseline' or action == 'replay':
        sensors = sensors[:-1]
    
    # For each sensor
    for s in sensors:
        # Number of chunks that pass activity recognition
        n_ar_chunks = 0
        
        # Replace filename 
        ar_json_file = json_file.replace(filename, s)
        
        # Load json file
        results, _ = load_gz_json_file(ar_json_file)
        
        # Update results
        results = get_scenario_data(results, tf)
        
        # Iterate over results
        for k,v in sorted(results.items()):
            # Check if chunk passes AR test
            if v['power_dB'] > p_thr and v['n_peaks'] >= n_peaks_thr and v['SNR'] > snr_thr:
                n_ar_chunks += 1
                
        # Adjust number of chunks that pass AR test in metadata
        if s_idx == 0:
            # For replay with compensation the structure is a bit different
            if action == 'keys':
                metadata['parameters']['data']['n_chunks_ar'] = n_ar_chunks
            else:
                metadata['parameters']['ar']['n_chunks1'] = n_ar_chunks
        else:
            # For replay with compensation the structure is a bit different (we should never get here with action == 'keys')
            if action == 'keys':
                metadata['parameters']['data']['n_chunks_ar'] = n_ar_chunks
            else:
                metadata['parameters']['ar']['n_chunks2'] = n_ar_chunks
        
        # Increment s_idx
        s_idx += 1
    
    # Adjust overall number of chunks per scenario
    if action == 'benign' or action == 'keys':
        metadata['parameters']['data']['n_chunks'] = len(results)
    
    elif action == 'baseline' or action == 'replay':
        metadata['parameters']['data']['n_chunks1'] = len(results)
    
    return metadata
    

def get_scenario_data(data, timeframe, rpwc=False):
    # Dictionary to store scenario data
    scenario_data = {}
    
    # Get necessary chunks based on timeframe
    if len(timeframe) == 1:
        # Iterate over data
        for k,v in sorted(data.items()):
            # Get chunk time in seconds
            chunk_time = k.split(',')[1].split('->')
            
            # Check if chunk_time is within a timeframe
            if int(chunk_time[0]) >= timeframe[0][0] and int(chunk_time[1]) <= timeframe[0][1]:
                if rpwc:
                    scenario_data[k] = v['01']['fp']
                else:
                    scenario_data[k] = v
            
    elif len(timeframe) == 2:
        # Iterate over data
        for k,v in sorted(data.items()):
            # Get chunk time in seconds
            chunk_time = k.split(',')[1].split('->')
            
            # Check if chunk_time is within a timeframe
            if (int(chunk_time[0]) >= timeframe[0][0] and int(chunk_time[1]) <= timeframe[0][1]) or \
            (int(chunk_time[0]) >= timeframe[1][0] and int(chunk_time[1]) <= timeframe[1][1]):
                if rpwc:
                    scenario_data[k] = v['01']['fp']
                else:
                    scenario_data[k] = v
    
    return scenario_data
       
    
def rpwc_wrapper(file_pair, scenario=None):
    
    # 'Try' clause is used for catching errors
    try:
        # Load two files extracting scenario-related data
        res1, _, md1 = process_json_file(file_pair[0], 'keys', scenario)
        res2, sensor_type, md2 = process_json_file(file_pair[1], 'keys', scenario)
        
        # Dictionary to store processed results
        res = {}
        
        # Dictionary to store metadata
        metadata = {}
        
        # Do the processing only if we have data
        if len(res1) > 0 and len(res2) > 0:
            # Process res1 and res2
            res = rpwc_processing(res1, res2)

        # Fill in metadata
        metadata['ar'] = {'n_chunks1': md1['parameters']['data']['n_chunks_ar'], 
                          'n_chunks2': md2['parameters']['data']['n_chunks_ar']}
        
        metadata['input'] = {'n_chunks1': md1['parameters']['data']['n_chunks'], 
                             'n_chunks2': md2['parameters']['data']['n_chunks']}
        
        return sensor_type, res, metadata
        
    except Exception as e:
        print(e)
        
        
def rpwc_processing(res1, res2):
    
    # Lists storing chunk timeframes for the 1st and 2nd sensor, e.g., 5->15
    tfs_s1 = []
    tfs_s2 = []
    
    # Drop timestamp from the key containting timestamp and timeframe
    for r1_key in list(sorted(res1.keys())):
        tfs_s1.append(r1_key.split(',')[1])
        
    for r2_key in list(sorted(res2.keys())):
        tfs_s2.append(r2_key.split(',')[1])
        
    # Resulting timeframes
    replay_tf = []
    
    # Additional timeframe index
    tf_idx_add = 0
    
    # Enter the loop
    while True:
        
        # Add timeframes to the replay_tf
        if tf_idx_add == 0:
            replay_tf.append((tfs_s1[0], tfs_s2[0]))
        else:
            tf_idx_add = 0
            
        # Break condition: one of the timeframe lists is exhausted
        if len(tfs_s1) < 2 or len(tfs_s2) < 2:
            break
            
        # Start and ending times of current and next timeframes for s1 and s2
        tf_s1_cur_start = int(tfs_s1[0].split('->')[0])
        tf_s1_cur_end = int(tfs_s1[0].split('->')[1])
        tf_s1_next_start = int(tfs_s1[1].split('->')[0])
        
        tf_s2_cur_start = int(tfs_s2[0].split('->')[0])
        tf_s2_cur_end = int(tfs_s2[0].split('->')[1])
        tf_s2_next_start = int(tfs_s2[1].split('->')[0])
        
        # Difference between current tf and the next one for s1 and s2
        delta1 = tf_s1_next_start - tf_s1_cur_start
        delta2 = tf_s2_next_start - tf_s2_cur_start
        
        # Find the corresponding timeframes based on the deltas
        if delta1 > delta2:
            # Set the idx to zero
            tf_idx_add = 0
            
            # Next timeframe for s2
            tf_s2 = str(tf_s2_cur_start + delta1) + '->' + str(tf_s2_cur_end + delta1)
            
            # Check if this exact timeframe exists
            if tf_s2 in tfs_s2:
                # If it does find its index
                tf_s2_idx = tfs_s2.index(tf_s2)
                
            else:
                # Flag to see if timeframe is partially found or not
                tf_not_found = True
                
                # Deltas between the existing timeframe and remaining ones
                delta_diff = []
                
                # Iterate over s2 timeframes
                for tf_s2 in tfs_s2:
                    # Get timeframe interval, e.g., (1300, 1310)
                    tf_int = tf_s2.split('->')
                    
                    # Find the delta diff
                    delta_diff.append(abs(delta1 - abs(tf_s2_cur_start - int(tf_int[0]))))
                    
                    # Check if current tf is partly contained in some other timeframe
                    if tf_int[0] < str(tf_s2_cur_start + delta1) < tf_int[1] or tf_int[0] < str(tf_s2_cur_end + delta1) < tf_int[1]:
                        tf_s2_idx = tfs_s2.index(tf_s2)
                        tf_not_found = False
                        break
                        
                # Deal with the case when no corresponding chunk was found
                if tf_not_found:
                    
                    # Find the timeframe that is closest to the target one
                    tf_s2_idx = delta_diff.index(min(delta_diff))
                    
                    # Add timeframes to the replay_tf
                    replay_tf.append((tfs_s1[1], tfs_s2[tf_s2_idx]))
                    
                    # Set the tf_idx_add to 1—do not need to add it twice
                    tf_idx_add = 1
                    
            # Get tf_s1_idx
            tf_s1_idx = tfs_s1.index(str(tf_s1_cur_start + delta1) + '->' + str(tf_s1_cur_end + delta1))
            
            # Adjust timeframe lists
            tfs_s1 = tfs_s1[tf_s1_idx + tf_idx_add:]
            tfs_s2 = tfs_s2[tf_s2_idx + tf_idx_add:]
            
        elif delta1 < delta2:
            # Set the idx to zero
            tf_idx_add = 0
            
            # Next timeframe for s1
            tf_s1 = str(tf_s1_cur_start + delta2) + '->' + str(tf_s1_cur_end + delta2)
            
            # Check if this exact timeframe exists
            if tf_s1 in tfs_s1:
                # If it does find its index
                tf_s1_idx = tfs_s1.index(tf_s1)
                
            else:
                # Flag to see if timeframe is partially found or not
                tf_not_found = True
                
                # Deltas between the existing timeframe and remaining ones
                delta_diff = []
                
                # Iterate over s1 timeframes
                for tf_s1 in tfs_s1:
                    # Get timeframe interval, e.g., (1300, 1310)
                    tf_int = tf_s1.split('->')
                    
                    # Find the delta diff
                    delta_diff.append(abs(delta2 - abs(tf_s1_cur_start - int(tf_int[0]))))
                    
                    # Check if current tf is partly contained in some other timeframe
                    if tf_int[0] < str(tf_s1_cur_start + delta2) < tf_int[1] or tf_int[0] < str(tf_s1_cur_end + delta2) < tf_int[1]:
                        tf_s1_idx = tfs_s1.index(tf_s1)
                        tf_not_found = False
                        break
                        
                # Deal with the case when no corresponding chunk was found
                if tf_not_found:
                    
                    # Find the timeframe that is closest to the target one
                    tf_s1_idx = delta_diff.index(min(delta_diff))
                    
                    # Add timeframes to the replay_tf
                    replay_tf.append((tfs_s1[tf_s1_idx], tfs_s2[1]))
                    
                    # Set the tf_idx_add to 1—do not need to add it twice
                    tf_idx_add = 1
                    
            # Get tf_s1_idx
            tf_s2_idx = tfs_s2.index(str(tf_s2_cur_start + delta2) + '->' + str(tf_s2_cur_end + delta2))
            
            # Adjust timeframe lists
            tfs_s1 = tfs_s1[tf_s1_idx + tf_idx_add:]
            tfs_s2 = tfs_s2[tf_s2_idx + tf_idx_add:]
                    
        else:
            tfs_s1 = tfs_s1[1:]
            tfs_s2 = tfs_s2[1:]
    
    # Dictionary containing resulting output
    res = {}
    
    # Construct result output
    for rtf in replay_tf:
        
        # Chunk result
        chunk_res = {}
        ts1 = ''
        ts2 = ''
        fp1 = ''
        fp2 = ''
        
        # Iterate over res1 to find a match
        for k,v in sorted(res1.items()):
            if k.split(',')[1] == rtf[0]:
                ts1 = k
                fp1 = v
                break
                
        # Iterate over res2 to find a match
        for k,v in sorted(res2.items()):
            if k.split(',')[1] == rtf[1]:
                ts2 = k
                fp2 = v
                break             
        
        # Compute fingerprint similarity
        _, sim = compute_hamming_dist(fp1, fp2)
        
        # Store rearrange the results
        chunk_res['fp'] = fp1
        chunk_res['sim'] = sim
        chunk_res[ts2] = {'fp': fp2}
        
        res[ts1] = chunk_res
        
    return res
        

# The code is adapted from here:
# https://stackoverflow.com/questions/35091557/replace-nth-occurrence-of-substring-in-string
def replace_nth(string, sub, repl, n):
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, repl, 1)
    new_str = before + after
    
    return new_str

    
def get_jsons_to_merge(json_file, action):
    
    # Get experiment name
    exp = json_file.split(action)[1].split('/')[1]
    
    # Get file name 
    regex = re.escape(exp) + r'(?:/|\\)(.*)\.json.gz'
    match = re.search(regex, json_file)

    # If there is no match - exit
    if not match:
        print('get_json_to_merge: no match for the file name %s using regex %s!' % (json_file, regex))
        return
    
    # Current file name
    filename = match.group(1).split('/')[-1]
    
    # Generate files depending on action and experiment
    if action == 'benign':
        
        if exp.split('-')[0] == 'sim':
            # Check which input has been provided
            if exp == SIM_NON_ADV:
                return [re.sub(re.escape(exp), SIM_ADV, json_file)]
            elif exp == SIM_ADV:
                return [re.sub(re.escape(exp), SIM_NON_ADV, json_file)]
            else:
                print('get_json_to_merge: unknown experiment name %s in the benign similar cars!' % (exp,))
                return
        
        elif exp.split('-')[0] == 'diff':
            # Check which input has been provided
            if exp == DIFF_NON_ADV:
                return [re.sub(re.escape(exp), DIFF_ADV, json_file), re.sub(re.escape(exp), DIFF_PARK, json_file)]
            elif exp == DIFF_ADV:
                return [re.sub(re.escape(exp), DIFF_NON_ADV, json_file), re.sub(re.escape(exp), DIFF_PARK, json_file)]
            elif exp == DIFF_PARK:
                return [re.sub(re.escape(exp), DIFF_NON_ADV, json_file), re.sub(re.escape(exp), DIFF_ADV, json_file)]
            else:
                print('get_json_to_merge: unknown experiment name %s in the benign different cars!' % (exp,))
                return 
        else:
            print('get_json_to_merge: unknown experiment type %s (benign eval), only "sim" or "diff" are possible!' % 
                  (exp.split('-')[0],))
            return
    
    elif action == 'baseline':
        
        # Get baseline mode: silent or moving
        bl_mode = json_file.split(exp)[1].split('/')[1]
        
        # Get baseline mode experiment name: same pool of names as exp
        bl_mode_exp = json_file.split(bl_mode)[1].split('/')[1]
        
        # List of extra json files to merge
        jsons_to_merge = []
        
        # Check which input has been provided
        if exp.split('-')[0] == 'sim':                                    
            if exp == SIM_NON_ADV:
                # Check which mode we are in
                if bl_mode == 'silent':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_NON_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, exp, DIFF_PARK, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                    
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                    
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                    elif bl_mode_exp == DIFF_NON_ADV:

                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                     
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
              
                elif bl_mode == 'moving':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_NON_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, exp, DIFF_PARK, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, exp, SIM_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                    elif bl_mode_exp == SIM_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                                                
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                else:
                    print('get_json_to_merge: unknown baseline mode %s in /%s!' % (bl_mode, exp))
                    return
                
            elif exp == SIM_ADV:
                # Check which mode we are in
                if bl_mode == 'silent':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                                                
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                    elif bl_mode_exp == DIFF_NON_ADV:

                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                     
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                    
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                  
                elif bl_mode == 'moving':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                    elif bl_mode_exp == SIM_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_ADV, 2)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_NON_ADV, 2)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_PARK, 2)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_NON_ADV, 2)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                    elif bl_mode_exp == DIFF_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of SIM_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                    
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                else:
                    print('get_json_to_merge: unknown baseline mode %s in /%s!' % (bl_mode, exp))
                    return
            else:
                print('get_json_to_merge: unknown experiment name %s in the baseline similar cars!' % (exp,))
                return
                
        elif exp.split('-')[0] == 'diff':
            # Check which input has been provided
            if exp == DIFF_NON_ADV:
                # Check which mode we are in
                if bl_mode == 'silent':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_PARK, 2)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_NON_ADV, 2)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                    
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                    
                elif bl_mode == 'moving':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    elif bl_mode_exp == SIM_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    elif bl_mode_exp == DIFF_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_PARK, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_NON_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                    elif bl_mode_exp == DIFF_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = replace_nth(new_filepath, bl_mode_exp, DIFF_NON_ADV, 2) 
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = replace_nth(new_filepath, bl_mode_exp, DIFF_PARK, 2) 
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = replace_nth(new_filepath, bl_mode_exp, SIM_ADV, 2) 
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = replace_nth(new_filepath, bl_mode_exp, SIM_NON_ADV, 2) 
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    elif bl_mode_exp == DIFF_PARK:
                        
                         # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                     
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                else:
                    print('get_json_to_merge: unknown baseline mode %s in /%s!' % (bl_mode, exp))
                    return
            
            elif exp == DIFF_ADV:
                # Check which mode we are in
                if bl_mode == 'silent':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                    
                elif bl_mode == 'moving':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                           
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)

                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)

                    elif bl_mode_exp == SIM_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_NON_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_PARK, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_NON_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                    
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_PARK experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                    
                else:
                    print('get_json_to_merge: unknown baseline mode %s in /%s!' % (bl_mode, exp))
                    return
            
            elif exp == DIFF_PARK:
                # Check which mode we are in
                if bl_mode == 'silent':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                                               
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    elif bl_mode_exp == DIFF_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_NON_ADV, 2)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_NON_ADV, 2)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                    
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                     
                elif bl_mode == 'moving':
                    # Files also depend on bl_mode_exp, uugh...
                    if bl_mode_exp == SIM_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)

                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == SIM_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)

                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_NON_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                    
                    elif bl_mode_exp == DIFF_ADV:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_x')[0] + '_' + filename.split('_x')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_PARK, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), SIM_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                    elif bl_mode_exp == DIFF_PARK:
                        
                        # Add files on the same level as bl_mode_exp
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, DIFF_NON_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = replace_nth(json_file, bl_mode_exp, SIM_NON_ADV, 2) 
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        # Add files on the level of DIFF_NON_ADV experiment
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(bl_mode_exp), DIFF_NON_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))

                        # Add files on the level of DIFF_ADV experiment
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, new_filepath, 1)
                        jsons_to_merge.append(new_filepath)
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), DIFF_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                        new_filepath = re.sub(re.escape(exp), DIFF_ADV, json_file, 1)
                        new_filepath = re.sub(re.escape(exp), SIM_NON_ADV, new_filepath, 1)
                        new_filename = filename.split('_')[0] + '_x' + filename.split('_')[1]
                        jsons_to_merge.append(re.sub(re.escape(filename), new_filename, new_filepath))
                        
                    else:
                        print('get_json_to_merge: unknown experiment name %s in /%s/%s (baseline)!' % 
                              (bl_mode_exp, exp, bl_mode))
                        return
                else:
                    print('get_json_to_merge: unknown baseline mode %s in /%s!' % (bl_mode, exp))
                    return
            else:
                print('get_json_to_merge: unknown experiment name %s in the baseline different cars!' % (exp,))
                return
        else:
            print('get_json_to_merge: unknown experiment type %s (baseline eval), only "sim" or "diff" are possible!' % 
                  (exp.split('-')[0],))
            return 
        
        return jsons_to_merge
    
    elif action == 'replay':
        
        if exp.split('-')[0] == 'sim':
            # Check which input has been provided
            if exp == SIM_NON_ADV:
                return [re.sub(re.escape(exp), SIM_ADV, json_file)]
            elif exp == SIM_ADV:
                return [re.sub(re.escape(exp), SIM_NON_ADV, json_file)]
            else:
                print('get_json_to_merge: unknown experiment name %s in the replay similar cars!' % (exp,))
                return
        
        elif exp.split('-')[0] == 'diff':
            # Check which input has been provided
            if exp == DIFF_NON_ADV:
                return [re.sub(re.escape(exp), DIFF_ADV, json_file), re.sub(re.escape(exp), DIFF_PARK, json_file)]
            elif exp == DIFF_ADV:
                return [re.sub(re.escape(exp), DIFF_NON_ADV, json_file), re.sub(re.escape(exp), DIFF_PARK, json_file)]
            elif exp == DIFF_PARK:
                return [re.sub(re.escape(exp), DIFF_NON_ADV, json_file), re.sub(re.escape(exp), DIFF_ADV, json_file)]
            else:
                print('get_json_to_merge: unknown experiment name %s in the replay different cars!' % (exp,))
                return 
        else:
            print('get_json_to_merge: unknown experiment type %s (replay eval), only "sim" or "diff" are possible!' % 
                  (exp.split('-')[0],))
            return
    
    elif action == 'keys':
        
        if exp.split('-')[0] == 'sim':
            # Check which input has been provided
            if exp == SIM_NON_ADV:
                return [re.sub(re.escape(exp), SIM_ADV, json_file)]
            elif exp == SIM_ADV:
                return [re.sub(re.escape(exp), SIM_NON_ADV, json_file)]
            else:
                print('get_json_to_merge: unknown experiment name %s in the keys similar cars!' % (exp,))
                return
        
        elif exp.split('-')[0] == 'diff':
            # Check which input has been provided
            if exp == DIFF_NON_ADV:
                return [re.sub(re.escape(exp), DIFF_ADV, json_file), re.sub(re.escape(exp), DIFF_PARK, json_file)]
            elif exp == DIFF_ADV:
                return [re.sub(re.escape(exp), DIFF_NON_ADV, json_file), re.sub(re.escape(exp), DIFF_PARK, json_file)]
            elif exp == DIFF_PARK:
                return [re.sub(re.escape(exp), DIFF_NON_ADV, json_file), re.sub(re.escape(exp), DIFF_ADV, json_file)]
            else:
                print('get_json_to_merge: unknown experiment name %s in the keys different cars!' % (exp,))
                return 
        else:
            print('get_json_to_merge: unknown experiment type %s (pairing time), only "sim" or "diff" are possible!' % 
                  (exp.split('-')[0],))
            return
    return 
