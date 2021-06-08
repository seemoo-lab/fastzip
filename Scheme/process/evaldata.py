import numpy as np
import os
import sys
import shutil
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime, timedelta
from process.procdata import ewma_filter, remove_noise
from process.normdata import normalize_signal
from process.activityrec import do_activty_recongition
from common.helper import get_sensors_len, check_tframe_boundaries, get_std_and_range, choose_bar_norm, compute_hamming_dist, sig_ratio_by_mean, compute_percentile, save_json_file, fps_str_to_list, idx_to_str, get_ewma_alpha
from quantization.generatefps import generate_random_points, generate_fingerprint, generate_fps_corpus, compute_zeros_ones_ratio, compute_qs_thr, generate_fps_corpus_chunk
from align.aligndata import fine_chunk_alignment, get_xcorr_delay
from const.globconst import *
from const.activityconst import *
from const.fpsconst import *


def eval_per_sensor(data, sensor_type, sensors, tframe, action, experiment, n_bits=0, rnoise=None, norm=None, powerful=False):
    # Sampling rate and length of data to derive a fingerprint on depends on the sensor type
    if sensor_type == 'acc_v' or sensor_type == 'acc_h':
        fs = ACC_FS
        fp_chunk = FP_ACC_CHUNK
        def_n_bits = BITS_ACC
        
         # Set the axis to select the correct acc component: vertical or horizontal 
        if sensor_type == 'acc_v':
            axis = 0
        elif sensor_type == 'acc_h':
            axis = 1
            
    elif sensor_type == 'gyrW':
        fs = GYR_FS
        fp_chunk = FP_GYR_CHUNK 
        def_n_bits = BITS_GYR
        
    elif sensor_type == 'unmag':
        fs = MAG_FS
        fp_chunk = FP_MAG_CHUNK  
        
    elif sensor_type == 'bar':
        fs = BAR_FS
        fp_chunk = FP_BAR_CHUNK
        def_n_bits = BITS_BAR
        
    else:
        print('eval_per_sensor: unknown sensor type "%s", only %s are allowed!' % (sensor_type, ALLOWED_SENSORS))
        return
    
    # If n_bits is not specified, use the default number
    if n_bits == 0:
        n_bits = def_n_bits
    
    # Check if the provided sensors is a non-empty list
    if sensors and isinstance(sensors, list):
        # Check if all elements of sensors list are strings
        for s in sensors:
            if not isinstance(s, str):
                print('eval_per_sensor: %s is not a string in %s, must be a list of strings!' % (s, sensors))
                return
        
        # Remove duplicates from the list if exist
        sensors = list(set(sensors))
        
        # Sort the list
        sensors.sort()
        
        # Check if the provided sensors list is a subset of all the data
        if not set(sensors).issubset(list(data.keys())):
            print('eval_per_sensor: provided "sensors" %s is not a subset of valid sensors %s!' % (sensors, list(data.keys())))
            return
    else:
        print('eval_per_sensor: %s must be a list of strings!' % (sensors,))
        return

    # Check timeframe boundaries
    if not check_tframe_boundaries(tframe):
        return
    
    # Check if we consider signal till the end, i.e., tframe[1] == -1 or not
    if tframe[1] < 0:
        # The ending window is based on the minimal length among signals
        end_win = min(get_sensors_len(data, sensors, fs)) * fs
    else:
        end_win = tframe[1] * fs
        
    # Begining of the signals' chunks
    begin_win = tframe[0] * fs
    
    # Dictionary containing data of the specified timeframe
    tf_data = {}
    
    # Iterate over data
    for k,v in sorted(data.items()):
        
        # Consider only relevant sensors
        if k not in sensors:
            continue
            
        # Check if we have acc or other modalities and chunk the data
        if sensor_type == 'acc_v' or sensor_type == 'acc_h':
            tf_data[k] = v[begin_win:end_win, axis]
        else:
            tf_data[k] = v[begin_win:end_win]
                  
    # Get lengths of all tframe chunks in samples
    chunks_len = get_sensors_len(tf_data, sensors)
    
    # We must esnure that all singal chunks have equal number of samples
    if not chunks_len.count(chunks_len[0]) == len(chunks_len):
        print('eval_per_sensor: all sensor chunks must have the same length %s!' % chunks_len)
        return
    
    # Find out sensors start time
    if experiment == SIM_NON_ADV:
        start_time = (tframe[0], datetime(2018, 12, 13, 9, 28, 51) + timedelta(seconds=tframe[0]))
    elif experiment == SIM_ADV:
        start_time = (tframe[0], datetime(2018, 12, 13, 13, 59, 39) + timedelta(seconds=tframe[0]))
    elif experiment == DIFF_PARK:
        start_time = (tframe[0], datetime(2018, 12, 17, 15, 26, 11) + timedelta(seconds=tframe[0]))
    elif experiment == DIFF_NON_ADV:
        start_time = (tframe[0], datetime(2018, 12, 18, 7, 13, 55) + timedelta(seconds=tframe[0]))
    elif experiment == DIFF_ADV:
        start_time = (tframe[0], datetime(2018, 12, 18, 10, 37, 0) + timedelta(seconds=tframe[0]))
    
    # Filepath to store the results
    filepath = RESULTS_PATH + '/' + action + '/' + experiment + '/' + sensor_type

    # If the results folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Do stuff in parallel depending on action: e.g., quality of signal estimation, random points insertion, etc.
    if action == AR:
        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)

        # Use partial to pass static parameters
        func = partial(compute_ar_metrics, sensor_type=sensor_type, sensors=sensors, fs=fs, fp_chunk=fp_chunk, 
                       filepath=filepath, start_time=start_time, rnoise=rnoise)
        
        # Let workers do the job
        pool.imap(func, list(tf_data.items()))

        # Wait for processes to terminate
        pool.close()
        pool.join()
    
    elif action == KEYS:
        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)

        # Use partial to pass static parameters
        func = partial(compute_keys_from_chunk, sensor_type=sensor_type, sensors=sensors, fs=fs, fp_chunk=fp_chunk, 
                       filepath=filepath, start_time=start_time, n_bits=n_bits, rnoise=rnoise, powerful=powerful)
        
        # Let workers do the job
        pool.imap(func, list(tf_data.items()))

        # Wait for processes to terminate
        pool.close()
        pool.join()
        
    else:
        # Merge actions
        err_action = [AR, KEYS]
        
        # Display error message
        print('eval_per_sensor: unknown action "%s", use one of the following actions: %s' % (action, err_action))
        
        # Remove unused folder
        shutil.rmtree(RESULTS_PATH + '/' + action + '/')
        
        return


def eval_per_sensor_pair(data, sensor_type, sensor_pairs, tframes, action, experiment, n_bits=0, rnoise=None, norm=None):
    # Sampling rate and length of data to derive a fingerprint on depends on the sensor type
    if sensor_type == 'acc_v' or sensor_type == 'acc_h':
        fs = ACC_FS
        fp_chunk = FP_ACC_CHUNK
        xcorr_add = XCORR_ACC
        def_n_bits = BITS_ACC
        
        # Set the axis to select the correct acc component: vertical or horizontal 
        if sensor_type == 'acc_v':
            axis = 0
        elif sensor_type == 'acc_h':
            axis = 1
        
    elif sensor_type == 'gyrW':
        fs = GYR_FS
        fp_chunk = FP_GYR_CHUNK
        xcorr_add = XCORR_GYR
        def_n_bits = BITS_GYR
        
    elif sensor_type == 'unmag':
        fs = MAG_FS
        fp_chunk = FP_MAG_CHUNK
        
    elif sensor_type == 'bar':
        fs = BAR_FS
        fp_chunk = FP_BAR_CHUNK
        xcorr_add = XCORR_BAR
        def_n_bits = BITS_BAR
        
    else:
        print('eval_per_sensor_pair: unknown sensor type "%s", only %s are allowed!' % (sensor_type, ALLOWED_SENSORS))
        return
    
    # If we want to find xcorr set it to zero
    if action == XCORR:
        xcorr_add = 0
        
    # If n_bits is not specified, use the default number
    if n_bits == 0:
        n_bits = def_n_bits
    
    # Check if the provided sensor_pairs is a non-empty list of tuples
    if sensor_pairs and isinstance(sensor_pairs, list):  
        # Check if all elements in sensor_paris are tuples of strings
        for sp in sensor_pairs:
            if not (isinstance(sp, tuple) and len(sp) == 2):
                print('eval_per_sensor_pair: %s must be a tuple of length 2!' % (sp,))
                return
            
            if not isinstance(sp[0], str) or not isinstance(sp[1], str):
                print('eval_per_sensor_pair: tuple %s contains non-string elements, must be a tuple of strings!' % (sp,))
                return
        
        # Remove duplicates from list
        sensor_pairs = list(set(sensor_pairs))
        
        # Sort the list
        sensor_pairs.sort()
        
        # Check if the provided sensor_pairs list is a subset of all the data
        if not set(list(set(list(sum(sensor_pairs, ()))))).issubset(list(data.keys())):
            print('eval_per_sensor_pair: provided "sensor_pairs" %s is not a subset of valid sensors %s!' % 
                  (sensor_pairs, list(data.keys())))
            return
    else:
        print('eval_per_sensor_pair: %s must be a list of tuples!' % (sensor_pairs,))
        return
    
    # Check if provided tframes is non-empty list of tuples
    if tframes and isinstance(tframes, list):
        # Check if all elements in tframes are tuples of tuples and present a valid timeframe
        for tf in tframes:
            if not (isinstance(tf, tuple) and len(tf) == 2):
                print('eval_per_sensor_pair: %s must be a tuple of length 2!' % (tf,))
                return
            
            # Check if timeframes are tuples
            if not (isinstance(tf[0], tuple) or isinstance(tf[1], tuple) or isinstance(tf[0], list) or isinstance(tf[1], list)):
                print('eval_per_sensor_pair: tuple %s contains non-list/tuple elements, must be a tuple of lists/tuples!' % (tf,))
                return
            
            # Check if timeframes are within boundaries
            if not check_tframe_boundaries(tf[0]) or not check_tframe_boundaries(tf[1]):
                return
    else:
        print('eval_per_sensor_pair: %s must be a list of tuples!' % (tframes,))
        return
   
    # Check if sensor pairs and tframes have the same len
    if len(sensor_pairs) != len(tframes):
        print('eval_per_sensor_pair: "sensor_pairs" (len = %d) and "tframes" (len = %d) must have the same length!' %
              (len(sensor_pairs), len(tframes)))
        return
    
    # Dictionary containing data of a sensor pair specified by its timeframes
    tf_data = {}
    
    # Dictionary containing indication if two timeframes are the same or not (required to perform alignment)
    tf_flag = {}
    
    # Index to keep correspondence between sensor pairs and tframes
    idx = 0
    
    # Iterate over sensor pairs
    for sp in sensor_pairs:
        # Check timeframe of the first sensor in pair
        if tframes[idx][0][1] < 0:
            tf1_len_sec = int(len(data[sp[0]]) / fs) - tframes[idx][0][0]
            # End of tf1 signal chunk
            end_win1 = int(len(data[sp[0]]) / fs) * fs
        else:
            tf1_len_sec = tframes[idx][0][1] - tframes[idx][0][0]
            
            # Check if the chunk specified by tframe is within bounds of a signal
            if tframes[idx][0][1] + xcorr_add > int(len(data[sp[0]]) / fs):
                print('eval_per_sensor_pair: tframe %s exceeds sensor "%s" len (= %d)!' % 
                      (tframes[idx][0], sp[0], int(len(data[sp[0]]) / fs)))
                return
            
            # End of tf1 signal chunk
            end_win1 = (tframes[idx][0][1] + xcorr_add) * fs
                
        # Check timeframe of the second sensor in pair
        if tframes[idx][1][1] < 0:
            tf2_len_sec = int(len(data[sp[1]]) / fs) - tframes[idx][1][0]
            # End of tf1 signal chunk
            end_win2 = int(len(data[sp[1]]) / fs) * fs
        else:
            tf2_len_sec = tframes[idx][1][1] - tframes[idx][1][0]
            
            # Check if the chunk specified by tframe is within bounds of a signal
            if tframes[idx][1][1] + xcorr_add > int(len(data[sp[1]]) / fs):
                print('eval_per_sensor_pair: tframe %s exceeds sensor "%s" len (= %d)!' % 
                      (tframes[idx][1], sp[1], int(len(data[sp[1]]) / fs)))
                return
           
            # End of tf1 signal chunk
            end_win2 = (tframes[idx][1][1] + xcorr_add) * fs
        
        # For benign action we must compare similar timeframes
        if action == EVAL_SCENARIOS[0]:
            # Check if timeframes are equal
            if tf1_len_sec != tf2_len_sec:
                print('eval_per_sensor_pair: lenghts of timeframes %s within a pair %s must be the same!' % (tframes[idx], sp,))
                return

        # Begining of the signals' chunks
        begin_win1 = tframes[idx][0][0] * fs
        begin_win2 = tframes[idx][1][0] * fs
        
        # Check if we have acc or other modalities and chunk the data
        if sensor_type == 'acc_v' or sensor_type == 'acc_h':
            tf_data[sp[0] + '_' + sp[1]] = (data[sp[0]][begin_win1:end_win1, axis], data[sp[1]][begin_win2:end_win2, axis])
        else:
            tf_data[sp[0] + '_' + sp[1]] = (data[sp[0]][begin_win1:end_win1], data[sp[1]][begin_win2:end_win2])
        
        # Check if timeframes of the sensor pair are the same
        if tframes[idx][0][0] == tframes[idx][1][0] and tframes[idx][0][1] == tframes[idx][1][1]:
            tf_flag[sp[0] + '_' + sp[1]] = True
        else: 
            tf_flag[sp[0] + '_' + sp[1]] = False
            
        # Increment idx
        idx += 1
    
    # Parse experiment
    exp = experiment.split('_')
    experiment = exp[0]
    
    # Set sensors' start date 
    if len(exp) > 1:
        # Inter experiment case: 1st experiment
        if exp[0] == SIM_NON_ADV:
            start_date1 = datetime(2018, 12, 13, 9, 28, 51)
        elif exp[0] == SIM_ADV:
            start_date1 = datetime(2018, 12, 13, 13, 59, 39)
        elif exp[0] == DIFF_PARK:
            start_date1 = datetime(2018, 12, 17, 15, 26, 11)
        elif exp[0] == DIFF_NON_ADV:
            start_date1 = datetime(2018, 12, 18, 7, 13, 55)
        elif exp[0] == DIFF_ADV:
            start_date1 = datetime(2018, 12, 18, 10, 37, 0)
            
        # Inter experiment case: 2nd experiment
        if exp[-1] == SIM_NON_ADV:
            start_date2 = datetime(2018, 12, 13, 9, 28, 51)
        elif exp[-1] == SIM_ADV:
            start_date2 = datetime(2018, 12, 13, 13, 59, 39)
        elif exp[-1] == DIFF_PARK:
            start_date2 = datetime(2018, 12, 17, 15, 26, 11)
        elif exp[-1] == DIFF_NON_ADV:
            start_date2 = datetime(2018, 12, 18, 7, 13, 55)
        elif exp[-1] == DIFF_ADV:
            start_date2 = datetime(2018, 12, 18, 10, 37, 0)
        
    else:
        # Intra experiment case
        if experiment == SIM_NON_ADV:
            start_date = datetime(2018, 12, 13, 9, 28, 51)
        elif experiment == SIM_ADV:
            start_date = datetime(2018, 12, 13, 13, 59, 39)
        elif experiment == DIFF_PARK:
            start_date = datetime(2018, 12, 17, 15, 26, 11)
        elif experiment == DIFF_NON_ADV:
            start_date = datetime(2018, 12, 18, 7, 13, 55)
        elif experiment == DIFF_ADV:
            start_date = datetime(2018, 12, 18, 10, 37, 0)
        
        # Both are set equal
        start_date1 = start_date
        start_date2 = start_date
    
    # Start time struct would differ for benign and adv cases
    if action == EVAL_SCENARIOS[0] or action == XCORR:
        # Find out sensors start time
        start_time = (tframes[0][0][0], start_date + timedelta(seconds=tframes[0][0][0]))
    else:
        # Start time must be a dict for the adversarial case, potentially different starting times, comparison slots
        start_time = {}

        # Idx to keep track of pairs
        idx = 0
        
        # Iterate over sensor pairs
        for sp in sensor_pairs:
            # Add start times
            start_time[sp[0] + '_' + sp[1]] = ((tframes[idx][0][0], start_date1 + timedelta(seconds=tframes[idx][0][0])), 
                                               (tframes[idx][1][0], start_date2 + timedelta(seconds=tframes[idx][1][0])))
            # Increment idx
            idx += 1
    
    # Chunk parameter to be passed
    chunk_params = (fs, fp_chunk, xcorr_add)
    
    # Construct filepath to store the results
    if len(exp) > 1:
        # Base filepath
        filepath = RESULTS_PATH + '/' + action + '/'
        
        # Append to filepath
        for e in exp:
            filepath += (e + '/')
            
        # Append the sensor type
        filepath += sensor_type
    else:                                            
        filepath = RESULTS_PATH + '/' + action + '/' + experiment + '/' + sensor_type
    
    # If the results folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Do stuff in parallel depending on action
    if action == XCORR:
        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)
        
        # Use partial to pass static parameters
        func = partial(xcorr_eval, sensor_type=sensor_type, chunk_params=chunk_params, start_time=start_time, 
                       filepath=filepath, rnoise=rnoise)

        # Let workers do the job
        pool.imap(func, list(tf_data.items()))

        # Wait for processes to terminate
        pool.close()
        pool.join()
        
    elif action == EVAL_SCENARIOS[0]:
        # Add car number 
        car_sensors = list(set(list(sum(sensor_pairs, ()))))
        car_sensors.sort()

        # Add car number to filepath
        if car_sensors == CAR1:
            filepath += '/' + 'car1'
        elif car_sensors == CAR2:
            filepath += '/' + 'car2'
            
        # Create car1 and car2 folders
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)
       
        # Use partial to pass static parameters
        func = partial(compute_fingerprints_benign, sensor_type=sensor_type, tf_flag=tf_flag, chunk_params=chunk_params, 
                       start_time=start_time, filepath=filepath, experiment=experiment, n_bits=n_bits, rnoise=rnoise)

        # Let workers do the job
        pool.imap(func, list(tf_data.items()))

        # Wait for processes to terminate
        pool.close()
        pool.join()
        
    elif action == EVAL_SCENARIOS[1] or action == EVAL_SCENARIOS[2]:
        # Create folder for each individual sensor
        for sp in sensor_pairs:
            # If the results folder does not exist, create it
            if not os.path.exists(filepath + '/' + sp[0]):
                os.makedirs(filepath + '/' + sp[0])
        
        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)
       
        # Use partial to pass static parameters
        func = partial(compute_fingerprints_adv, sensor_type=sensor_type, tf_flag=tf_flag, chunk_params=chunk_params, 
                       start_time=start_time, filepath=filepath, action=action, n_bits=n_bits, rnoise=rnoise)

        # Let workers do the job
        pool.imap(func, list(tf_data.items()))

        # Wait for processes to terminate
        pool.close()
        pool.join()
        
    else:
        # Merge actions
        err_action = [XCORR] + EVAL_SCENARIOS
        
        # Display error message
        print('eval_per_sensor_pair: unknown action "%s", use one of the following actions: %s' % (action, err_action))
        
        # Remove unused folder
        shutil.rmtree(RESULTS_PATH + '/' + action + '/')
        
        return


def compute_fingerprints_adv(data, sensor_type='', tf_flag={}, chunk_params=(), start_time={}, filepath='', action='', n_bits=24,
                            rnoise=None, norm=None):
    # Try-catch clause for debugging, prints out errors in multiprocessing
    try:
        # Set defaults
        n_peaks_thr = 0
        alpha = 0
        
        # Activity recongition parameters
        if sensor_type == 'acc_v':
            p_thr = P_THR_ACC_V
            n_peaks_thr = NP_THR_ACC_V
            snr_thr = SNR_THR_ACC_V
            alpha = EWMA_ACC_V
            sim_thr = SIM_THR_ACC_V
            bias = BIAS_ACC_V
        elif sensor_type == 'acc_h':
            p_thr = P_THR_ACC_H
            n_peaks_thr = NP_THR_ACC_H
            snr_thr = SNR_THR_ACC_H
            alpha = EWMA_ACC_H
            sim_thr = SIM_THR_ACC_H
            bias = BIAS_ACC_H
        elif sensor_type == 'gyrW':
            p_thr = P_THR_GYR
            snr_thr = SNR_THR_GYR
            bias = BIAS_GYR
            sim_thr = SIM_THR_GYR
        elif sensor_type == 'bar':
            p_thr = P_THR_BAR
            snr_thr = SNR_THR_BAR
            bias = BIAS_BAR
            sim_thr = SIM_THR_BAR
        
        # Extract a key-value pair from the data tuple: (sensor pair, e.g., "01_02", sensor_data)
        k = data[0]
        v = data[1]
        
        # Check if chunk params are valid
        if not isinstance(chunk_params, tuple) or len(chunk_params) != 3:
            print('compute_fingerprints: chunk params must be a tuple of length 3!')
            return
        
        # Check all elements in chunk_params are integers
        for cp in chunk_params:
            if not isinstance(cp, int) or cp < 0:
                print('compute_fingerprints: chunk parameters "%s" can only contain positive integers!' % (chunk_params,))
                return
        
        # Extract chunk params  
        fs = chunk_params[0]
        fp_chunk = chunk_params[1]
        xcorr_add = chunk_params[2]
        
        # Get sensors numbers
        sensors = k.split('_')
        
        # Size of the sliding window is 25% of the fp_chunk for the bar and 50% for other modalities
        if sensor_type == 'bar':
            win_size = int(fp_chunk / 4)
        else:
            win_size = int(fp_chunk / 2)
        
        # Compute number of chunks from the data
        n_chunks1 = int((len(v[0]) - xcorr_add * fs) / (win_size * fs)) - int((fp_chunk - win_size) / win_size)
        n_chunks2 = int((len(v[1]) - xcorr_add * fs) / (win_size * fs)) - int((fp_chunk - win_size) / win_size)
        
        # Dictionary to store metadata
        metadata = {}

        # Dictionary to store parameters in the metadata
        params = {}

        # Add initial metadata fields
        params['data'] = {'chunk_size': fp_chunk, 'win_size': win_size, 'n_chunks1': n_chunks1, 'n_chunks2': n_chunks2}
        params['key'] = {'key_size': n_bits, 'sim_thr': sim_thr}
        if alpha:
            params['sig'] = {'ewma_alpha': alpha}
        metadata['parameters'] = params
        metadata['modality'] = sensor_type
        metadata['eval_scenario'] = action
        metadata['processing_start'] = datetime.now().isoformat().replace('T', ' ')
        metadata['generator_script'] = sys._getframe().f_code.co_name
        
        # Output results dictionary
        results = {}
        
        # Output keys dictionary
        keys = {}
        
        # Timestamp corresponding to signal chunks in seconds and an actual date in format YYYY-MM-DD HH:MM:SS
        ts_sec1 = start_time[k][0][0]
        ts_date1 = start_time[k][0][1]
        ts_sec2 = start_time[k][1][0]
        ts_date2 = start_time[k][1][1]
        
        # Save initial ts_date2
        init_ts_date2 = ts_date2
        
        # Actual number of chunks1 satisfying the AR criteria
        chunks1_idx = 0
        
        # Iterate over the 1st signal
        for i in range(0, n_chunks1):
            
            # Store results of one iteration
            chunk_res = {}
            
            chunk_keys = {}
            
            # Results for each chunk
            chunk1_res = {}
            chunk2_res = {}
            
            # Get signal chunk, we take xcorr_add so that we could end up with fp_chunk * fs signal after xcorr alignment
            s1_chunk = v[0][i * win_size * fs:(i * win_size + fp_chunk + xcorr_add) * fs]
            
            # Perform activity recognition on the chunk1 of lenght fp_chunk * fs
            if sensor_type == 'bar':
                power1, snr1, n_peaks1 = do_activty_recongition(normalize_signal(s1_chunk[:fp_chunk * fs], 'meansub'), sensor_type, fs)
            else:
                power1, snr1, n_peaks1 = do_activty_recongition(s1_chunk[:fp_chunk * fs], sensor_type, fs)
                
            # Check if chunk1 passes AR test
            if power1 > p_thr and n_peaks1 >= n_peaks_thr and snr1 > snr_thr:
                # Process chunk1
                if sensor_type == 'acc_v' or sensor_type == 'acc_h':
                    # Remove noise per chunk basis here if necessary
                    if rnoise is not None:
                        s1_chunk = remove_noise(s1_chunk, rnoise)

                    # Filter signal chunks with EWMA
                    s1_chunk = ewma_filter(abs(s1_chunk), alpha)
                    
                elif sensor_type == 'gyrW':
                    # Remove noise per chunk basis here if necessary
                    if rnoise is not None:
                        s1_chunk = remove_noise(s1_chunk, rnoise)
                
                elif sensor_type == 'bar':
                    # Scale signal chunks
                    s1_chunk = normalize_signal(s1_chunk, 'meansub')

                    # Remove noise per chunk basis here if necessary
                    if rnoise is not None:
                        s1_chunk = remove_noise(s1_chunk, rnoise)
                
                # Make chunk fp_chunk size again
                s1_chunk = s1_chunk[:fp_chunk * fs]
                
                # Compute QS threshold for chunk1
                chunk1_qs_thr = compute_qs_thr(s1_chunk, bias)
                
                # Actual number of chunks2 satisfying the AR criteria
                chunks2_idx = 0
                
                # Reset ts_date2
                ts_date2 = init_ts_date2
                
                # Iterate over the 2nd signal
                for j in range(0, n_chunks2):
                    
                    # Activity recongition flag
                    ar_s2_chunk = False
                    
                    # Get signal chunk, we take xcorr_add so that we could end up with fp_chunk * fs signal after xcorr alignment
                    s2_chunk = v[1][j * win_size * fs:(j * win_size + fp_chunk + xcorr_add) * fs]
                    
                    # Perform activity recognition on the chunk2 of lenght fp_chunk * fs
                    if sensor_type == 'bar':
                        power2, snr2, n_peaks2 = do_activty_recongition(normalize_signal(s2_chunk[:fp_chunk * fs], 'meansub'), 
                                                                        sensor_type, fs)
                    else:
                        power2, snr2, n_peaks2 = do_activty_recongition(s2_chunk[:fp_chunk * fs], sensor_type, fs)
                        
                    # Check if chunk2 passes AR test
                    if power2 > p_thr and n_peaks2 >= n_peaks_thr and snr2 > snr_thr:
                        ar_s2_chunk = True
                        
                    # Check if we continue or not depending on the AR output and adv case
                    if ar_s2_chunk == True and action == 'baseline':
                        # Increment date timestamp
                        ts_date2 = ts_date2 + timedelta(seconds=win_size)
                        
                        continue
                        
                    if ar_s2_chunk == False and (action == 'replay' or action == 'stalk'):
                        # Increment date timestamp
                        ts_date2 = ts_date2 + timedelta(seconds=win_size)
                        
                        continue
                    
                    # Process chunk2
                    if sensor_type == 'acc_v' or sensor_type == 'acc_h':
                        # Remove noise per chunk basis here if necessary
                        if rnoise is not None:
                            s2_chunk = remove_noise(s2_chunk, rnoise)

                        # Filter signal chunks with EWMA
                        s2_chunk = ewma_filter(abs(s2_chunk), alpha)

                    elif sensor_type == 'gyrW':
                        # Remove noise per chunk basis here if necessary
                        if rnoise is not None:
                            s2_chunk = remove_noise(s2_chunk, rnoise)
                    
                    elif sensor_type == 'bar':
                        # Scale signal chunks
                        s2_chunk = normalize_signal(s2_chunk, 'meansub')

                        # Remove noise per chunk basis here if necessary
                        if rnoise is not None:
                            s2_chunk = remove_noise(s2_chunk, rnoise)
                    
                    # Make chunk fp_chunk size again
                    s2_chunk = s2_chunk[:fp_chunk * fs]
                    
                    # Compute QS threshold for chunk2
                    chunk2_qs_thr = compute_qs_thr(s2_chunk, bias)
                    
                    # Number of iterations to compute average similarity
                    sim_iter = 1
                    
                    # Store similarlity results from each iteration
                    sim_results = np.zeros(sim_iter)
                
                    # Compute the average similarity ratio over a few iterations
                    for siter in range(0, sim_iter):
                        # Compute the corpus of fingerprints with found quantization thresholds
                        fps1, fps2, sim, rps = generate_fps_corpus(s1_chunk, chunk1_qs_thr, s2_chunk, chunk2_qs_thr, n_bits, 1)

                    # Store a timestamp corresponding to a signal chunk
                    sec_key = ',' + str(ts_sec2 + j * win_size) + '->' + str(ts_sec2 + j * win_size + fp_chunk)
                    chunk2_res[ts_date2.strftime('%Y-%m-%d %H:%M:%S') + sec_key] = {'sim': np.mean(sim), 'fp': fps2}
                    
                    # Increment date timestamp
                    ts_date2 = ts_date2 + timedelta(seconds=win_size)
                    
                    # Increment number of chunks2
                    chunks2_idx += 1
                
                # Store a timestamp corresponding to a signal chunk
                sec_key = ',' + str(ts_sec1 + i * win_size) + '->' + str(ts_sec1 + i * win_size + fp_chunk)
                chunk_res['ar'] = {'power_dB': power1, 'n_peaks': n_peaks1, 'SNR': snr1}
                chunk_res['fp'] = fps1
                chunk_res[sensors[1]] = chunk2_res
                results[ts_date1.strftime('%Y-%m-%d %H:%M:%S') + sec_key] = chunk_res
                
                # Increment number of chunks1
                chunks1_idx += 1
            
            # Increment date timestamp
            ts_date1 = ts_date1 + timedelta(seconds=win_size)
            
        # Add remaining metadata fields
        params['ar'] = {'power_thr': p_thr, 'snr_thr': snr_thr, 'n_peaks_thr': n_peaks_thr, 'n_chunks1': chunks1_idx, 
                        'n_chunks2': chunks2_idx}
        metadata['processing_end'] = datetime.now().isoformat().replace('T', ' ')
        metadata['created_on'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save json file
        save_json_file({'metadata':metadata, 'results':results}, filepath + '/' + sensors[0], k, True)
                    
    except Exception as e:
        print(e)


def compute_fingerprints_benign(data, sensor_type='', tf_flag={}, chunk_params=(), start_time=(), filepath='', experiment='', n_bits=24,
                                rnoise=None, norm=None):
    # Try-catch clause for debugging, prints out errors in multiprocessing
    try:
        # Set defaults
        n_peaks_thr = 0
        alpha = 0
        
        # Activity recongition parameters
        if sensor_type == 'acc_v':
            p_thr = P_THR_ACC_V
            n_peaks_thr = NP_THR_ACC_V
            snr_thr = SNR_THR_ACC_V
            alpha = EWMA_ACC_V
            sim_thr = SIM_THR_ACC_V
            bias = BIAS_ACC_V
        elif sensor_type == 'acc_h':
            p_thr = P_THR_ACC_H
            n_peaks_thr = NP_THR_ACC_H
            snr_thr = SNR_THR_ACC_H
            alpha = EWMA_ACC_H
            sim_thr = SIM_THR_ACC_H
            bias = BIAS_ACC_H
        elif sensor_type == 'gyrW':
            p_thr = P_THR_GYR
            snr_thr = SNR_THR_GYR
            bias = BIAS_GYR
            sim_thr = SIM_THR_GYR
        elif sensor_type == 'bar':
            p_thr = P_THR_BAR
            snr_thr = SNR_THR_BAR
            bias = BIAS_BAR
            sim_thr = SIM_THR_BAR
        
        # Extract a key-value pair from the data tuple: (sensor pair, e.g., "01_02", sensor_data)
        k = data[0]
        v = data[1]
        
        # Check if chunk params are valid
        if not isinstance(chunk_params, tuple) or len(chunk_params) != 3:
            print('compute_fingerprints: chunk params must be a tuple of length 3!')
            return
        
        # Check all elements in chunk_params are integers
        for cp in chunk_params:
            if not isinstance(cp, int) or cp < 0:
                print('compute_fingerprints: chunk parameters "%s" can only contain positive integers!' % (chunk_params,))
                return
        
        # Extract chunk params  
        fs = chunk_params[0]
        fp_chunk = chunk_params[1]
        xcorr_add = chunk_params[2]
        
        # Get sensors numbers
        sensors = k.split('_')
        
        # Size of the sliding window is 25% of the fp_chunk for the bar and 50% for other modalities
        if sensor_type == 'bar':
            win_size = int(fp_chunk / 4)
        else:
            win_size = int(fp_chunk / 2)
        
        # Compute number of chunks from the data
        n_chunks = int((len(v[0]) - xcorr_add * fs) / (win_size * fs)) - int((fp_chunk - win_size) / win_size)
        
        # Dictionary to store metadata
        metadata = {}

        # Dictionary to store parameters in the metadata
        params = {}

        # Add initial metadata fields
        params['data'] = {'chunk_size': fp_chunk, 'win_size': win_size, 'n_chunks': n_chunks}
        params['key'] = {'key_size': n_bits, 'sim_thr': sim_thr}
        if alpha:
            params['sig'] = {'ewma_alpha':alpha}
        metadata['parameters'] = params
        metadata['modality'] = sensor_type
        metadata['eval_scenario'] = EVAL_SCENARIOS[0]
        metadata['processing_start'] = datetime.now().isoformat().replace('T', ' ')
        metadata['generator_script'] = sys._getframe().f_code.co_name
        
        # Output results dictionary
        results = {}
        
        # Timestamp corresponding to signal chunks in seconds and an actual date in format YYYY-MM-DD HH:MM:SS
        ts_sec = start_time[0]
        ts_date = start_time[1]
       
        # Actual number of chunks satisfying the AR criteria
        chunks1_idx = 0
        chunks2_idx = 0
        
        # Iterate over signal in chunks
        for i in range(0, n_chunks):
            
            # Store results of one iteration
            chunk_res = {}
            
            # Results for each chunk
            chunk1_res = {}
            chunk2_res = {}
            
            # Activity recongition flags
            ar_s1_chunk = False
            ar_s2_chunk = False
            
            # Get signal chunk, we take xcorr_add so that we could end up with fp_chunk * fs signal after xcorr alignment
            s1_chunk = v[0][i * win_size * fs:(i * win_size + fp_chunk + xcorr_add) * fs]
            s2_chunk = v[1][i * win_size * fs:(i * win_size + fp_chunk + xcorr_add) * fs]
            
            # Perform activity recognition on the chunk of lenght fp_chunk * fs
            if sensor_type == 'bar':
                power1, snr1, n_peaks1 = do_activty_recongition(normalize_signal(s1_chunk[:fp_chunk * fs], 'meansub'), sensor_type, fs)
                power2, snr2, n_peaks2 = do_activty_recongition(normalize_signal(s2_chunk[:fp_chunk * fs], 'meansub'), sensor_type, fs)
            else:
                power1, snr1, n_peaks1 = do_activty_recongition(s1_chunk[:fp_chunk * fs], sensor_type, fs)
                power2, snr2, n_peaks2 = do_activty_recongition(s2_chunk[:fp_chunk * fs], sensor_type, fs)

            # Check if chunk1 passes AR test
            if power1 > p_thr and n_peaks1 >= n_peaks_thr and snr1 > snr_thr:
                ar_s1_chunk = True
                chunks1_idx += 1
                
            # Check if chunk2 passes AR test
            if power2 > p_thr and n_peaks2 >= n_peaks_thr and snr2 > snr_thr:
                ar_s2_chunk = True
                chunks2_idx += 1
            
            # If both chunks pass AR test we carry on
            if ar_s1_chunk and ar_s2_chunk:
                # Set the alignment flag
                align_flag = False
                
                # Check if sensors are co-located or not
                if (sensors[0] in CAR1 and sensors[1] in CAR1) or (sensors[0] in CAR2 and sensors[1] in CAR2):
                    # Check if provided timeframes are the same or not
                    if tf_flag[k]:
                        align_flag = True
                
                if sensor_type == 'acc_v' or sensor_type == 'acc_h':
                    # Remove noise per chunk basis here if necessary
                    if rnoise is not None:
                        s1_chunk = remove_noise(s1_chunk, rnoise)
                        s2_chunk = remove_noise(s2_chunk, rnoise)
                    
                    # Filter signal chunks with EWMA
                    s1_chunk = ewma_filter(abs(s1_chunk), alpha)
                    s2_chunk = ewma_filter(abs(s2_chunk), alpha)
                    
                elif sensor_type == 'gyrW':
                    # Remove noise per chunk basis here if necessary
                    if rnoise is not None:
                        s1_chunk = remove_noise(s1_chunk, rnoise)
                        s2_chunk = remove_noise(s2_chunk, rnoise)
                
                elif sensor_type == 'bar':
                    # Scale signal chunks
                    s1_chunk = normalize_signal(s1_chunk, 'meansub')
                    s2_chunk = normalize_signal(s2_chunk, 'meansub')
        
                    # Remove noise per chunk basis here if necessary
                    if rnoise:
                        s1_chunk = remove_noise(s1_chunk, rnoise)
                        s2_chunk = remove_noise(s2_chunk, rnoise)

                # Perform fine-grained alignment
                if align_flag:
                    # Workaround for fuckupery case of sensor 01 (diff cars parking, 735->745)
                    if sensor_type == 'gyrW' and experiment == DIFF_PARK and (sensors[0] == '01' or sensors[1] == '01') and \
                    ts_sec + i * win_size == 735 and ts_sec + i * win_size + fp_chunk == 745:
                        # Do not perform alingment
                        s1_chunk = s1_chunk[:fp_chunk * fs]
                        s2_chunk = s2_chunk[:fp_chunk * fs]
                        delay = 0
                    
                    # Handling weird cases of sensors 06/07 (diff-adv)
                    elif sensor_type == 'bar' and experiment == DIFF_ADV and ts_sec + i * win_size == 1800 and \
                    ts_sec + i * win_size + fp_chunk == 1820:
                        if (sensors[0] == '06' and sensors[1] == '07') or (sensors[0] == '07' and sensors[1] == '06'):
                            # Do not perform alingment
                            s1_chunk = s1_chunk[:fp_chunk * fs]
                            s2_chunk = s2_chunk[:fp_chunk * fs]
                            delay = 0
                                                    
                    elif sensor_type == 'bar' and experiment == DIFF_ADV and ts_sec + i * win_size == 1805 and \
                    ts_sec + i * win_size + fp_chunk == 1825:
                        if (sensors[0] == '06' and sensors[1] == '07') or (sensors[0] == '07' and sensors[1] == '06'):
                            # Do not perform alingment
                            s1_chunk = s1_chunk[:fp_chunk * fs]
                            s2_chunk = s2_chunk[:fp_chunk * fs]
                            delay = 0
                        
                    else:
                        # Construct tuple to handle exceptions in 'fine_chunk_alignment'
                        check = (experiment, sensor_type, sensors, ts_sec + i * win_size, ts_sec + i * win_size + fp_chunk)
                        
                        # Perform alignment as usual
                        s1_chunk, s2_chunk, delay = fine_chunk_alignment(s1_chunk, s2_chunk, fs, fp_chunk, check)
                
                else:
                    s1_chunk = s1_chunk[:fp_chunk * fs]
                    s2_chunk = s2_chunk[:fp_chunk * fs]
                
                # Compute QS thresholds for both chunks
                chunk1_qs_thr = compute_qs_thr(s1_chunk, bias)
                chunk2_qs_thr = compute_qs_thr(s2_chunk, bias)
                
                # Number of iterations to compute average similarity
                sim_iter = 1
            
                # Store similarlity results from each iteration
                sim_results = np.zeros(sim_iter)
                
                # Compute the average similarity ratio over a few iterations (this loop is for backward compatibility)
                for siter in range(0, sim_iter):
                    # Compute the corpus of fingerprints with computed quantization thresholds
                    fps1, fps2, sim, rps = generate_fps_corpus(s1_chunk, chunk1_qs_thr, s2_chunk, chunk2_qs_thr, n_bits, 1)

                # Populate chunkX_res dictionaries
                chunk1_res['ar'] = {'power_dB': power1, 'n_peaks': n_peaks1, 'SNR': snr1}
                chunk2_res['ar'] = {'power_dB': power2, 'n_peaks': n_peaks2, 'SNR': snr2}
                
                chunk1_res['qs_thr'] = chunk1_qs_thr
                chunk1_res['fp'] = fps1
                chunk2_res['qs_thr'] = chunk2_qs_thr
                chunk2_res['fp'] = fps2
                
                # Entry for a chunk pair
                chunk_res[sensors[0]] = chunk1_res
                chunk_res[sensors[1]] = chunk2_res
                chunk_res[XCORR] = delay
                chunk_res['sim'] = np.mean(sim)
                
                # Store a timestamp corresponding to a signal chunk
                sec_key = ',' + str(ts_sec + i * win_size) + '->' + str(ts_sec + i * win_size + fp_chunk)
                results[ts_date.strftime('%Y-%m-%d %H:%M:%S') + sec_key] = chunk_res
            
            # Increment date timestamp
            ts_date = ts_date + timedelta(seconds=win_size)
         
        # Add remaining metadata fields
        params['ar'] = {'power_thr': p_thr, 'snr_thr': snr_thr, 'n_peaks_thr':n_peaks_thr, 'n_chunks1': chunks1_idx, 
                        'n_chunks2': chunks2_idx}
        metadata['processing_end'] = datetime.now().isoformat().replace('T', ' ')
        metadata['created_on'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save json file
        save_json_file({'metadata':metadata, 'results':results}, filepath, k, True)
       
    except Exception as e:
        print(e)
       
    
def xcorr_eval(data, sensor_type='', chunk_params=(), start_time=(), filepath='', rnoise=None):
    # Try-catch clause for debugging, prints out errors in multiprocessing
    try:
        # Defalut params
        n_peaks_thr = 0
        alpha = 0
        
        # Activity recongition parameters
        if sensor_type == 'acc_v':
            p_thr = P_THR_ACC_V
            n_peaks_thr = NP_THR_ACC_V
            snr_thr = SNR_THR_ACC_V
            alpha = EWMA_ACC_V
        elif sensor_type == 'acc_h':
            p_thr = P_THR_ACC_H
            n_peaks_thr = NP_THR_ACC_H
            snr_thr = SNR_THR_ACC_H
            alpha = EWMA_ACC_H
        elif sensor_type == 'gyrW':
            p_thr = P_THR_GYR
            snr_thr = SNR_THR_GYR
        elif sensor_type == 'bar':
            p_thr = P_THR_BAR
            snr_thr = SNR_THR_BAR
        
        # Extract a key-value pair from the data tuple: (sensor pair, e.g., "01_02", sensor_data)
        k = data[0]
        v = data[1]
        
        # Check if chunk params are valid
        if not isinstance(chunk_params, tuple) or len(chunk_params) != 3:
            print('compute_fingerprints: chunk params must be a tuple of length 3!')
            return
        
        # Check all elements in chunk_params are integers
        for cp in chunk_params:
            if not isinstance(cp, int) or cp < 0:
                print('compute_fingerprints: chunk parameters "%s" can only contain positive integers!' % (chunk_params,))
                return
        
        # Extract chunk params  
        fs = chunk_params[0]
        fp_chunk = chunk_params[1]
        xcorr_add = chunk_params[2]
        
        # Get sensors numbers
        sensors = k.split('_')
        
        # Size of the sliding window is 25% of the fp_chunk for the bar and 50% for other modalities
        if sensor_type == 'bar':
            win_size = int(fp_chunk / 4)
        else:
            win_size = int(fp_chunk / 2)
        
        # Compute number of chunks from the data
        n_chunks = int((len(v[0]) - xcorr_add * fs) / (win_size * fs)) - int((fp_chunk - win_size) / win_size)

        # Dictionary to store metadata
        metadata = {}

        # Dictionary to store parameters in the metadata
        params = {}

        # Add initial metadata fields
        params['data'] = {'chunk_size': fp_chunk, 'win_size': win_size, 'n_chunks':n_chunks}
        params['ar'] = {'power_thr': p_thr, 'snr_thr': snr_thr, 'n_peaks_thr':n_peaks_thr}
        if alpha:
            params['sig'] = {'ewma_alpha':alpha}
        metadata['parameters'] = params
        metadata['modality'] = sensor_type
        metadata['eval_scenario'] = XCORR
        metadata['processing_start'] = datetime.now().isoformat().replace('T', ' ')
        metadata['generator_script'] = sys._getframe().f_code.co_name
        
        # Output results dictionary
        results = {}
        
        # Timestamp corresponding to signal chunks in seconds and an actual date in format YYYY-MM-DD HH:MM:SS
        ts_sec = start_time[0]
        ts_date = start_time[1]
        
        # Iterate over signal in chunks
        for i in range(0, n_chunks):
            # Store results of one iteration
            chunk_res = {}
               
            # Activity recongition flags
            ar_s1_chunk = False
            ar_s2_chunk = False
            
            # Get signal chunk, we take xcorr_add so that we could end up with fp_chunk * fs signal after xcorr alignment
            s1_chunk = v[0][i * win_size * fs:(i * win_size + fp_chunk + xcorr_add) * fs]
            s2_chunk = v[1][i * win_size * fs:(i * win_size + fp_chunk + xcorr_add) * fs]
            
            # Perform activity recognition on the chunk of lenght fp_chunk * fs
            if sensor_type == 'bar':
                power1, snr1, n_peaks1 = do_activty_recongition(normalize_signal(s1_chunk[:fp_chunk * fs], 'meansub'), sensor_type, fs)
                power2, snr2, n_peaks2 = do_activty_recongition(normalize_signal(s2_chunk[:fp_chunk * fs], 'meansub'), sensor_type, fs)
            else:
                power1, snr1, n_peaks1 = do_activty_recongition(s1_chunk[:fp_chunk * fs], sensor_type, fs)
                power2, snr2, n_peaks2 = do_activty_recongition(s2_chunk[:fp_chunk * fs], sensor_type, fs)
            
            # Check if chunk1 passes AR test
            if power1 > p_thr and n_peaks1 >= n_peaks_thr and snr1 > snr_thr:
                ar_s1_chunk = True
                
            # Check if chunk2 passes AR test
            if power2 > p_thr and n_peaks2 >= n_peaks_thr and snr2 > snr_thr:
                ar_s2_chunk = True
                
            # If both chunks pass AR test we carry on
            if ar_s1_chunk and ar_s2_chunk:
                
                # Process data
                if sensor_type == 'acc_v' or sensor_type == 'acc_h':
                    # Remove noise per chunk basis here if necessary
                    if rnoise is not None:
                        s1_chunk = remove_noise(s1_chunk, rnoise)
                        s2_chunk = remove_noise(s2_chunk, rnoise)
                    
                    s1_chunk = ewma_filter(abs(s1_chunk), alpha)
                    s2_chunk = ewma_filter(abs(s2_chunk), alpha)
                    
                elif sensor_type == 'gyrW':
                    # Remove noise per chunk basis here if necessary
                    if rnoise is not None:
                        s1_chunk = remove_noise(s1_chunk, rnoise)
                        s2_chunk = remove_noise(s2_chunk, rnoise)
                
                elif sensor_type == 'bar':
                    # Scale signal chunks
                    s1_chunk = normalize_signal(s1_chunk, 'meansub')
                    s2_chunk = normalize_signal(s2_chunk, 'meansub')
        
                    # Remove noise per chunk basis here if necessary
                    if rnoise:
                        s1_chunk = remove_noise(s1_chunk, rnoise)
                        s2_chunk = remove_noise(s2_chunk, rnoise)

                # Compute xcorr delay betweentwo chunks
                delay = get_xcorr_delay(s1_chunk, s2_chunk)
                
                # Entry for a chunk pair
                chunk_res[XCORR] = delay
                
                # Store a timestamp corresponding to a signal chunk
                sec_key = ',' + str(ts_sec + i * win_size) + '->' + str(ts_sec + i * win_size + fp_chunk)
                results[ts_date.strftime('%Y-%m-%d %H:%M:%S') + sec_key] = chunk_res
                
            # Increment date timestamp
            ts_date = ts_date + timedelta(seconds=win_size)
                
        # Add remaining metadata fields
        metadata['processing_end'] = datetime.now().isoformat().replace('T', ' ')
        metadata['created_on'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save json file
        save_json_file({'metadata':metadata, 'results':results}, filepath, k, False)
    
    except Exception as e:
        print(e)
            
        
def compute_ar_metrics(data, sensor_type='', sensors=[], fs=100, fp_chunk=10, filepath='', start_time=(), rnoise=None):
    # Try-catch clause for debugging, prints out errors in multiprocessing
    try:
        # Extract a key-value pair from the data tuple: (sensor_number, sensor_data)
        k = data[0]
        v = data[1]

        # If the subset of sensors is provided use it, otherwise ignore
        if not sensors:
            sensors = [k]

        # If the sensor is valid do stuff
        if k in sensors:

            # Size of the sliding window is 25% of the fp_chunk for the bar and 50% for other modalities; win_size = 5 sec
            if sensor_type == 'bar':
                win_size = int(fp_chunk / 4)
            else:
                win_size = int(fp_chunk / 2)
            
            # Compute number of chunks from the data, here n_chunks here is +1 compared to 'compute_fingerprints' as we also consider
            # the last piece of signal which is smaller than the whole chunk
            n_chunks = int(len(v) / (win_size * fs))

            # Dictionary to store metadata
            metadata = {}

            # Dictionary to store parameters in the metadata
            params = {}

            # Get EWMA alpha if applicable
            alpha = get_ewma_alpha(sensor_type)
            
            # Add initial metadata fields
            if alpha:
                params['data'] = {'chunk_size': fp_chunk, 'win_size': win_size, 'n_chunks':n_chunks, 'ewma_alpha':alpha}
            else:
                params['data'] = {'chunk_size': fp_chunk, 'win_size': win_size, 'n_chunks':n_chunks}
            metadata['parameters'] = params
            metadata['modality'] = sensor_type
            metadata['processing_start'] = datetime.now().isoformat().replace('T', ' ')
            metadata['generator_script'] = sys._getframe().f_code.co_name

            # Dictionary storing resluts of activity recognition
            results = {}

            # Timestamp corresponding to signal chunks in seconds and actual date in format YYYY-MM-DD HH:MM:SS
            ts_sec = start_time[0]
            ts_date = start_time[1]

            # Iterate over signal
            for i in range(0, n_chunks):

                # Store results of one iteration
                chunk_res = {}
                
                # Do chunk-based noise reduction if necessary
                if rnoise is not None:
                    sig_chunk = remove_noise(v[i * win_size * fs:(i * win_size + fp_chunk) * fs], rnoise)
                else:
                    sig_chunk = v[i * win_size * fs:(i * win_size + fp_chunk) * fs]
                
                # Do activity recognition, in the case of bar we subtract mean from the chunk first 
                if sensor_type == 'bar':
                    power, snr, n_peaks = do_activty_recongition(normalize_signal(sig_chunk, 'meansub'), sensor_type, fs)
                else:
                    power, snr, n_peaks = do_activty_recongition(sig_chunk, sensor_type, fs)

                # Add AR results to dictionary
                if power:
                    chunk_res['power_dB'] = power
                if snr: 
                    chunk_res['SNR'] = snr
                
                chunk_res['n_peaks'] = n_peaks
                
                # Store a timestamp corresponding to a signal chunk
                sec_key = ',' + str(ts_sec + i * win_size) + '->' + str(ts_sec + i * win_size + fp_chunk)
                results[ts_date.strftime('%Y-%m-%d %H:%M:%S') + sec_key] = chunk_res

                # Increment date timestamp
                ts_date = ts_date + timedelta(seconds=win_size)
                
            # Add remaining metadata fields
            metadata['processing_end'] = datetime.now().isoformat().replace('T', ' ')
            metadata['created_on'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

            # Save json file
            save_json_file({'metadata':metadata, 'results':results}, filepath, k, True)
    
    except Exception as e:
        print(e)

        
def compute_keys_from_chunk(data, sensor_type='', sensors=[], fs=100, fp_chunk=10, filepath='', start_time=(), n_bits=0, rnoise=None, 
                            powerful=False):
    # Try-catch clause for debugging, prints out errors in multiprocessing
    try:
        # Set defaults
        n_peaks_thr = 0
        alpha = 0
        
        # Activity recongition parameters
        if sensor_type == 'acc_v':
            p_thr = P_THR_ACC_V
            n_peaks_thr = NP_THR_ACC_V
            snr_thr = SNR_THR_ACC_V
            alpha = EWMA_ACC_V
            bias = BIAS_ACC_V
            delta = DELTA_ACC
        elif sensor_type == 'acc_h':
            p_thr = P_THR_ACC_H
            n_peaks_thr = NP_THR_ACC_H
            snr_thr = SNR_THR_ACC_H
            alpha = EWMA_ACC_H
            bias = BIAS_ACC_H
            delta = DELTA_ACC
        elif sensor_type == 'gyrW':
            p_thr = P_THR_GYR
            snr_thr = SNR_THR_GYR
            bias = BIAS_GYR
            delta = DELTA_GYR
        elif sensor_type == 'bar':
            p_thr = P_THR_BAR
            snr_thr = SNR_THR_BAR
            bias = BIAS_BAR
            delta = DELTA_BAR
        
        # To generate fingerprints for powerful adversary: we consider all sensor chunks
        if powerful:
            p_thr = -100
            n_peaks_thr = 0
            snr_thr = 0
        
        # Extract a key-value pair from the data tuple: (sensor_number, sensor_data)
        k = data[0]
        v = data[1]
     
        # If the subset of sensors is provided use it, otherwise ignore
        if not sensors:
            sensors = [k]
        
        # If the sensor is valid do stuff
        if k in sensors:

            # Size of the sliding window is 25% of the fp_chunk for the bar and 50% for other modalities; win_size = 5 sec
            if sensor_type == 'bar':
                win_size = int(fp_chunk / 4)
            else:
                win_size = int(fp_chunk / 2)
            
            # Compute number of chunks from the data—we only consider complete chunks
            n_chunks = int(len(v) / (win_size * fs)) - int((fp_chunk - win_size) / win_size)
            
            # Dictionary to store metadata
            metadata = {}

            # Dictionary to store parameters in the metadata
            params = {}

            # Add initial metadata fields
            params['ar'] = {'power_thr': p_thr, 'snr_thr': snr_thr, 'n_peaks_thr':n_peaks_thr}
            if alpha:
                params['sig'] = {'ewma_alpha':alpha}
            params['key'] = {'key_size': n_bits, 'keys_per_chunk': int(fp_chunk * fs / delta) + 1}
            metadata['parameters'] = params
            metadata['modality'] = sensor_type
            metadata['eval_scenario'] = KEYS
            metadata['processing_start'] = datetime.now().isoformat().replace('T', ' ')
            metadata['generator_script'] = sys._getframe().f_code.co_name
            
            # Dictionary storing resluts of activity recognition
            results = {}

            # Timestamp corresponding to signal chunks in seconds and actual date in format YYYY-MM-DD HH:MM:SS
            ts_sec = start_time[0]
            ts_date = start_time[1]
            
            # Number of chunks that pass AR
            n_chunks_ar = 0
            
            # Iterate over signal
            for i in range(0, n_chunks):
                
                # Store results of one iteration
                chunk_res = {}
                
                # Get signal chunk
                sig_chunk = v[i * win_size * fs:(i * win_size + fp_chunk) * fs]
                
                # Do activity recognition, in the case of bar we subtract mean from the chunk first 
                if sensor_type == 'bar':
                    power, snr, n_peaks = do_activty_recongition(normalize_signal(sig_chunk, 'meansub'), sensor_type, fs)
                else:
                    power, snr, n_peaks = do_activty_recongition(sig_chunk, sensor_type, fs)
                
                # Check if chunk passes AR
                if power > p_thr and n_peaks >= n_peaks_thr and snr >= snr_thr:
                    
                    # Process data
                    if sensor_type == 'acc_v' or sensor_type == 'acc_h':
                        # Remove noise per chunk basis here if necessary
                        if rnoise is not None:
                            sig_chunk = remove_noise(sig_chunk, rnoise)

                        sig_chunk = ewma_filter(abs(sig_chunk), alpha)

                    elif sensor_type == 'gyrW':
                        # Remove noise per chunk basis here if necessary
                        if rnoise is not None:
                            sig_chunk = remove_noise(sig_chunk, rnoise)

                    elif sensor_type == 'bar':
                        # Scale signal chunks
                        sig_chunk = normalize_signal(sig_chunk, 'meansub')

                        # Remove noise per chunk basis here if necessary
                        if rnoise:
                            sig_chunk = remove_noise(sig_chunk, rnoise)
                            
                    # Compute QS threshold for a chunk
                    chunk_qs_thr = compute_qs_thr(sig_chunk, bias)
                    
                    # Compute the corpus of fingerprints with provided QS threshold
                    fps, rps = generate_fps_corpus_chunk(sig_chunk, chunk_qs_thr, n_bits, 1)
                    
                    # Log the 1st key (default points used by the QS)
                    chunk_res[idx_to_str(1)] = {'fp': fps[0], 'points': rps[0]}
                    
                    # Generate a corpus of keys with shifting: we do not need to do it for powerful adversary (one key, default points)
                    if not powerful:                           
                        # Compute the corpus of fingerprints with provided QS threshold (shifting)
                        fps, rps = generate_fps_corpus_chunk(sig_chunk, chunk_qs_thr, n_bits, eqd_delta=delta)

                        # Log remaining keys obtained via shifting
                        for j in range(0, len(fps)):
                            chunk_res[idx_to_str(j + 2)] = {'fp': fps[j], 'points': rps[j]}
                    
                    # Store a timestamp corresponding to a signal chunk
                    sec_key = ',' + str(ts_sec + i * win_size) + '->' + str(ts_sec + i * win_size + fp_chunk)
                    results[ts_date.strftime('%Y-%m-%d %H:%M:%S') + sec_key] = chunk_res
                    
                    # Increment n_chunks_ar
                    n_chunks_ar += 1
                    
                # Increment date timestamp
                ts_date = ts_date + timedelta(seconds=win_size)
            
            # Add remaining metadata fields
            params['data'] = {'chunk_size': fp_chunk, 'win_size': win_size, 'n_chunks':n_chunks, 'n_chunks_ar': n_chunks_ar}
            metadata['processing_end'] = datetime.now().isoformat().replace('T', ' ')
            metadata['created_on'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save json file
            save_json_file({'metadata':metadata, 'results':results}, filepath, k, True)
    
    except Exception as e:
        print(e)
