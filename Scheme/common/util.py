from glob import glob
from json import loads
import itertools
import os
import re
import gzip
import re
import numpy as np
from math import ceil
from datetime import datetime
from collections import ChainMap
from multiprocessing import Pool, cpu_count
from functools import partial
from common.helper import check_tframe_boundaries, del_dict_elem, load_gz_json_file, get_common_timestamps, parse_timestamp, construct_bar_ts, pad_with_zeros, get_jsons_to_merge, process_json_file, rpwc_wrapper, save_json_file, compute_avg_error_rate, get_proximate_chunks, compute_hamming_dist, repack_padv_results
from common.visualizedata import plot_in_car_similarity
from common.loaddata import read_data_files
from process.procdata import process_data
from align.aligndata import align_adv_data
from const.globconst import *
from const.activityconst import *
from const.fpsconst import *


def get_sig_activity_stat(filepath, sensor_type, sensors=[], tframe=[0, -1]):
    # Number of peaks and range are set to zero by default
    n_peaks = 0
    rng_thr = 0
    
    # Activity recongition parameters
    if sensor_type == 'acc_v':
        p_thr = P_THR_ACC_V
        n_peaks = NP_THR_ACC_V
        snr_thr = SNR_THR_ACC_V
    elif sensor_type == 'acc_h':
        p_thr = P_THR_ACC_H
        n_peaks = NP_THR_ACC_H
        snr_thr = SNR_THR_ACC_H
    elif sensor_type == 'gyrW':
        p_thr = P_THR_GYR
        snr_thr = SNR_THR_GYR
    elif sensor_type == 'bar':
        p_thr = P_THR_BAR
        snr_thr = SNR_THR_BAR
        rng_thr = RNG_THR_BAR
    
    # Display activity recongition parameters
    print('"%s": power thr = %.2f, # of peaks = %d, SNR thr = %.2f, range thr = %.2f' % (sensor_type, p_thr, n_peaks, snr_thr, rng_thr))
    print()
    
    # Check if the provided sensors list is a subset of all the data
    if isinstance(sensors, list):
        all_sensors = list(itertools.chain.from_iterable([CAR1, CAR2]))
        if not set(sensors).issubset(all_sensors):
            print('get_sig_activity_stat: Provided "sensors" list %s is not a subset of all sensors %s!' % (sensor_pairs, all_sensors))
            return
    else:
        print('get_sig_activity_stat: %s must be a list of strings!' % (tframe,))
        return

    # Check if tframe is a list
    if not isinstance(tframe, list):
        print('get_sig_activity_stat: tframe must be provided as list!')
        return
    
    # Check timeframe boundaries
    if not check_tframe_boundaries(tframe):
        return
    
    # Dictionary to store necesary filenames
    cached_files = {}
    
    # Load json files
    for s in sensors:
        # Get list containing a single file
        cached_file = glob(filepath + '/' + s + '.json', recursive=True)
        
        # Check if we indeed have only one file (legitimate case)
        if len(cached_file) != 1:
            print('get_sig_activity_stat: "%s" must contain only one file!' % cached_file)
            return
        
        # Add filename to the dictionary
        cached_files[s] = cached_file[0]
       
    # Iterate over json files
    for k,v in sorted(cached_files.items()):
        # Read a json file
        with open(v, 'r') as f:
            results = loads(f.read())['results']
        
        # Check if we consider the full signal or not
        if tframe[1] < 0:
            end = parse_timestamp(list(results.keys())[-1])[1]
        else:
            end = tframe[1]
        
        # List storing keys to be dropped from results dictionary due to timeframe restriciton
        drop_ts = []

        # List storing keys to be dropped from results dictionary due to activity recognition logic
        drop_act = []
        
        # In case of bar we need to consider acc and gyr 
        if sensor_type == 'bar':
            # Load acc_h, acc_v and gyr files
            with open(v.replace(sensor_type, 'acc_h'), 'r') as f:
                ah = loads(f.read())['results']

            with open(v.replace(sensor_type, 'acc_v'), 'r') as f:
                av = loads(f.read())['results']

            with open(v.replace(sensor_type, 'gyrW'), 'r') as f:
                gw = loads(f.read())['results']

            # Chunks for acc and gyr to iterate over
            if list(ah.keys()) == list(av.keys()) == list(gw.keys()): 
                keys = list(ah.keys())
            else:
                print('get_sig_activity_stat: # of acc and gyr chunks is not the same!')
                return
            
            # Store timstamps of good chunks according to acc_v, acc_h and gyrW
            good_ts = []
            
            # Iterate over chunks
            for key in keys:
                # Parse a key
                pk = parse_timestamp(key)

                # Consider samples we are interested in
                if not (pk[0] >= tframe[0] and pk[1] <= end):
                    continue

                # Check acc_h
                if ah[key]['power_dB'] > P_THR_ACC_H and ah[key]['n_peaks'] >= NP_THR_ACC_H and ah[key]['SNR'] > SNR_THR_ACC_H:
                    good_ts.append(key)

                # Check acc_v
                if av[key]['power_dB'] > P_THR_ACC_V and av[key]['n_peaks'] >= NP_THR_ACC_V and av[key]['SNR'] > SNR_THR_ACC_V:
                    good_ts.append(key)

                # Check gyrW
                if gw[key]['power_dB'] > P_THR_GYR and gw[key]['SNR'] > SNR_THR_GYR:
                    good_ts.append(key)
                
            # Remove duplicates from the good_ts
            good_ts = list(set(good_ts))
            
            # Restore order of timestamps
            good_ts.sort()

        # Iterate over results
        for k1, v1 in sorted(results.items()):
            # Parse result key
            parsed_key = parse_timestamp(k1)

            # Check if the sample should be dropped or not depending on timeframe
            if parsed_key[0] >= tframe[0] and parsed_key[1] <= end:
                # Retain only good samples, passing activity recognition check
                if sensor_type == 'acc_v' or sensor_type == 'acc_h':
                    # Activity recognition logic 
                    if not (v1['power_dB'] > p_thr and v1['n_peaks'] >= n_peaks and v1['SNR'] > snr_thr):
                        drop_act.append(k1)
                elif sensor_type == 'gyrW':
                    # Activity recognition logic 
                    if not (v1['power_dB'] > p_thr and v1['SNR'] > snr_thr):
                        drop_act.append(k1)
                elif sensor_type == 'bar':
                    # Intersection flag
                    int_flag = False
                    
                    # Iterate over good timestamps
                    for gts in good_ts:
                        # Parse key from good_ts
                        pk = parse_timestamp(gts)
                        
                        # We check upon the end of chunk, makes more sense for a real-time system, unless you know howto travel in time;)
                        if parsed_key[1] == pk[1]:
                            # Get the scaled range value, we opt for a smaller range, which minimizes RMSE between signals
                            if v1['range_var'] > v1['range_mean']:
                                scaled_range = v1['range_mean']
                            else:
                                scaled_range = v1['range_var']
                            
                            # Activity recognition logic 
                            if not (v1['power_dB'] > p_thr and v1['SNR'] > snr_thr and scaled_range > rng_thr):
                                drop_act.append(k1)
                                
                            # Set up the flag if intersection between bar and good timestamps exists
                            int_flag = True
                            
                            # Leave the loop
                            break
                            
                    # Drop bar timestamps that do not have correspondence in good_ts
                    if not int_flag:
                        drop_ts.append(k1)
            else:
                drop_ts.append(k1)
        
        # Drop unnecesary samples (timeframe)
        results = del_dict_elem(results, drop_ts)

        active_samples = list(set(results) - set(drop_act))
        active_samples.sort()
        print(active_samples)
        return
        
        # Display some stat about individual sensors 
        res_len = len(results)
        active_len = res_len - len(drop_act)
        ratio_percent = active_len / res_len * 100
        print('"%s": all samples = %d, active samples = %d, ratio percentage = %.3f' % (k, res_len, active_len, ratio_percent))
         
    
def get_sensor_data(exp1, exp2=None):
    # Check if the experements are different
    if exp1 == exp2:
        print('get_sensor_data: provided experiments are the same, returning data for "%s"' % (exp1, ))
        exp2 = None
    
    # Load data from the 1st experiment
    if exp1 == SIM_NON_ADV:
        data_path1 = SIM_NON_ADV_PATH
        
    elif exp1 == SIM_ADV:
        data_path1 = SIM_ADV_PATH
        
    elif exp1 == DIFF_PARK:
        data_path1 = DIFF_PARK_PATH
        
    elif exp1 == DIFF_NON_ADV:
        data_path1 = DIFF_NON_ADV_PATH
        
    elif exp1 == DIFF_ADV:
        data_path1 = DIFF_ADV_PATH
    
    else:
        print('get_sensor_data: unknown experiment name, use one of the following "%s", "%s", "%s", "%s", "%s"' % 
              (SIM_NON_ADV, SIM_ADV, DIFF_PARK, DIFF_NON_ADV, DIFF_ADV))
        return
        
    # Load data files
    acc1 = read_data_files(data_path1, 'acc')
    gyr1 = read_data_files(data_path1, 'gyrW')
    bar1 = read_data_files(data_path1, 'bar')
    
    # Process the data
    acc1 = process_data(acc1, 'acc', ACC_FS)
    gyr1 = process_data(gyr1, 'gyrW', GYR_FS)
    bar1 = process_data(bar1, 'bar', BAR_FS)
    
    # Pad the data
    if exp1 == SIM_NON_ADV:
        # Pad gyr sensor 05 with a zero to make length 996800
        gyr1['05'] = pad_with_zeros(gyr1['05'], 1)
    
        # Pad bar sensors 01, 02 and 05 with zeros to make length 99680
        bar1['01'] = pad_with_zeros(bar1['01'], 1)
        bar1['02'] = pad_with_zeros(bar1['02'], 3)
        bar1['04'] = pad_with_zeros(bar1['04'], 5)
    
        # Pad bar sensors 06, 07, 08 and 10 with zeros to make length 114680
        bar1['06'] = pad_with_zeros(bar1['06'], 7)
        bar1['07'] = pad_with_zeros(bar1['07'], 4)
        bar1['08'] = pad_with_zeros(bar1['08'], 2)
        bar1['10'] = pad_with_zeros(bar1['10'], 8)
        
    elif exp1 == SIM_ADV:
        # Sync up sensor signals, compensating for missing samples due to 'onSensorChange' issue when the cars were stationary
        acc1 = align_adv_data(acc1, SIM_ADV)
        gyr1 = align_adv_data(gyr1, SIM_ADV)
        
        # Drop a few samples in acc1['01'] to make recordings equal
        acc1['01'] = acc1['01'][:-10]
    
    elif exp1 == DIFF_PARK:
        # Pad bar sensor 05 with a zero to make length 28620
        bar1['05'] = pad_with_zeros(bar1['05'], 1) 
    
    elif exp1 == DIFF_NON_ADV:
        # Pad acc sensor 05 with zeros to make length 1038500
        acc1['05'] = pad_with_zeros(acc1['05'], 3)
    
        # Pad acc sensors 08, 09, 10 with zeros to make length 954500
        acc1['08'] = pad_with_zeros(acc1['08'], 6)
        acc1['09'] = pad_with_zeros(acc1['09'], 1)
        acc1['10'] = pad_with_zeros(acc1['10'], 4)
    
        # Pad gyr sensor 03 with zeros to make length 1038500
        gyr1['03'] = pad_with_zeros(gyr1['03'], 4)
        
    elif exp1 == DIFF_ADV:
        # Sync up sensor signals, compensating for missing samples due to 'onSensorChange' issue when the cars were stationary
        acc1 = align_adv_data(acc1, DIFF_ADV)
        gyr1 = align_adv_data(gyr1, DIFF_ADV)
    
    # If the 2nd experiment is provided add its data to the 1st experiment
    if exp2 is not None:
        # Load data from the 2nd experiment
        if exp2 == SIM_NON_ADV:
            data_path2 = SIM_NON_ADV_PATH
            
        elif exp2 == SIM_ADV:
            data_path2 = SIM_ADV_PATH
            
        elif exp2 == DIFF_PARK:
            data_path2 = DIFF_PARK_PATH
            
        elif exp2 == DIFF_NON_ADV:
            data_path2 = DIFF_NON_ADV_PATH
            
        elif exp2 == DIFF_ADV:
            data_path2 = DIFF_ADV_PATH
            
        else:
            print('get_sensor_data: unknown experiment name, use one of the following "%s", "%s", "%s", "%s", "%s"' % 
                  (SIM_NON_ADV, SIM_ADV, DIFF_PARK, DIFF_NON_ADV, DIFF_ADV))
            return
        
        # Load data files
        acc2 = read_data_files(data_path2, 'acc')
        gyr2 = read_data_files(data_path2, 'gyrW')
        bar2 = read_data_files(data_path2, 'bar')
        
        # Process the data
        acc2 = process_data(acc2, 'acc', ACC_FS)
        gyr2 = process_data(gyr2, 'gyrW', GYR_FS)
        bar2 = process_data(bar2, 'bar', BAR_FS)
        
        # Pad the data
        if exp2 == SIM_NON_ADV:
            # Pad gyr sensor 05 with a zero to make length 996800
            gyr2['05'] = pad_with_zeros(gyr2['05'], 1)

            # Pad bar sensors 01, 02 and 05 with zeros to make length 99680
            bar2['01'] = pad_with_zeros(bar2['01'], 1)
            bar2['02'] = pad_with_zeros(bar2['02'], 3)
            bar2['04'] = pad_with_zeros(bar2['04'], 5)

            # Pad bar sensors 06, 07, 08 and 10 with zeros to make length 114680
            bar2['06'] = pad_with_zeros(bar2['06'], 7)
            bar2['07'] = pad_with_zeros(bar2['07'], 4)
            bar2['08'] = pad_with_zeros(bar2['08'], 2)
            bar2['10'] = pad_with_zeros(bar2['10'], 8)
            
        elif exp2 == SIM_ADV:
            # Sync up sensor signals, compensating for missing samples due to 'onSensorChange' issue when the cars were stationary
            acc2 = align_adv_data(acc2, SIM_ADV)
            gyr2 = align_adv_data(gyr2, SIM_ADV)
            
            # Drop a few samples in acc2['01'] to make recordings equal
            acc2['01'] = acc2['01'][:-10]
        
        elif exp2 == DIFF_PARK:
            # Pad bar sensor 05 with a zero to make length 28620
            bar2['05'] = pad_with_zeros(bar2['05'], 1)
            
        elif exp2 == DIFF_NON_ADV:
            # Pad acc sensor 05 with zeros to make length 1038500
            acc2['05'] = pad_with_zeros(acc2['05'], 3)

            # Pad acc sensors 08, 09, 10 with zeros to make length 954500
            acc2['08'] = pad_with_zeros(acc2['08'], 6)
            acc2['09'] = pad_with_zeros(acc2['09'], 1)
            acc2['10'] = pad_with_zeros(acc2['10'], 4)

            # Pad gyr sensor 03 with zeros to make length 1038500
            gyr2['03'] = pad_with_zeros(gyr2['03'], 4)
            
        elif exp2 == DIFF_ADV:
            # Sync up sensor signals, compensating for missing samples due to 'onSensorChange' issue when the cars were stationary
            acc2 = align_adv_data(acc2, DIFF_ADV)
            gyr2 = align_adv_data(gyr2, DIFF_ADV)
            
        # Merge exp1 and exp2 data dictionaries
        for k,v in sorted(acc2.items()):
            # Check if the k is in acc1
            if k in acc1:
                # Add an entries to exp1 dict
                acc1['x' + k] = v
                gyr1['x' + k] = gyr2[k]
                bar1['x' + k] = bar2[k]
               
                # Delete entries from exp2 dict
                del acc2[k]
                del gyr2[k]
                del bar2[k]
    
    return acc1, gyr1, bar1
       
    
def get_indiv_error_rates(filepath, sensor_type, action='benign', scenario=None, merge=True, cache=True):
    
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('get_indiv_error_rates: %s is not a valid path!' % (filepath, ))
        return
    
    # Check if provided sensor type is valid
    if not isinstance(sensor_type, str) or sensor_type not in ALLOWED_SENSORS:
        print('get_indiv_error_rates: unknown sensor type "%s", only %s are allowed!' % (sensor_type, ALLOWED_SENSORS))
        return
    
    # Check if action is valid
    if not isinstance(action, str) or not action in EVAL_SCENARIOS:
        print('get_indiv_error_rates: %s is invalid "action", use one of the following: %s!' % (action, EVAL_SCENARIOS))
        return 
    
    # Check if scenario is valid
    if scenario is not None:
        if scenario not in SUBSCENARIOS:
            print('get_indiv_error_rates: %s is invalid "scenario", use one of the following: %s!' % (scenario, SUBSCENARIOS))
            return
     
    # Check if merge is boolean
    if not isinstance(merge, bool):
        print('get_indiv_error_rates: "merge" parameter must be boolean: use True or False!')
        return
    
    # Check if cache is boolean
    if not isinstance(cache, bool):
        print('get_indiv_error_rates: "cache" parameter must be boolean: use True or False!')
        return
    
    # Check if we are using correct data paths
    if action == 'benign':
        if not 'benign' in filepath:
            print('Warning: double check if filepath points to "%s" log files!' % (action,))
            return
    elif action == 'baseline':
        if not 'baseline' in filepath:
            print('Warning: double check if filepath points to "%s" log files!' % (action,))
            return
    elif action == 'replay':
        if not 'replay' in filepath:
            print('Warning: double check if filepath points to "%s" log files!' % (action,))  
            return
    else:
        print('Action %s does not yet have clause here!' % (action,))
        return
    
    # List subdirectories under provided filepath + sensor_type
    st_subdir = glob(filepath + '/' + sensor_type  + '/*/')
    
    # Dict to store log files under specific subdirectory
    log_files = {}
        
    # Iterate over subdirectories
    for st_sd in st_subdir:
            
        # Extract folder name
        regex = re.escape(filepath + '/' + sensor_type) + r'(?:/|\\)(.*)(?:/|\\)'
        match = re.search(regex, st_sd)

        # If there is no match - exit
        if not match:
            print('get_indiv_error_rates: no match for the folder name %s using regex %s!' % (st_sd, regex))
            return

        # Read json.gz log files under subdirectory
        json_files = glob(st_sd + '*.json.gz', recursive=True)

        # Sort read files
        json_files.sort()

        # Store full path of read json files in a dict
        log_files[match.group(1)] = json_files
    
    # Check if we need to cache results
    if merge and cache:
        
        # Dictionary to store cached error rates and number of chunks
        cached_error_rates = {}
        
        # Path to store cached error rates
        sim_diff_cars = filepath.split(action)[1].split('/')[1].split('-')[0]
        
        # Create additional folders for the baseline case
        if action == 'baseline':
            if 'silent' in filepath:
                err_path = CACHE_PATH + '/' + action + '/far/' + 'silent/' + sim_diff_cars + '/indiv/' + sensor_type
            elif 'moving' in filepath:
                err_path = CACHE_PATH + '/' + action + '/far/' + 'moving/' + sim_diff_cars + '/indiv/' + sensor_type
        
        elif action == 'replay':
            err_path = CACHE_PATH + '/' + action + '/far/' + filepath.split(action)[1].split('/')[1] + '/indiv/' + sensor_type
            
        elif action == 'benign':
            err_path = CACHE_PATH + '/' + action + '/tar/' + sim_diff_cars + '/indiv/' + sensor_type
         
        # Create cache folder if it do not exist
        if not os.path.exists(err_path):
            os.makedirs(err_path)
    
    # Iterate over log files
    for k,v in sorted(log_files.items()):
        # Display subfolder names and sensor types
        print('%s %s: ' % (k, sensor_type))
        
        # Iterate over subdirectory
        for json_file in v:
            
            # Check if we need to do merging (default behavior)
            if merge and action != 'replay':
                
                # Store similarity results and number of chunks
                sim = []
                n_chunks1 = 0
                n_chunks2 = 0
                n_chunks = 0
                n_chunks1_all = 0
                n_chunks2_all = 0
                
                # Initilaize list of files
                json_merge = [json_file]
                
                # Add remaining files to be merged
                json_merge.extend(get_jsons_to_merge(json_file, action))

                # Initiate a pool of workers, use all available cores
                pool = Pool(processes=cpu_count(), maxtasksperchild=1)

                # Use partial to pass static parameters
                func = partial(process_json_file, action=action, scenario=scenario)
                
                # Let the workers do their job, we convert data dict to list of tuples
                merged_results = pool.map(func, json_merge)

                # Wait for processes to terminate
                pool.close()
                pool.join()
                
                # Iterate over merged results
                for mr in merged_results:
                    sim += list(mr[0].values())
                    n_chunks1 += mr[1]['parameters']['ar']['n_chunks1']
                    n_chunks2 += mr[1]['parameters']['ar']['n_chunks2']
                    
                    # Some metadata is different for benign vs. baseline and replay
                    if action == 'benign':
                        n_chunks += mr[1]['parameters']['data']['n_chunks']
                        
                    elif action == 'baseline' or action == 'replay':
                        n_chunks1_all += mr[1]['parameters']['data']['n_chunks1']
                        n_chunks2_all += mr[1]['parameters']['data']['n_chunks2']
                
                # Get filename and sim_thr
                filename = merged_results[0][2]
                sim_thr = merged_results[0][1]['parameters']['key']['sim_thr']
                
            else:
                # Get similarity results
                sim, metadata, filename = process_json_file(json_file, action, scenario)
                
                # Similarity threshold is the same for the same modality
                n_chunks1 = metadata['parameters']['ar']['n_chunks1']
                n_chunks2 = metadata['parameters']['ar']['n_chunks2']
                sim_thr = metadata['parameters']['key']['sim_thr']
                
                # Some metadata is different for benign vs. baseline and replay
                if action == 'benign':
                    n_chunks = metadata['parameters']['data']['n_chunks']
                    
                elif action == 'baseline' or action == 'replay':
                    n_chunks1_all = metadata['parameters']['data']['n_chunks1']
                    n_chunks2_all = metadata['parameters']['data']['n_chunks2']
                
                # Extract dict values 
                sim = list(sim.values())
            
            # Display results
            if action == 'benign':
                
                # Convert to numpy array
                if type(sim).__module__ != np.__name__:
                    sim = np.array(sim)

                # Check if we are hitting a border case
                if len(sim) > 0:
                    # Compute True Accept Rate in % 
                    tar = np.count_nonzero(sim >= sim_thr) / len(sim) * 100
                    print('%s,%.2f,%d,%d,%d,%d' % (filename, tar, len(sim), n_chunks1, n_chunks2, n_chunks))
                
                else:
                    # Compute True Accept Rate in % 
                    tar = 'n/a'
                    print('%s,%s,%d' % (filename, tar, len(sim)))
                    
                # Cache results if necessary
                if merge and cache:
                    cached_error_rates[filename] = {'ar': {'n_chunks1': n_chunks1, 'n_chunks2': n_chunks2}, 'input': n_chunks, 
                                                    'tar': tar, 'tar_chunks': len(sim)}

            elif action == 'baseline' or action == 'replay':

                # Counters to track number of overall chunks and those above the sim_thr
                n_chunks_all = 0
                n_chunks_abv_thr = 0
                
                # Iterate over sim
                for s in sim:
                    n_chunks_abv_thr += np.count_nonzero(s >= sim_thr)
                    n_chunks_all += len(s)
                
                # Check if we are hitting a border case
                if n_chunks_all > 0:
                    # Compute False Accept Rate in %
                    far = n_chunks_abv_thr / n_chunks_all * 100
                    
                    # Dispaly results
                    print('%s,%.2f,%d,%d,%d,%d,%d' % (filename, far, n_chunks_all, n_chunks1, n_chunks2, n_chunks1_all, n_chunks2_all))
                else:
                    # Compute False Accept Rate in %
                    far = 'n/a'
                    
                    # Dispaly results
                    print('%s,%s,%d,%d,%d,%d,%d' % (filename, far, n_chunks_all, n_chunks1, n_chunks2, n_chunks1_all, n_chunks2_all))
                    
                # Cache results if necessary
                if merge and cache:
                    # Check if we have 'x' in the file name, remove it
                    if 'x' in filename:
                        filename = filename.replace('x', '')
                    
                    cached_error_rates[filename] = {'ar': {'n_chunks1': n_chunks1, 'n_chunks2': n_chunks2}, 
                                                    'input': {'n_chunks1': n_chunks1_all, 'n_chunks2': n_chunks2_all}, 
                                                    'far': far, 'far_chunks': n_chunks_all}
        if action == 'benign':
            print('%.2f' % compute_avg_error_rate(cached_error_rates))
            plot_in_car_similarity(cached_error_rates, [action, sim_diff_cars, 'indiv', scenario, sensor_type])
        print()

    # Store cached results
    if merge and cache:
        # Check if the scenario is provided or it is the full case
        if scenario is None:
            scenario = 'full'
        
        # Save json file
        save_json_file({'results': cached_error_rates}, err_path, scenario)
        
              
def replay_with_compensation(filepath, sensor_types, scenario='full', car=1):
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('replay_with_compensation: %s is not a valid path!' % (filepath, ))
        return
    
    # Check if sensor_types is a list of len at least 2
    if not isinstance(sensor_types, list) or len(sensor_types) < 1:
        print('replay_with_compensation: %s must be a list of at least length 1!' % (sensor_types,))
        return
    
    # Check if we are using correct data path
    if not 'keys' in filepath:
        print('Warning: double check if filepath points to "keys" log files!')
        return
    
    # Extract experiment name
    exp = filepath.split('keys')[1].split('/')[1]
    
    # Check if scenario is valid
    if scenario != 'full':
        if scenario not in SUBSCENARIOS:
            print('replay_with_compensation: %s is invalid "scenario", use one of the following: %s!' % (scenario, SUBSCENARIOS))
            return
        
        # Sanity checks 
        if exp == DIFF_NON_ADV or exp == DIFF_ADV:
            if scenario == 'parking':
                print('replay_with_compensation: "diff-non-adv" and "diff-adv" experiments do not have "parking" scenario!')
                return
        elif exp == DIFF_PARK:
            if scenario != 'parking':
                print('replay_with_compensation: "diff-park" experiment only contains "parking" scenario!')
                return
        
        scenarios = [scenario]
    else:
        
        # Some experiments do not have parking
        if exp == SIM_NON_ADV or exp == SIM_ADV:
            scenarios = SUBSCENARIOS
        elif exp == DIFF_PARK:
            scenarios = ['parking']
        else:
            scenarios = SUBSCENARIOS[0:-1]
    
    # Check if car param is valid
    if car != 1 and car != 2:
        print('replay_with_compensation: "car" parameter can only integer 1 or 2!')
        return
    else:
        # Replay configuration: which car is victim and which is adv
        if car == 1:
            replay_pairs = ['01_06', '01_07', '01_08', '01_09', '01_10',
                            '02_06', '02_07', '02_08', '02_09', '02_10',
                            '03_06', '03_07', '03_08', '03_09', '03_10',
                            '04_06', '04_07', '04_08', '04_09', '04_10',
                            '05_06', '05_07', '05_08', '05_09', '05_10']
            
            # Needed for data saving
            car_config = 'car1-2'
            
        else:
            replay_pairs = ['06_01', '06_02', '06_03', '06_04', '06_05',
                            '07_01', '07_02', '07_03', '07_04', '07_05',
                            '08_01', '08_02', '08_03', '08_04', '08_05',
                            '09_01', '09_02', '09_03', '09_04', '09_05',
                            '10_01', '10_02', '10_03', '10_04', '10_05']
            
            # Needed for data saving
            car_config = 'car2-1'
    
    # Store key params: similairty threshold and number of fingerprint bits
    key_params = {}
    
    # Store similarity threshold and # bits for each sensor modality
    for st in sensor_types:
        if st == 'acc_v':
            key_params[st] = (SIM_THR_ACC_V, BITS_ACC)
        elif st == 'acc_h':
            key_params[st] = (SIM_THR_ACC_H, BITS_ACC)
        elif st == 'gyrW':
            key_params[st] = (SIM_THR_GYR, BITS_GYR)
        elif st == 'bar':
            key_params[st] = (SIM_THR_BAR, BITS_BAR)
    
    # Create part of the path showing sensor fusion
    st_path = ''

    # Iterate over sensor_types
    for st in sensor_types:
        if st_path:
            st_path += '-' + st
        else:
            st_path += st
    
    # Create path to store the results
    if len(sensor_types) > 1:
        # Path to store error rate results
        err_path = CACHE_PATH + '/replay-compensation/far/' + exp + '/' + car_config + '/fused/' + st_path
                
    else:
        # Path to store error rate results
        err_path = CACHE_PATH + '/replay-compensation/far/' + exp + '/' + car_config + '/indiv/' + st_path
    
    # Create cache folder if it do not exist
    if not os.path.exists(err_path):
        os.makedirs(err_path)
    
    # Store error rates
    cached_error_rates = {}
    
    # Iterate over replay pairs
    for rp in replay_pairs:
        
        # List to store log files
        log_files = []
        
        # Iterate over sensor types
        for st in sensor_types:
            st_files = []
            for s in rp.split('_'):
                st_files.append(filepath + '/' + st + '/' + s + '.json.gz')
                
            log_files.append(st_files)

        # Dictionary to store the results
        results = {}
        
        # Dictionary to store scenario metadata for the fusion case
        scen_metadata = {}

        # Iterate over scenarios
        for scen in scenarios:
            
            # Initiate a pool of workers, use all available cores
            pool = Pool(processes=cpu_count(), maxtasksperchild=1)
            
            # Use partial to pass static parameters
            func = partial(rpwc_wrapper, scenario=scen)

            # Let the workers do their job, we convert data dict to list of tuples
            scen_results = pool.map(func, log_files)
            
            # Wait for processes to terminate
            pool.close()
            pool.join()
            
            # Check if we need to do the fusion or not
            if len(scen_results) > 1:
                # Index for scen_results
                idx = 0
                
                # Iterate over scenario results
                for scr in scen_results:
                    # Extract metadata
                    if scen in scen_metadata:
                        scen_metadata[scen].update({scr[0]: scr[2]})
                    else:
                        scen_metadata[scen] = {scr[0]: scr[2]}
                    
                    # Update scen_results
                    scen_results[idx] = scr[0:-1]
                    
                    # Increment idx
                    idx += 1
                
                # From list of tuples to a dict
                scen_results = dict(scen_results)
                
                # Get common timestamps for all sensor types
                common_ts = get_common_timestamps(scen_results)
                
                # Count number of chunks that can be corrected by ECC
                n_good_chunks = 0
                
                # Iterate over common timestamps
                for cts in common_ts:
                    # Keep track of the number of bits above or below error correction threshold
                    ext_bits = 0
                    req_bits = 0
                    
                    # Iterate over sensor types
                    for st in sensor_types:
                        # Adjust common timestamp if we are dealing with bar
                        if st == 'bar':
                            cts_i = construct_bar_ts(cts)
                        else:
                            cts_i = cts
                            
                        # Compute number of error correction bits defined by the threshold and those that chunk really has
                        n_ecc_bits_thr = ceil(key_params[st][1] * key_params[st][0] / 100)
                        n_ecc_bits_res = int(key_params[st][1] * scen_results[st][cts_i]['sim'] / 100)
                        
                        # Check if fingerprint similarity is above or below threshold
                        if scen_results[st][cts_i]['sim'] >= key_params[st][0]:
                            ext_bits += n_ecc_bits_res - n_ecc_bits_thr
                        else:
                            req_bits += n_ecc_bits_thr - n_ecc_bits_res
                    
                    # Check if the fused chunk can be corrected with ECC
                    if ext_bits >= req_bits:
                        n_good_chunks += 1
                
                # Store fusion results
                results[scen] = (n_good_chunks, len(common_ts))
            else:
                # List to store similarities
                sim = []
                
                # Iterate over the scen_results
                for k,v in sorted(scen_results[0][1].items()):
                    sim.append(v['sim'])
                
                # Convert sim to numpy array
                sim = np.array(sim)

                # Cache results
                if len(sim) > 1:
                    # Compute FAR in %
                    far = np.count_nonzero(sim >= key_params[scen_results[0][0]][0]) / len(sim) * 100
                    
                    # Store similarities along with sensor_type in results
                    results[scen] = sim
                else:
                    far = 'n/a'
                    
                    # Store similarities along with sensor_type in results
                    results[scen] = 'n/a'
                
                # Construct FAR part and add metadata to it
                cached_res = {'far': far, 'far_chunks': len(sim)}
                cached_res.update(scen_results[0][2])
                
                # Add 1st entry or update cached_error_rates
                if rp in cached_error_rates:
                    cached_error_rates[rp].update({scen: cached_res})
                else:
                    cached_error_rates[rp] = {scen: cached_res}

        # Display results
        # Counters to track number of overall chunks and those above the sim_thr
        n_chunks = 0
        n_chunks_abv_thr = 0
        
        # Counters for full metadata
        ar_n_chunks1 = 0
        ar_n_chunks2 = 0
        input_n_chunks1 = 0
        input_n_chunks2 = 0
        
        print(rp, sensor_types)
        # Iterate over scenario results
        for k,v in results.items():
            # Check if we have a fusion or individual modality case
            if len(sensor_types) > 1:
                # Sum up chunk to compute FAR for the full scenario
                n_chunks_abv_thr += v[0]
                n_chunks += v[1]
                
                # Check if we are hitting a border case
                if v[1] > 0:
                    # Compute FAR in %
                    far = v[0] / v[1] * 100
                    
                    # Dispaly FAR in %
                    print('%s,%.2f,%d' % (k, far, v[1]))
                else:
                    far = 'n/a'
                    
                    # Display FAR
                    print('%s,%s,%d' % (k, far, v[1]))
                    
                # Construct FAR part and add metadata to it
                cached_res = {'far': far, 'far_chunks': v[1]}
                cached_res.update(scen_metadata[k])
                
                # Cache FAR
                if rp in cached_error_rates:
                    cached_error_rates[rp].update({k: cached_res})
                else:
                    cached_error_rates[rp] = {k: cached_res}
                
            else:
                # Check if we are hitting a border case
                if type(v).__module__ == np.__name__:
                    # Sum up chunk to compute FAR for the full scenario
                    n_chunks_abv_thr += np.count_nonzero(v >= key_params[sensor_types[0]][0])
                    n_chunks += len(v)
                    
                    # Dispaly FAR in %
                    print('%s,%.2f,%d' % (k, np.count_nonzero(v >= key_params[sensor_types[0]][0]) / len(v) * 100, len(v)))
                else:
                    # Display FAR
                    print('%s,%s,%d' % (k, v, 0))
                    
                # Sum up metadata entries
                ar_n_chunks1 += cached_error_rates[rp][k]['ar']['n_chunks1']
                ar_n_chunks2 += cached_error_rates[rp][k]['ar']['n_chunks2']
                input_n_chunks1 += cached_error_rates[rp][k]['input']['n_chunks1']
                input_n_chunks2 += cached_error_rates[rp][k]['input']['n_chunks2']
        
        if len(scenarios) > 1:
            # Check if we are hitting a border case
            if n_chunks > 0:
                # Compute FAR in % and display results
                far = n_chunks_abv_thr / n_chunks * 100
                print('%s,%.2f,%d' % ('full', far, n_chunks))
            else:
                far = 'n/a'
                print('%s,%s,%d' % ('full', far, n_chunks))
            
            # Check if we do the fusion or individual modalities
            if len(sensor_types) > 1:
                # Full metadata
                full_metadata = {}
                
                # Iterate over sensor types
                for st in sensor_types:
                    # Full metadata
                    full_metadata[st] = {'ar': {'n_chunks1': 0, 'n_chunks2': 0}, 'input': {'n_chunks1': 0, 'n_chunks2': 0}}
                    
                    # Iterate over scenarios
                    for scen in scenarios:
                        full_metadata[st]['input']['n_chunks1'] += scen_metadata[scen][st]['input']['n_chunks1']
                        full_metadata[st]['input']['n_chunks2'] += scen_metadata[scen][st]['input']['n_chunks2']
                        full_metadata[st]['ar']['n_chunks1'] += scen_metadata[scen][st]['ar']['n_chunks1']
                        full_metadata[st]['ar']['n_chunks2'] += scen_metadata[scen][st]['ar']['n_chunks2']
                    
                # Update cached error rates
                cached_error_rates[rp].update({'full': {'far': far, 'far_chunks': n_chunks}})
                cached_error_rates[rp]['full'].update(full_metadata)
                
            else:
                # Cache full results 
                cached_error_rates[rp].update({'full': {'far': far, 'far_chunks': n_chunks, 
                                                        'input': {'n_chunks1': input_n_chunks1, 'n_chunks2': input_n_chunks2}, 
                                                        'ar': {'n_chunks1': ar_n_chunks1, 'n_chunks2': ar_n_chunks2}}})
        print() 
    
    # Save json file
    save_json_file({'results': cached_error_rates}, err_path, 'results')
 
    
def powerful_adv(filepath, sensor_types, scenario=None, car=1):
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('powerful_adv: %s is not a valid path!' % (filepath, ))
        return
    
    # Check if we are using correct data paths
    if not 'keys' in filepath or not ('sim-adv' in filepath or 'diff-adv' in filepath):
        print('Warning: double check if filepath points to "keys/sim-adv" or "keys/diff-adv" log files!')
        return
    
    # Check if sensor_types is a list of len at least 2
    if not isinstance(sensor_types, list) or len(sensor_types) < 2:
        print('powerful_adv: %s must be a list of at least length 2!' % (sensor_types,))
        return
    
    # Check if scenario is valid
    if scenario is not None:
        if scenario not in SUBSCENARIOS[:-1]:
            print('powerful_adv: %s is invalid "scenario", use one of the following: %s!' % (scenario, SUBSCENARIOS[:-1]))
            return
    
    # Check if car param is valid
    if car != 1 and car != 2:
        print('powerful_adv: "car" parameter can only integer 1 or 2!')
        return
    else:
        # Check which car is victim and which adv
        if car == 1:
            sensor_pairs = ['01_06', '01_07', '01_08', '01_09', '01_10',
                            '02_06', '02_07', '02_08', '02_09', '02_10',
                            '03_06', '03_07', '03_08', '03_09', '03_10',
                            '04_06', '04_07', '04_08', '04_09', '04_10',
                            '05_06', '05_07', '05_08', '05_09', '05_10']
            
            # Needed for data saving
            car_config = 'car1-2'
            
        else:
            sensor_pairs = ['06_01', '06_02', '06_03', '06_04', '06_05',
                            '07_01', '07_02', '07_03', '07_04', '07_05',
                            '08_01', '08_02', '08_03', '08_04', '08_05',
                            '09_01', '09_02', '09_03', '09_04', '09_05',
                            '10_01', '10_02', '10_03', '10_04', '10_05']
            
            # Needed for data saving
            car_config = 'car2-1'
    
    # Extract experiment name
    exp = filepath.split('keys')[1].split('/')[1]
    
    # Create part of the path showing sensor fusion
    st_path = ''
        
    # Iterate over sensor_types
    for st in sensor_types:
        if st_path:
            st_path += '-' + st
        else:
            st_path += st
    
    # Path to store error rate results
    err_path = CACHE_PATH + '/powerful/far/' + exp + '/' + car_config + '/' + st_path
    
    # Create err_pth if it does not exist
    if not os.path.exists(err_path):
        os.makedirs(err_path)
    
    # Store key params: similairty threshold and number of fingerprint bits
    key_params = {}
    
    # Fill in keys similarity params
    for st in sensor_types:
        # Store similarity threshold and # bits for each sensor modality
        if st == 'acc_v':
            key_params[st] = (SIM_THR_ACC_V, BITS_ACC)
        elif st == 'acc_h':
            key_params[st] = (SIM_THR_ACC_H, BITS_ACC)
        elif st == 'gyrW':
            key_params[st] = (SIM_THR_GYR, BITS_GYR)
        elif st == 'bar':
            key_params[st] = (SIM_THR_BAR, BITS_BAR)
    
    # Dict to store all log files to be used in evaluation
    log_files = {}
    
    # Iterate over sensor pairs (constructing log_files)
    for sp in sensor_pairs:
        # Get sensors
        sensors = sp.split('_')
        
        # Lists to store json files to we are working on
        victim = []
        stalker = []
        
        # Get victim and stalker files
        for st in sensor_types:
            victim.append(str(filepath + '/' + st + '/' + sensors[0] + '.json.gz'))
            stalker.append(str(filepath.replace('keys', 'powerful') + '/' + st + '/' + sensors[1] + '.json.gz'))
            
        # Check if sensor entry exists
        if sensors[0] not in log_files:
            log_files[sensors[0]] = victim
        
        if sensors[1] not in log_files:
            log_files[sensors[1]] = stalker
    
    # Dict to store results
    results = {}
    
    # Iterate over sensor paris (compute similarities)
    for sp in sensor_pairs:
        # Get sensors
        sensors = sp.split('_')
        
        # Load victim's files
        
        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)

        # Use partial to pass static parameters
        func = partial(process_json_file, action='keys', scenario=scenario, fuse=True)

        # Let the workers do their job, we convert data dict to list of tuples
        v_res = pool.map(func, log_files[sensors[0]])
        
        # Wait for processes to terminate
        pool.close()
        pool.join()
        
        # Repack data to dictionary
        v_res = {vr[1]: vr[0] for vr in v_res}

        # Load stalker files
        
        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)

        # Use partial to pass static parameters
        func = partial(process_json_file, action='keys', scenario=scenario, fuse=True)

        # Let the workers do their job, we convert data dict to list of tuples
        s_res = pool.map(func, log_files[sensors[1]])

        # Wait for processes to terminate
        pool.close()
        pool.join()
        
        # Repack data to dictionary
        s_res = {sr[1]: sr[0] for sr in s_res}
        
        # Get common timestamps for all sensor types (victim)
        common_ts = get_common_timestamps(v_res) 
        
        # Get timestamp of stalker (they are the same irrespective of modality, except bar)  
        if sensor_types[0] != 'bar':
            s_ts = list(s_res[sensor_types[0]].keys())
        else:
            for st in sensor_types:
                if st != 'bar':
                    s_ts = list(s_res[st].keys())
                    break
        
        # Sort them just in case
        s_ts.sort()
           
        idx = 0
        
        # Dictionary to store results of a single sensor pair
        sp_res = {}
        
        # Iterate over common timestamps
        for cts in common_ts:
            # Dict to store results of one iteration
            iter_res = {}
            
            # Get a list of stalker's chunks within 90 sec vicinity
            s_chunks = get_proximate_chunks(s_ts.index(cts), s_ts)
            
            # Iterate over modalities
            for st in sensor_types:
                # Fingerprint similarity between victims fp and a set of stalkers fps
                fp_sim = {}
            
                # Keep track of the number of bits above or below error correction threshold
                ext_bits = 0
                req_bits = 0
                
                # Iterate over stalker's chunks
                for s_chunk in s_chunks:
                    # Adjust common timestamp if we are dealing with bar
                    if st == 'bar':
                        cts_i = construct_bar_ts(cts)
                        s_chunk_i = construct_bar_ts(s_chunk)
                        
                        # Handle the case when we do not have enough data
                        if s_chunk_i not in s_res[st]:
                            continue
                    else:
                        cts_i = cts
                        s_chunk_i = s_chunk
                    
                    # Compute similarity between vicitms chunk a set of stalkers chunks
                    _, fp_sim[s_chunk] = compute_hamming_dist(v_res[st][cts_i], s_res[st][s_chunk_i])

                # Pick the stalker's chunks yeilding the highest similarity                           
                best_s_chunk = max(fp_sim, key=fp_sim.get)

                # If we want to get 2nd highest similarlity
#                 del fp_sim[best_s_chunk]
#                 best_s_chunk = max(fp_sim, key=fp_sim.get)
                
                # Compute number of error correction bits defined by the threshold and those that chunk really has
                n_ecc_bits_thr = ceil(key_params[st][1] * key_params[st][0] / 100)
                n_ecc_bits_res = int(key_params[st][1] * fp_sim[best_s_chunk] / 100)
                
                # Check if fingerprint similarity is above or below threshold
                if fp_sim[best_s_chunk] >= key_params[st][0]:
                    ext_bits += n_ecc_bits_res - n_ecc_bits_thr
                else:
                    req_bits += n_ecc_bits_thr - n_ecc_bits_res

                # Store similarity results of best matched modalilty
                fp_sim = {'match': fp_sim[best_s_chunk]}
                
                # Remaining sensor types
                rem_sensor_types = sensor_types.copy()
                
                # Remove current st
                rem_sensor_types.remove(st)
                
                # Iterate over remaining sensor types
                for rst in rem_sensor_types:
                    # Adjust common timestamp if we are dealing with bar
                    if rst == 'bar':
                        cts_i = construct_bar_ts(cts)
                        best_s_chunk_i = construct_bar_ts(best_s_chunk)
                        
                        # Handle the case where best_s_chunk does not have the corresponding bar value
                        if best_s_chunk_i not in s_res[rst]:
                            sim = 0
                            fp_sim[rst] = 0

                            # Compute number of error correction bits defined by the threshold and those that chunk really has
                            n_ecc_bits_thr = ceil(key_params[rst][1] * key_params[rst][0] / 100)
                            n_ecc_bits_res = int(key_params[rst][1] * sim / 100)
                    
                            # Check if fingerprint similarity is above or below threshold
                            if sim >= key_params[rst][0]:
                                ext_bits += n_ecc_bits_res - n_ecc_bits_thr
                            else:
                                req_bits += n_ecc_bits_thr - n_ecc_bits_res
                            
                            continue
                    
                    else:
                        cts_i = cts
                        best_s_chunk_i = best_s_chunk
                    
                    # Compute similarity between vicitms chunk a stalkers chunk
                    _, sim = compute_hamming_dist(v_res[rst][cts_i], s_res[rst][best_s_chunk_i])

                    # Store similarity results of remaining modalities
                    fp_sim[rst] = sim
                    
                    # Compute number of error correction bits defined by the threshold and those that chunk really has
                    n_ecc_bits_thr = ceil(key_params[rst][1] * key_params[rst][0] / 100)
                    n_ecc_bits_res = int(key_params[rst][1] * sim / 100)
                    
                    # Check if fingerprint similarity is above or below threshold
                    if sim >= key_params[rst][0]:
                        ext_bits += n_ecc_bits_res - n_ecc_bits_thr
                    else:
                        req_bits += n_ecc_bits_thr - n_ecc_bits_res
                
                # Check if the fused chunk can be corrected with ECC
                if ext_bits >= req_bits:
                    fp_sim['res'] = 1
                else:
                    fp_sim['res'] = 0
                
                # Store results of one iteration
                iter_res[st] = fp_sim
            
            # Store results for all modalities in one iteration
            sp_res[cts] = iter_res

        # Store results of a sensor pair
        results[sp] = repack_padv_results(sp_res, sensor_types)
        
        if len(sensor_types) == 2:
            print('%s,%d,%.2f,%.4f,%.2f,%.4f' % (sp, results[sp]['far_chunks'], 
                                                 results[sp][sensor_types[0]]['matched_far'], results[sp][sensor_types[0]]['far'], 
                                                 results[sp][sensor_types[1]]['matched_far'], results[sp][sensor_types[1]]['far']))
        elif len(sensor_types) == 3:
            print('%s,%d,%.2f,%.4f,%.2f,%.4f,%.2f,%.4f' %
                  (sp, results[sp]['far_chunks'], results[sp][sensor_types[0]]['matched_far'], results[sp][sensor_types[0]]['far'], 
                   results[sp][sensor_types[1]]['matched_far'], results[sp][sensor_types[1]]['far'], 
                   results[sp][sensor_types[2]]['matched_far'], results[sp][sensor_types[2]]['far']))
        elif len(sensor_types) == 4:
            print('%s,%d,%.2f,%.4f,%.2f,%.4f,%.2f,%.4f,%.2f,%.4f' % 
                  (sp, results[sp]['far_chunks'], results[sp][sensor_types[0]]['matched_far'], results[sp][sensor_types[0]]['far'], 
                   results[sp][sensor_types[1]]['matched_far'], results[sp][sensor_types[1]]['far'], 
                   results[sp][sensor_types[2]]['matched_far'], results[sp][sensor_types[2]]['far'], 
                   results[sp][sensor_types[3]]['matched_far'], results[sp][sensor_types[3]]['far']))
        
        print()
    
    # Check if the scenario is provided or it is the full case
    if scenario is None:
        scenario = 'full'

    # Save json file
    save_json_file({'results': results}, err_path, scenario)
      
        
def compute_pairing_time(filepath, sensor_types, scenario=None, req_chunks=2, prot='fpake'):
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('compute_pairing_time: %s is not a valid path!' % (filepath, ))
        return
    
    # Check if sensor_types is a list of len at least 2
    if not isinstance(sensor_types, list) or len(sensor_types) < 1:
        print('compute_pairing_time: %s must be a list of at least length 1!' % (sensor_types,))
        return
    
    # Check if we are using correct data path
    if not 'keys' in filepath:
        print('Warning: double check if filepath points to "keys" log files!')
        return
    
    # Check if scenario is valid
    if scenario is not None:
        if scenario not in SUBSCENARIOS:
            print('compute_pairing_time: %s is invalid "scenario", use one of the following: %s!' % (scenario, SUBSCENARIOS))
            return
    
    # Check if the protocol is valid
    if prot != 'fpake' and prot != 'fcom':
        print('compute_pairing_time: %s is invalid "prot", use either "fpake" or "fcom"!' % (prot,))
        return
    
    # Extract experiment name
    exp = filepath.split('keys')[1].split('/')[1]
    
    # Create part of the path showing sensor fusion
    st_path = ''
        
    # Iterate over sensor_types
    for st in sensor_types:
        if st_path:
            st_path += '-' + st
        else:
            st_path += st
    
    # Path to store pairing time results
    if len(sensor_types) > 1:
        ptime_path = CACHE_PATH + '/pairing-time-' + prot + '/' + exp.split('-')[0] +  '/fused/'  + st_path
    else:
        ptime_path = CACHE_PATH + '/pairing-time-' + prot + '/' + exp.split('-')[0] + '/indiv/'  + st_path
    
    # Create ptime_path if it does not exist
    if not os.path.exists(ptime_path):
        os.makedirs(ptime_path)
    
    # Check number of req_chunks for fusion either fPAKE or fuzzy commitment case
    # these number are computed according to Tables III and X in the paper
    if st_path == 'acc_v-acc_h':
        if prot == 'fpake':
            req_chunks = 3
        else:
            req_chunks = 5
    
    elif st_path == 'acc_v-gyrW':
        if prot == 'fpake':
            req_chunks = 3
        else:
            req_chunks = 5
        
    elif st_path == 'acc_v-bar':
        if prot == 'fpake':
            req_chunks = 4
        else:
            req_chunks = 6
    
    elif st_path == 'acc_h-gyrW':
        if prot == 'fpake':
            req_chunks = 3
        else:
            req_chunks = 5
    
    elif st_path == 'acc_h-bar':
        if prot == 'fpake':
            req_chunks = 3
        else:
            req_chunks = 5
        
    elif st_path == 'gyrW-bar':
        if prot == 'fpake':
            req_chunks = 2
        else:
            req_chunks = 6
    
    elif st_path == 'acc_v-acc_h-gyrW':
        if prot == 'fpake':
            req_chunks = 2
        else:
            req_chunks = 3
    
    elif st_path == 'acc_v-acc_h-bar':
        if prot == 'fpake':
            req_chunks = 2
        else:
            req_chunks = 4
    
    elif st_path == 'acc_v-gyrW-bar':
        if prot == 'fpake':
            req_chunks = 2
        else:
            req_chunks = 4
    
    elif st_path == 'acc_h-gyrW-bar':
        if prot == 'fpake':
            req_chunks = 2
        else:
            req_chunks = 4
    
    elif st_path == 'acc_v-acc_h-gyrW-bar':
        if prot == 'fpake':
            req_chunks = 2
        else:
            req_chunks = 3
    
    # Dict to store all log files to be used in evaluation
    log_files = {}
    
    # Iterate over sensor_types
    for st in sensor_types:
        # Get filepath for each sensor type
        st_filepath = filepath + '/' + st
        
        # Read json.gz files under st_filepath
        log_files[st] = sorted(glob(st_filepath + '/'  + '*.json.gz', recursive=True))
    
    # Index to track json files to be merged
    idx = 0
    
    # If we need 2 chunks to get e.g., our 20 bits, we need a current chunk + 1
    req_chunks = req_chunks - 1
    
    # Dict to store cached resutls
    cached_pairing_time = {}
    
    # Iterate over json files
    for json_file in log_files[sensor_types[0]]:
        # Dict to store results of json files
        results = {}
            
        # Dict to store metadata of json files
        metadata = {}
        
         # Get file name 
        regex = re.escape(sensor_types[0]) + r'(?:/|\\)(.*)\.json.gz'
        match = re.search(regex, json_file)

        # If there is no match - exit
        if not match:
            print('compute_pairing_time: no match for the file name %s using regex %s!' % (json_file, regex))
            return
        
        # Current file name
        filename = match.group(1).split('/')[-1]

        # Initilaize list of files
        json_merge = [json_file]
        
        # Add remaining files to be merged
        json_merge.extend(get_jsons_to_merge(json_file, 'keys'))
        
        # Add remaining sensor types to json_merge
        for i in range(1, len(sensor_types)):
            json_merge.append(log_files[sensor_types[i]][idx])
            json_merge.extend(get_jsons_to_merge(log_files[sensor_types[i]][idx], 'keys'))

        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)
        
        # Use partial to pass static parameters
        func = partial(process_json_file, action='keys', scenario=scenario, fuse=True)

        # Let the workers do their job, we convert data dict to list of tuples
        merged_results = pool.map(func, json_merge)

        # Wait for processes to terminate
        pool.close()
        pool.join()
        
        # Iterate over merged results             
        for mr in merged_results:
            if mr[1] in results:
                # Update results
                results[mr[1]].update(mr[0])

                # Update metadata
                metadata[mr[1]]['input']['n_chunks'] += mr[2]['parameters']['data']['n_chunks']
                metadata[mr[1]]['ar']['n_chunks'] += mr[2]['parameters']['data']['n_chunks_ar']
            
            else:
                # Add results
                results[mr[1]] = mr[0]
                
                # Add metadata
                metadata[mr[1]] = {'input': {'n_chunks': mr[2]['parameters']['data']['n_chunks']}, 
                                   'ar': {'n_chunks': mr[2]['parameters']['data']['n_chunks_ar']}}
        
        # Get common timestamps
        if len(sensor_types) > 1:
            cts = get_common_timestamps(results)
        else:
            cts = sorted(list(results[sensor_types[0]].keys()))
        
        # Store pairing time
        pairing_time = []
        
        # Index to be used in while loop
        i = 0
        
        # Border case counter
        bc_count = 0
        
        # Iterate over common cts
        while i + req_chunks < len(cts) - 1:
            # Get timestamps of two successive chunks w.r.t. req_chunks
            ts1 = datetime.strptime(cts[i].split(',')[0], '%Y-%m-%d %H:%M:%S')
            ts2 = datetime.strptime(cts[i + req_chunks].split(',')[0], '%Y-%m-%d %H:%M:%S')
            
            # Check if the diff between two successive timestamps is bigger than
            # the minimum pause between two experiments, which is 30 minutes
            if (ts2 - ts1).total_seconds() > 1800:
                # Update i index
                i = i + req_chunks

                # Increment border case counter
                bc_count += 1
                continue
                
            # Get timeframes of successive chunks w.r.t. req_chunks
            tf1 = int(cts[i].split(',')[1].split('->')[0])
            tf2 = int(cts[i + req_chunks].split(',')[1].split('->')[1])
            
            # Sanity check
            if tf2 - tf1 < 0:
                print('compute_pairing_time: fishy timframe diff, check exp="%s", mod="%s", scen="%s", file="%s", ts1="%s", ts2="%s"' % 
                      (exp.split('-')[0], st_path, scenario, filename, cts[i], cts[i + req_chunks]))
                return

            # Record pairing time
            pairing_time.append(tf2 - tf1)
    
            # Increment index
            i += 1
        
        # Check if border case counter is pausible
        if scenario is None:
            if exp.split('-')[0] == 'sim':
                if bc_count != 1:
                    print('compute_pairing_time: fishy bc_count="%d", check exp="%s", mod="%s", scen="%s", file="%s"' % 
                          (bc_count, exp.split('-')[0], st_path, scenario, filename))
            elif exp.split('-')[0] == 'diff':
                if bc_count != 2:
                    print('compute_pairing_time: fishy bc_count="%d", check exp="%s", mod="%s", scen="%s", file="%s"' % 
                          (bc_count, exp.split('-')[0], st_path, scenario, filename))
        elif scenario == 'parking':
            if exp.split('-')[0] == 'sim':
                if bc_count != 1:
                    print('compute_pairing_time: fishy bc_count="%d", check exp="%s", mod="%s", scen="%s", file="%s"' % 
                          (bc_count, exp.split('-')[0], st_path, scenario, filename))
            elif exp.split('-')[0] == 'diff':
                if bc_count != 0:
                    print('compute_pairing_time: fishy bc_count="%d", check exp="%s", mod="%s", scen="%s", file="%s"' % 
                          (bc_count, exp.split('-')[0], st_path, scenario, filename))
        elif scenario == 'city':
            if bc_count != 2:
                print('compute_pairing_time: fishy bc_count="%d", check exp="%s", mod="%s", scen="%s", file="%s"' % 
                      (bc_count, exp.split('-')[0], st_path, scenario, filename))
        elif scenario == 'highway':
            if bc_count != 1:
                print('compute_pairing_time: fishy bc_count="%d", check exp="%s", mod="%s", scen="%s", file="%s"' % 
                      (bc_count, exp.split('-')[0], st_path, scenario, filename))
        elif scenario == 'country':
            if bc_count != 1:
                print('compute_pairing_time: fishy bc_count="%d", check exp="%s", mod="%s", scen="%s", file="%s"' % 
                      (bc_count, exp.split('-')[0], st_path, scenario, filename))

        # Add pairing time and metadata to the results
        cached_pairing_time[filename] = {'pairing_time_sec': np.mean(np.array(pairing_time))}
        cached_pairing_time[filename].update(metadata)

        # Increment index
        idx += 1
    
    # Check if the scenario is provided or it is the full case
    if scenario is None:
        scenario = 'full'

    # Save json file
    save_json_file({'results': cached_pairing_time}, ptime_path, scenario)
        
    
def get_fused_error_rates(filepath, sensor_types, action='benign', scenario=None, merge=True, cache=True):
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('get_fused_error_rates: %s is not a valid path!' % (filepath, ))
        return
    
    # Check if sensor_types is a list of len at least 2
    if not isinstance(sensor_types, list) or len(sensor_types) < 2:
        print('get_fused_error_rates: %s must be a list of at least length 2!' % (sensor_types,))
        return
    
    # Check if sensor types do not exceed the length of available sensor_types
    if len(sensor_types) > len(ALLOWED_SENSORS) - 1:
        print('get_fused_error_rates: length of provided "sensor_types" must be smaller than %d!' % (len(ALLOWED_SENSORS) - 1,))
        return
    
    # Check if provided sensor_types are valid
    if not (set(sensor_types).issubset(set(ALLOWED_SENSORS))):
        print('get_fused_error_rates: provided "sensor_types" are invalid, only %s are valid!' % (ALLOWED_SENSORS,))
        return
    
    # Check if action is valid
    if not action in EVAL_SCENARIOS:
        print('get_fused_error_rates: %s is invalid "action" use one of the following: %s!' % (action, EVAL_SCENARIOS))
        return 
    
    # Check if scenario is valid
    if scenario is not None:
        if scenario not in SUBSCENARIOS:
            print('get_fused_error_rates: %s is invalid "scenario", use one of the following: %s!' % (scenario, SUBSCENARIOS))
            return
    
    # Check if merge is of bool type
    if not isinstance(merge, bool):
        print('get_fused_error_rates: "merge" parameter must be boolean: use True or False!')
        return
    
    # Check if cache is boolean
    if not isinstance(cache, bool):
        print('get_fused_error_rates: "cache" parameter must be boolean: use True or False!')
        return
    
    # Check if we are using correct data paths
    if action == 'benign':
        if not 'benign' in filepath:
            print('Warning: double check if filepath points to "%s" log files!' % (action,))
            return
    elif action == 'baseline':
        if not 'baseline' in filepath:
            print('Warning: double check if filepath points to "%s" log files!' % (action,))
            return
    elif action == 'replay':
        if not 'replay' in filepath:
            print('Warning: double check if filepath points to "%s" log files!' % (action,))  
            return
    else:
        print('Action %s does not yet have clause here!' % (action,))
        return
    
    # Store key params: similarity threshold and number of fingerprint bits
    key_params = {}
    
    # Dict to store all log files to be used in evaluation
    log_files = {}
    
    # Iterate over sensor_types
    for st in sensor_types:
        # Get filepath for each sensor type
        st_filepath = filepath + '/' + st
        
        # List subdirectories under provided st_filepath
        st_subdir = glob(st_filepath  + '/*/')
        
        # Dict to store log files under specific subdirectory
        subdir_log_files = {}
        
        # Iterate over subdirectories
        for st_sd in st_subdir:
            
            # Extract folder name
            regex = re.escape(st_filepath) + r'(?:/|\\)(.*)(?:/|\\)'
            match = re.search(regex, st_sd)
            
            # If there is no match - exit
            if not match:
                print('get_fused_error_rates: no match for the folder name %s using regex %s!' % (st_sd, regex))
                return
            
            # Read json.gz log files under subdirectory
            json_files = glob(st_sd + '*.json.gz', recursive=True)
            
            # Sort read files
            json_files.sort()
            
            # Store full path of read json files in a dict
            subdir_log_files[match.group(1)] = json_files
            
        # Store similarity threshold and # bits for each sensor modality
        if st == 'acc_v':
            key_params[st] = (SIM_THR_ACC_V, BITS_ACC)
        elif st == 'acc_h':
            key_params[st] = (SIM_THR_ACC_H, BITS_ACC)
        elif st == 'gyrW':
            key_params[st] = (SIM_THR_GYR, BITS_GYR)
        elif st == 'bar':
            key_params[st] = (SIM_THR_BAR, BITS_BAR)
            
        # Store log files per sensor type
        log_files[st] = subdir_log_files
    
    # Check if we need to cache results
    if merge and cache:
        # Dictionary to store cached error rates and number of chunks
        cached_error_rates = {}
        
        # Path to store cached error rates
        sim_diff_cars = filepath.split(action)[1].split('/')[1].split('-')[0]
        
        # Create part of the path showing sensor fusion
        st_path = ''
        
        # Iterate over sensor_types
        for st in sensor_types:
            if st_path:
                st_path += '-' + st
            else:
                st_path += st
        
        # Create additional folders for the baseline case
        if action == 'baseline':
            if 'silent' in filepath:
                err_path = CACHE_PATH + '/' + action + '/far/' + 'silent/' + sim_diff_cars + '/fused/' + st_path
            elif 'moving' in filepath:
                err_path = CACHE_PATH + '/' + action + '/far/' + 'moving/' + sim_diff_cars + '/fused/' + st_path
        
        elif action == 'replay':
            err_path = CACHE_PATH + '/' + action + '/far/' + filepath.split(action)[1].split('/')[1] + '/fused/' + st_path
        
        elif action == 'benign':
            err_path = CACHE_PATH + '/' + action + '/tar/' + sim_diff_cars + '/fused/' + st_path
            
            # Metadata for cached fps
            metadata_cached_fps = {}
            fp_len = 0
            ecc_bits = 0
            
            metadata_cached_fps['modalities'] = sensor_types
            
            # Iterate over sensor types
            for st in sensor_types:
                # Get fingerprint lenght and ECC capability in bits per modality 
                fp_len += key_params[st][1]
                ecc_bits += key_params[st][1] - ceil(key_params[st][1] * (key_params[st][0] / 100))
            
            metadata_cached_fps['fp_len'] = fp_len
            metadata_cached_fps['ecc_bits'] = ecc_bits
            
            # For the benign case we also want to log fingerprints
            fp_path =  CACHE_PATH + '/' + action + '/fps/' + sim_diff_cars + '/fused/' + st_path
            
            # Create fp_path if it does not exist
            if not os.path.exists(fp_path):
                os.makedirs(fp_path)
        
        # Create err_pth if it does not exist
        if not os.path.exists(err_path):
            os.makedirs(err_path)
    
    # Iterate over all log files
    for k,v in sorted(log_files[sensor_types[0]].items()):
        # Index to retreive a correct file
        f_idx = 0
        
        # Display subfolder names and sensor types
        print('%s %s: ' % (k, sensor_types))
        
        # Iterate over log files of subdirectory
        for json_file in v:
            
            # Dict to store results of json files
            results = {}
            
            # Dict to store metadata of json files
            metadata = {}
            
            # Dictionary to store cached fignerprints for benchmarking and randomness eval
            cached_fps = {}
            
            # Check if we need to do merging (default behavior)
            if merge and action != 'replay':
                
                # Initilaize list of files
                json_merge = [json_file]

                # Add remaining files to be merged
                json_merge.extend(get_jsons_to_merge(json_file, action))
                
                # Add remaining sensor types to json_merge
                for i in range(1, len(sensor_types)):
                    json_merge.append(log_files[sensor_types[i]][k][f_idx])
                    json_merge.extend(get_jsons_to_merge(log_files[sensor_types[i]][k][f_idx], action))

                # Initiate a pool of workers, use all available cores
                pool = Pool(processes=cpu_count(), maxtasksperchild=1)
            
                # Use partial to pass static parameters
                func = partial(process_json_file, action=action, scenario=scenario, fuse=True)

                # Let the workers do their job, we convert data dict to list of tuples
                merged_results = pool.map(func, json_merge)

                # Wait for processes to terminate
                pool.close()
                pool.join()
                
                # Iterate over merged results
                for mr in merged_results:
                    # Some tweaks are necessary depending on the action
                    if action == 'benign':
                        if mr[0] in results:
                            # Update results
                            results[mr[0]].update(mr[1])
                            
                            # Update metadata
                            metadata[mr[0]]['input']['n_chunks'] += mr[3]['parameters']['data']['n_chunks']
                            metadata[mr[0]]['ar']['n_chunks1'] += mr[3]['parameters']['ar']['n_chunks1']
                            metadata[mr[0]]['ar']['n_chunks2'] += mr[3]['parameters']['ar']['n_chunks2']
                            metadata[mr[0]]['ar']['n_chunks12'] += len(mr[1])
                            
                        else:
                            # Add results
                            results[mr[0]] = mr[1]
                            
                            # Add metadata
                            metadata[mr[0]] = {'input': {'n_chunks': mr[3]['parameters']['data']['n_chunks']}, 
                                               'ar': {'n_chunks1': mr[3]['parameters']['ar']['n_chunks1'], 
                                                      'n_chunks2': mr[3]['parameters']['ar']['n_chunks2'],
                                                      'n_chunks12': len(mr[1])}}
                    
                    elif action == 'baseline' or action == 'replay':
                        # Temp dictionary to restructure results
                        res = {}
                        
                        if mr[0] in results:
                            
                            # Extract relevant portion of the data
                            for k1,v1 in sorted(mr[1].items()):
                                res[k1] = v1[mr[2].split('_')[1]]
                        
                            # Iterate over res
                            for k1,v1 in sorted(res.items()):
                                if k1 in results[mr[0]]:
                                    # Update results
                                    results[mr[0]][k1].update(v1)
                                else:
                                    # Add a complete new key-value pair to results
                                    results[mr[0]][k1] = v1
                             
                            # Update metadata
                            metadata[mr[0]]['input']['n_chunks1'] += mr[3]['parameters']['data']['n_chunks1']
                            metadata[mr[0]]['input']['n_chunks2'] += mr[3]['parameters']['data']['n_chunks2']
                            metadata[mr[0]]['ar']['n_chunks1'] += mr[3]['parameters']['ar']['n_chunks1']
                            metadata[mr[0]]['ar']['n_chunks2'] += mr[3]['parameters']['ar']['n_chunks2']
                            
                        else:
                            # Extract relevant portion of the data
                            for k1,v1 in sorted(mr[1].items()):
                                res[k1] = v1[mr[2].split('_')[1]]
                            
                            # Add res to results
                            results[mr[0]] = res
                            
                            # Add metadata
                            metadata[mr[0]] = {'input': {'n_chunks1': mr[3]['parameters']['data']['n_chunks1'], 
                                                         'n_chunks2': mr[3]['parameters']['data']['n_chunks2']}, 
                                               'ar': {'n_chunks1': mr[3]['parameters']['ar']['n_chunks1'], 
                                                      'n_chunks2': mr[3]['parameters']['ar']['n_chunks2']}}
                
                # Get file name for displaying results
                filename = merged_results[0][2]
                
                # Adjust metadata taking into account data repetition with fusion
                if action == 'baseline':
                    if 'silent' in filepath:
                        for k1,v1 in metadata.items():
                            metadata[k1]['input']['n_chunks1'] = int(v1['input']['n_chunks1'] / 3)
                            metadata[k1]['input']['n_chunks2'] = int(v1['input']['n_chunks2'] / 2)
                            metadata[k1]['ar']['n_chunks1'] = int(v1['ar']['n_chunks1'] / 3)
                            metadata[k1]['ar']['n_chunks2'] = int(v1['ar']['n_chunks2'] / 2)
                        
                    elif 'moving' in filepath:
                        for k1,v1 in metadata.items():
                            metadata[k1]['input']['n_chunks1'] = int(v1['input']['n_chunks1'] / 5)
                            metadata[k1]['input']['n_chunks2'] = int(v1['input']['n_chunks2'] / 2)
                            metadata[k1]['ar']['n_chunks1'] = int(v1['ar']['n_chunks1'] / 5)
                            metadata[k1]['ar']['n_chunks2'] = int(v1['ar']['n_chunks2'] / 2)
           
            else:
                # Load results for the 1st sensor_type
                _, results[sensor_types[0]], filename, md = process_json_file(json_file, action, scenario, fuse=True)
                
                # Some metadata is different for benign vs. baseline and replay
                if action == 'benign':
                    metadata[sensor_types[0]] = {'input': {'n_chunks': md['parameters']['data']['n_chunks']}, 
                                                 'ar': {'n_chunks1': md['parameters']['ar']['n_chunks1'], 
                                                        'n_chunks2': md['parameters']['ar']['n_chunks2'],
                                                        'n_chunks12': len(results[sensor_types[0]])}}
                    
                elif action == 'baseline' or action == 'replay':
                    metadata[sensor_types[0]] = {'input': {'n_chunks1': md['parameters']['data']['n_chunks1'], 
                                                           'n_chunks2': md['parameters']['data']['n_chunks2']}, 
                                                 'ar': {'n_chunks1': md['parameters']['ar']['n_chunks1'], 
                                                        'n_chunks2': md['parameters']['ar']['n_chunks2']}}
                
                # Add remaining sensor types to json_merge
                for i in range(1, len(sensor_types)):
                    _, results[sensor_types[i]], _, md = process_json_file(log_files[sensor_types[i]][k][f_idx], action, 
                                                                           scenario, fuse=True)
                    
                    # Some metadata is different for benign vs. baseline and replay
                    if action == 'benign':
                        metadata[sensor_types[i]] = {'input': {'n_chunks': md['parameters']['data']['n_chunks']}, 
                                                     'ar': {'n_chunks1': md['parameters']['ar']['n_chunks1'], 
                                                            'n_chunks2': md['parameters']['ar']['n_chunks2'],
                                                            'n_chunks12': len(results[sensor_types[i]])}}
                    
                    elif action == 'baseline' or action == 'replay':
                        metadata[sensor_types[i]] = {'input': {'n_chunks1': md['parameters']['data']['n_chunks1'], 
                                                               'n_chunks2': md['parameters']['data']['n_chunks2']}, 
                                                     'ar': {'n_chunks1': md['parameters']['ar']['n_chunks1'], 
                                                            'n_chunks2': md['parameters']['ar']['n_chunks2']}}
            
            # Get common timestamps for all sensor types
            common_ts = get_common_timestamps(results)  
            
            # Metrics depend on the scenario
            if action == 'benign':
                # Count number of chunks that can be corrected by ECC
                n_good_chunks = 0
            
            elif action == 'baseline' or action == 'replay':
                # Compute error rates per chunk—we take the mean of all subchunks
                error_rates = []
                
                # Just for backward compatibility
                if not merge or action == 'replay':
                    # Get sensor number necessary for parsing
                    sn = filename.split('_')[1]
            
            # Iterate over common timestamps
            for cts in common_ts:
                
                # Compute error rates depending on evaluation scenario
                if action == 'benign':
                    # Keep track of the number of bits above or below error correction threshold
                    ext_bits = 0
                    req_bits = 0
                    
                    # Vars to store fingerprints
                    fp1 = ''
                    fp2 = ''
                    
                    # Iterate over sensor types
                    for st in sensor_types:
                        # Adjust common timestamp if we are dealing with bar
                        if st == 'bar':
                            cts_i = construct_bar_ts(cts)
                        else:
                            cts_i = cts
                            
                        # Compute number of error correction bits defined by the threshold and those that chunk really has
                        n_ecc_bits_thr = ceil(key_params[st][1] * key_params[st][0] / 100)
                        n_ecc_bits_res = int(key_params[st][1] * results[st][cts_i]['sim'] / 100)
                        
                        # Check if fingerprint similarity is above or below threshold
                        if results[st][cts_i]['sim'] >= key_params[st][0]:
                            ext_bits += n_ecc_bits_res - n_ecc_bits_thr
                        else:
                            req_bits += n_ecc_bits_thr - n_ecc_bits_res
                            
                        # Store fingerprints
                        fp1 += results[st][cts_i][filename.split('_')[0]]['fp']
                        fp2 += results[st][cts_i][filename.split('_')[1]]['fp']
                    
                    # Check if the fused chunk can be corrected with ECC
                    if ext_bits >= req_bits:
                        n_good_chunks += 1
                        
                        # Cache fingerprints
                        cached_fps[cts] = {'fp' + filename.split('_')[0]: fp1, 'fp' + filename.split('_')[1]: fp2}
                
                elif action == 'baseline' or action == 'replay':
                    # Chunk results for adversarial cases
                    chunk_results = {}
                    
                    # Iterate over sensor types
                    for st in sensor_types:
                        # Adjust common timestamp if we are dealing with bar
                        if st == 'bar':
                            cts_i = construct_bar_ts(cts)
                        else:
                            cts_i = cts
                        
                        # Working on a per chunk basis
                        if not merge or action == 'replay':
                            chunk_results[st] = results[st][cts_i][sn] 
                        else:
                            chunk_results[st] = results[st][cts_i]                             
                    
                    # Get common timestamps for all chunks checked against a single chunk
                    common_ts_ch = get_common_timestamps(chunk_results, 1)
                    
                    # Count number of bad chunks checked against a single chunk 
                    n_bad_chunks = 0
                    
                    # Iterate over common timestamps for chunks
                    for cts_ch in common_ts_ch:
                        # Keep track of the number of bits above or below error correction threshold
                        ext_bits = 0
                        req_bits = 0
                        
                        # Iterate over sensor types
                        for st in sensor_types:
                            # Adjust common timestamp if we are dealing with bar
                            if st == 'bar':
                                cts_ch_i = construct_bar_ts(cts_ch, 1)
                            else:
                                cts_ch_i = cts_ch
                            
                            # Compute number of error correction bits defined by the threshold and those that chunk really has
                            n_ecc_bits_thr = ceil(key_params[st][1] * key_params[st][0] / 100)
                            n_ecc_bits_res = int(key_params[st][1] * chunk_results[st][cts_ch_i]['sim'] / 100)

                            # Check if fingerprint similarity is above or below threshold
                            if chunk_results[st][cts_ch_i]['sim'] >= key_params[st][0]:
                                ext_bits += n_ecc_bits_res - n_ecc_bits_thr
                            else:
                                req_bits += n_ecc_bits_thr - n_ecc_bits_res
                        
                        # Check if the fused chunk can be corrected wtih ECC
                        if ext_bits >= req_bits:
                            n_bad_chunks += 1
                    
                    # Check if we are hitting a border case
                    if len(common_ts_ch) > 0:
                        error_rates.append(n_bad_chunks / len(common_ts_ch))
                    else:
                        error_rates.append('n/a')
            
            # Display results here
            if action == 'benign':
                # Check if we are hitting a border case
                if len(common_ts) > 0:
                    # Compute True Accept Rate in %
                    tar = n_good_chunks / len(common_ts) * 100
                    print('%s,%.3f,%d' % (filename, tar, len(common_ts)))
                else:
                    # Compute True Accept Rate in %
                    tar = 'n/a'
                    print('%s,%s,%d' % (filename, tar, len(common_ts)))
                
                # Cache results if necessary 
                if merge and cache:
                    # Add error rates and number of chunks
                    cached_error_rates[filename] = {'tar': tar, 'tar_chunks': len(common_ts)}
                    cached_error_rates[filename].update(metadata)

            elif action == 'baseline' or action == 'replay': 
                # Check if we are hitting a border case
                if len(common_ts) > 0:
                    # Check if we are hitting another border case
                    if 'n/a' in error_rates:
                        # Compute False Accept Rate in %
                        far = 'n/a'
                        print('%s,%s,%d,%d' % (filename, far, len(common_ts), len(common_ts_ch)))
                    else:
                        # Compute False Accept Rate in %
                        far = np.mean(np.array(error_rates)) * 100
                        print('%s,%.3f,%d,%d' % (filename, far, len(common_ts), len(common_ts_ch)))
                else:
                    # Compute False Accept Rate in %
                    far = 'n/a'
                    common_ts_ch = []
                    print('%s,%s,%d,%d' % (filename, far, len(common_ts), len(common_ts_ch)))
                
                # Cache results if necessary
                if merge and cache:
                    # Check if we have 'x' in the file name, remove it
                    if 'x' in filename:
                        filename = filename.replace('x', '')
                    
                    # Add error rates and number of chunks
                    cached_error_rates[filename] = {'far': far, 'far_chunks': len(common_ts), 'far_sub_chunks': len(common_ts_ch)}
                    cached_error_rates[filename].update(metadata)

            # Cache fingerprints
            if action == 'benign':
                save_json_file({'metadata': metadata_cached_fps, 'results': cached_fps}, fp_path, filename)
            
            # Increment f_idx
            f_idx += 1

        if action == 'benign':
            print('%.2f' % compute_avg_error_rate(cached_error_rates))
        print() 
    
    # Store cached results
    if merge and cache:
        # Check if the scenario is provided or it is the full case
        if scenario is None:
            scenario = 'full'
        
        # Save json file
        save_json_file({'results': cached_error_rates}, err_path, scenario)


def fuse_fps(filepath, sensor_types, fp_len='full'):
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('fuse_fps: %s is not a valid path!' % (filepath, ))
        return
    
    # Check if sensor_types is a list of len at least 2
    if not isinstance(sensor_types, list) or len(sensor_types) < 2:
        print('fuse_fps: %s must be a list of at least length 2!' % (sensor_types,))
        return
    
    # Check if we are using correct data path
    if not 'keys' in filepath:
        print('Warning: double check if filepath points to "keys" log files!')
        return
    
    # Check if fp_len is valid
    if fp_len != 'full' and fp_len != 'reduced':
        print('fuse_fps: unknown fp_len "%s", can only be "full" or "reduced"!' % (fp_len,))
        return
    
    # Extract experiment name
    exp = filepath.split('keys')[1].split('/')[1]
    
    # Create part of the path showing sensor fusion
    st_path = ''
        
    # Iterate over sensor_types
    for st in sensor_types:
        if st_path:
            st_path += '-' + st
        else:
            st_path += st
    
    # Path to store fused fingerprints
    fps_path = FP_PATH + '/' + fp_len + '/fused/' + exp.split('-')[0] + '/' + st_path
    
    # Create ptime_path if it does not exist
    if not os.path.exists(fps_path):
        os.makedirs(fps_path)
    
    # Dict to store all log files to be used in evaluation
    log_files = {}
    
    # Iterate over sensor_types
    for st in sensor_types:
        # Get filepath for each sensor type
        st_filepath = filepath + '/' + st
        
        # Read json.gz files under st_filepath
        log_files[st] = sorted(glob(st_filepath + '/'  + '*.json.gz', recursive=True))
    
    # Index to track json files to be merged
    idx = 0
    
    # List to store cached resutls
    cached_fps = []
    
    # Iterate over json files
    for json_file in log_files[sensor_types[0]]:
        
        # Dict to store results of json files
        results = {}
        
        # Get file name 
        regex = re.escape(sensor_types[0]) + r'(?:/|\\)(.*)\.json.gz'
        match = re.search(regex, json_file)

        # If there is no match - exit
        if not match:
            print('fuse_fps: no match for the file name %s using regex %s!' % (json_file, regex))
            return
        
        # Current file name
        filename = match.group(1).split('/')[-1]

        # Initilaize list of files
        json_merge = [json_file]
        
        # Add remaining files to be merged
        json_merge.extend(get_jsons_to_merge(json_file, 'keys'))
        
        # Add remaining sensor types to json_merge
        for i in range(1, len(sensor_types)):
            json_merge.append(log_files[sensor_types[i]][idx])                    
            json_merge.extend(get_jsons_to_merge(log_files[sensor_types[i]][idx], 'keys'))

        # Initiate a pool of workers, use all available cores
        pool = Pool(processes=cpu_count(), maxtasksperchild=1)
        
        # Use partial to pass static parameters
        func = partial(process_json_file, action='keys', scenario=None, fuse=True)

        # Let the workers do their job, we convert data dict to list of tuples
        merged_results = pool.map(func, json_merge)

        # Wait for processes to terminate
        pool.close()
        pool.join()
        
        # Iterate over merged results             
        for mr in merged_results:
            if mr[1] in results:
                # Update results
                results[mr[1]].update(mr[0])

            else:
                # Add results
                results[mr[1]] = mr[0]
        
        # Get common timestamps
        common_ts = get_common_timestamps(results)
        
        # Iterate over common timestamps
        for cts in common_ts:
            # Vars to store a fused fingerprint
            fp = ''

            # Iterate over sensor types
            for st in sensor_types:
                # Adjust common timestamp if we are dealing with bar
                if st == 'bar':
                    cts_i = construct_bar_ts(cts)
                else:
                    cts_i = cts

                # Construct a fused fingerprint
                fp += results[st][cts_i]
                
            # Add a fingerprint to the output list 
            cached_fps.append(fp)
        
        # Increment index
        idx += 1
    
    # Open file for writing
    with open(fps_path + '/results.txt', mode='w') as f:
        f.write('\n'.join(cached_fps))
        f.write('\n')
    
    
def extract_fingerptins(filepath, sensor_type, fp_len='full', n_keys=0):
    # Check if filepath exists
    if not os.path.exists(filepath):
        print('extract_fingerptins: %s is not a valid path!' % (filepath, ))
        return
    
    # Check if provided sensor type is valid
    if not isinstance(sensor_type, str) or sensor_type not in ALLOWED_SENSORS:
        print('extract_fingerptins: unknown sensor type "%s", only %s are allowed!' % (sensor_type, ALLOWED_SENSORS))
        return
    
    # Check if we are using correct data path
    if not 'keys' in filepath:
        print('Warning: double check if filepath points to "keys" log files!')
        return
    
    # Check if fp_len is valid
    if fp_len != 'full' and fp_len != 'reduced':
        print('extract_fingerptins: unknown fp_len "%s", can only be "full" or "reduced"!' % (fp_len,))
        return
    
    # Read files under filepath
    json_files = glob(filepath + '/' + sensor_type  + '/*.json.gz', recursive=True)
    
    # Sort read files
    json_files.sort()
    
    # Dictionary to store fingerprints
    fps_results = {}
    
    # Count overall number of fingerprints
    counter = 0
    
    # Display input file and sensor type
    print(filepath, sensor_type)
    
    # Iterate over json files
    for json_file in json_files:
        # Load a file
        results, _ = load_gz_json_file(json_file)
        
        # Get file name 
        regex = re.escape(sensor_type) + r'(?:/|\\)(.*)\.json.gz'
        match = re.search(regex, json_file)

        # If there is no match - exit
        if not match:
            print('extract_fingerptins: no match for the file name %s using regex %s!' % (json_file, regex))
            return

        # Current file name
        filename = match.group(1).split('/')[-1]
        
        # List to store fingerprints from a json_file
        fps = []
        
        # Iterate over results
        for k,v in sorted(results.items()):
            if n_keys == 0:
                # Add a single key
                fps.append(v['01']['fp'])
            else:
                # Add every nth key (n = n_keys)
                key_ids = sorted(list(v.keys()))
                
                for kidx in key_ids[0::n_keys]:
                    if kidx == '100':
                        fps.append(v['11']['fp'])
                    else:
                        fps.append(v[kidx]['fp'])
        
        # Add fingerprint results 
        fps_results[filename] = fps
        
        # Display sensor and number of fingerprints 
        print(filename, len(fps))
        
        # Update coutner
        counter += len(fps)
    
    # Show counter value
    print()
    print(counter)
    
    # Create save path
    n_keys_folder = ''
 
    if fp_len == 'full':
        if n_keys == 0:
            n_keys_folder = '1x'
        elif n_keys == 20 or n_keys == 8:
            n_keys_folder = '5x'
        elif n_keys == 10 or n_keys == 4:
            n_keys_folder = '10x'
        else:
            print('extract_fingerptins: unsupported n_keys = %d in the "full" case!' % (n_keys,))
            return
    else:
        if n_keys == 0:
            n_keys_folder = '1x'
        elif n_keys == 2:
            if sensor_type == 'gyrW' or sensor_type == 'bar':
                n_keys_folder = '5x'
            else:
                n_keys_folder = '10x'
        elif n_keys == 4:
            n_keys_folder = '5x'
        elif n_keys == 1:
            n_keys_folder = '10x'
        else:
            print('extract_fingerptins: unsupported n_keys = %d in the "reduced" case!' % (n_keys,))
            return
        
    # Resulting save path
    save_path = FP_PATH + '/' + fp_len + '/' + n_keys_folder
    
    # Check if file exists
    if os.path.isfile(save_path + '/' + sensor_type + '.txt'):
        # We only need fps: convert from dict to a list
        fps_results = list(itertools.chain.from_iterable(list(fps_results.values())))
        
        # Append fps to the file
        with open(save_path  + '/' + sensor_type +'.txt', mode='a') as f:
            f.write('\n'.join(fps_results))
            f.write('\n')
    
    else:
        # Create filepath if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # We only need fps: convert from dict to a list
        fps_results = list(itertools.chain.from_iterable(list(fps_results.values())))
        
        # Open file for writing
        with open(save_path  + '/' + sensor_type +'.txt', mode='w') as f:
            f.write('\n'.join(fps_results))
            f.write('\n')
