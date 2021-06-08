import numpy as np
from common.helper import check_tframe_boundaries, get_sensors_len, choose_bar_norm, compute_sig_diff, idx_to_str
from process.procdata import ewma_filter
from process.normdata import normalize_signal
from const.activityconst import *
from const.globconst import *


def align_adv_data(data, experiment):
    # Adjusted data to be returned
    aligned_data = dict(data)

    # Do the alignment depending on the experiment
    if experiment == SIM_ADV:
        # Check if we have 2D data (acc) or 1D—other modalities
        if len(list(aligned_data.values())[0].shape) == 2:
            # Iterate over sensors
            for k, v in sorted(aligned_data.items()):
                if k == '01' or k == '05':
                    # Get horizontal and vertical components
                    acc_v = v[:, 0]
                    acc_h = v[:, 1]

                    # Adjust horiz and vert acc based on the observed delays
                    if k == '01':
                        acc_v = np.concatenate((np.zeros(22), acc_v))
                        acc_h = np.concatenate((acc_h, np.zeros(22)))
                    else:
                        acc_v = np.concatenate((acc_v[40:], np.zeros(5)))
                        acc_h = acc_h[35:]

                    # Adjusted acc array
                    acc_adj = np.zeros((len(acc_v), 2))
                    acc_adj[:, 0] = acc_v.T
                    acc_adj[:, 1] = acc_h.T

                    # Update the corresponding sensor
                    aligned_data[k] = acc_adj

                elif k == '08' or k == '09':
                    # Get horizontal and vertical components
                    acc_v = v[:, 0]
                    acc_h = v[:, 1]

                    # Adjust horiz and vert acc based on the observed delays
                    if k == '08':
                        acc_v = np.concatenate((acc_v[45:], np.zeros(8)))
                        acc_h = acc_h[37:]
                    elif k == '09':
                        acc_v = np.concatenate((acc_v[88:], np.zeros(9)))
                        acc_h = acc_h[79:]

                    # Adjusted acc array
                    acc_adj = np.zeros((len(acc_v), 2))
                    acc_adj[:, 0] = acc_v.T
                    acc_adj[:, 1] = acc_h.T

                    # Update the corresponding sensor
                    aligned_data[k] = acc_adj
        else:
            # Iterate over sensors
            for k, v in sorted(aligned_data.items()):
                # Update the corresponding sensor
                if k == '05':
                    aligned_data[k] = v[40:]
                elif k == '08':
                    aligned_data[k] = v[37:]
                elif k == '09':
                    aligned_data[k] = v[77:]

    elif experiment == DIFF_ADV:
        # Check if we have 2D data (acc) or 1D—other modalities
        if len(list(aligned_data.values())[0].shape) == 2:
            # Iterate over sensors
            for k, v in sorted(aligned_data.items()):
                if k == '01':
                    # Get horizontal and vertical components
                    acc_v = v[:, 0]
                    acc_h = v[:, 1]
                    
                    # Adjust horiz and vert acc based on the observed delays
                    acc_v = np.concatenate((np.zeros(20), acc_v[:-20]))

                    # Adjusted acc array
                    acc_adj = np.zeros((len(acc_v), 2))
                    acc_adj[:, 0] = acc_v.T
                    acc_adj[:, 1] = acc_h.T

                    # Update the corresponding sensor
                    aligned_data[k] = acc_adj

                elif k == '06' or k == '08' or k == '09' or k == '10':
                    # Get horizontal and vertical components
                    acc_v = v[:, 0]
                    acc_h = v[:, 1]

                    # Adjust horiz and vert acc based on the observed delays
                    if k == '06':
                        acc_v = np.concatenate((np.zeros(43), acc_v[:-43]))
                        acc_h = np.concatenate((np.zeros(18), acc_h[:-18]))
                    elif k == '08':
                        acc_v = np.concatenate((np.zeros(30), acc_v[:-30]))
                        acc_h = np.concatenate((np.zeros(9), acc_h[:-9]))
                    elif k == '09':
                        acc_v = np.concatenate((np.zeros(14), acc_v[:-14]))
                    elif k == '10':
                        acc_v = np.concatenate((np.zeros(60), acc_v[:-60]))
                        acc_h = np.concatenate((np.zeros(14), acc_h[:-14]))

                    # Adjusted acc array
                    acc_adj = np.zeros((len(acc_v), 2))
                    acc_adj[:, 0] = acc_v.T
                    acc_adj[:, 1] = acc_h.T

                    # Update the corresponding sensor
                    aligned_data[k] = acc_adj
        else:
            # Iterate over sensors
            for k, v in sorted(aligned_data.items()):
                # Update the corresponding sensor
                if k == '07':
                    aligned_data[k] = v[44:]
                elif k == '08':
                    aligned_data[k] = v[11:]
                elif k == '09':
                    aligned_data[k] = v[27:]
                elif k == '10':
                    aligned_data[k] = np.concatenate((np.zeros(20), v[:-20]))

    return aligned_data


def get_xcorr_delay(sig1, sig2):
    # Check if sig1 and sig2 are valid
    if len(sig1) != len(sig2):
        print('get_xcorr_delay: two signals must have the same length to perform correlation!')
        return
    else:
        if len(sig1) <= 0:
            print('get_xcorr_delay: signals must have non-zero length!')
            return
    
    # Compute xcorr between two signals, the result should be similar to MATLAB's xcorr function
    xcorr = np.correlate(sig1, sig2, 'full')

    # Get the delay between two signals
    return int(np.argmax(xcorr) - int(xcorr.size / 2))


def display_xcorr_delays(data, sensor_type, sensors, tframe, norm=''):
    # Sampling rate depends on the sensor type
    if sensor_type == 'acc_v' or sensor_type == 'acc_h':
        fs = ACC_FS
        
        # Set the axis to select the correct acc component: vertical or horizontal 
        if sensor_type == 'acc_v':
            axis = 0
            alpha = EWMA_ACC_V
        elif sensor_type == 'acc_h':
            axis = 1
            alpha = EWMA_ACC_H
    
    elif sensor_type == 'gyrW':
        fs = GYR_FS
    elif sensor_type == 'unmag':
        fs = MAG_FS
    elif sensor_type == 'bar':
        fs = BAR_FS
    else:
        print('display_xcorr_delays: unknown sensor type "%s", only %s are allowed!' % (sensor_type, ALLOWED_SENSORS))
        return
    
    # Check if the provided sensors is a non-empty list
    if sensors and isinstance(sensors, list):
        # Check if all elements of sensors are strings
        for s in sensors:
            if not isinstance(s, str):
                print('display_xcorr_delays: %s is not a string in %s, must be a list of strings!' % (s, sensors))
                return
        
        # Remove duplicates from the list if exist
        sensors = list(set(sensors))
        
        # Sort the list
        sensors.sort()
        
        # Check if the provided sensors list is a subset of all the data
        if not set(sensors).issubset(list(data.keys())):
            print('display_xcorr_delays: provided "sensors" %s is not a subset of valid sensors %s!' % (sensors, list(data.keys())))
            return
    else:
        print('display_xcorr_delays: %s must be a list of strings!' % (sensors,))
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
       
    print('Computing "%s" xcorr delays:' % sensor_type)
    print()

    # Iterate over sensors
    for i in range(0, len(sensors)):
        for j in range(i + 1, len(sensors)):
            # Check if we have acc or gyr, compute xcorr between sensors
            if sensor_type == 'acc_v' or sensor_type == 'acc_h':
                # Smooth chunks with EWMA filter
                sig1 = ewma_filter(data[sensors[i]][begin_win:end_win, axis], alpha)
                sig2 = ewma_filter(data[sensors[j]][begin_win:end_win, axis], alpha)
                
                # Perform chunk normalization if necesary
                if norm:
                    sig1 = normalize_signal(data[sensors[i]][begin_win:end_win, axis], norm)
                    sig2 = normalize_signal(data[sensors[j]][begin_win:end_win, axis], norm)
            
            elif sensor_type == 'gyrW':
                # Process chunks if necessary
                sig1 = data[sensors[i]][begin_win:end_win]
                sig2 = data[sensors[j]][begin_win:end_win]
                
            elif sensor_type == 'bar':
                # Perform chunk normalization: 
                # we choose between 'meansub' and 'zscore', the one that gives smaller range has a lower RMSE
                sig1 = choose_bar_norm(data[sensors[i]][begin_win:end_win])
                sig2 = choose_bar_norm(data[sensors[j]][begin_win:end_win])
            
            # Compute xcorr and display the results
            print('%s-%s: %d' % (sensors[i], sensors[j], get_xcorr_delay(sig1, sig2)))
            
        print()


def align_two_signals(sig1, sig2, delay=None, limit=None):
    # Check if sig1 and sig2 are valid
    if len(sig1) != len(sig2):
        print('align_two_signals: two signals must have the same length to perform correlation!')
        return 
    else:
        if len(sig1) <= 0:
            print('align_two_signals: signals must have non-zero length!')
            return
    
    # Sanity checks for delay
    if delay is not None:
        if not isinstance(delay, int):
            print('align_two_signals: "delay" must be an integer value!')
            return
        
        if delay >= len(sig1):
            print('align_two_signals: "delay" must be smaller than signal length!')
            return
        
    # Sanity checks for limit
    if limit is not None:
        if not isinstance(limit, int):
            print('align_two_signals: "limit" must be an integer value!')
            return
        
        if limit <= 0:
            print('align_two_signals: "limit" must be > 0!')
            return
    
    # Aligned signals to be returned
    a_sig1 = np.copy(sig1)
    a_sig2 = np.copy(sig2)
    
    # Find a delay in samples between two signals
    if delay is None:
        delay1 = get_xcorr_delay(a_sig1, a_sig2)

    # Adjust signals according to delay
    if delay > 0:
        a_sig1 = a_sig1[delay:]
    elif delay < 0:
        a_sig2 = a_sig2[-delay:]
    
    # Find length of a shorter signal
    if limit is not None:
        sig_len = limit
    else:
        sig_len = min(len(a_sig1), len(a_sig2))

    a_sig1 = a_sig1[:sig_len]
    a_sig2 = a_sig2[:sig_len]

    return a_sig1, a_sig2


def align_and_compute_dist(sig1, sig2, fs, delay=None, sig_limit=None):
    # Align two signals
    s1, s2 = align_two_signals(sig1, sig2, delay, sig_limit)
    
    # Windows size is set to 10% of the signal
    win_size = int(len(s1) / fs * 0.1)
    
    # Chunk size is set to 20% of the signal
    chunk_size = int(len(s1) / fs * 0.2)
    
    # Compute number of chunks to iterate over
    n_chunks = int(len(s1) / (win_size * fs)) - 1

    # Output dictionary, containing distances for each chunk of the signal
    rmse = {}
    
    # Iterate over signal
    for i in range(0, n_chunks):
        # Get signal chunk
        s1_chunk = s1[i * win_size * fs:(i * win_size + chunk_size) * fs]
        s2_chunk = s2[i * win_size * fs:(i * win_size + chunk_size) * fs]
        
        # Compute RMSE error
        dist = compute_sig_diff(s1_chunk, s2_chunk, 'rmse')
        
        # Check if RMSE was correcly computed
        if isinstance(dist, int):
            return -1
                   
        # Save chunk's RMSE error
        rmse[str(idx_to_str(i * win_size)) + '-' + str(idx_to_str(i * win_size + chunk_size))] = dist
    
    return rmse


def fine_chunk_alignment(sig1, sig2, fs, fp_chunk, check=()):
    # Make copies of signals to be worked with
    a_sig1 = np.copy(sig1)
    a_sig2 = np.copy(sig2)
    delay = 0
    
    # Find a delay between two signals using xcorr (delay is searched on chunks of len = fp_chunk * fs) 
    delay1 = get_xcorr_delay(a_sig1[:fp_chunk * fs], a_sig2[:fp_chunk * fs])

    # Find a delay between two signals using xcorr (delay is searched on chunks of len = (fp_chunk + xcorr_add) * fs) 
    delay2 = get_xcorr_delay(a_sig1, a_sig2)
    
    # Check if we are hitting a weird cases
    if int(abs(delay1) / (fp_chunk * fs) * 100) > 20 and int(abs(delay2) / (fp_chunk * fs) * 100) > 20:
        # Somehow we are hitting a weird case in 'diff-non-adv' cars: acc_v ('01', '05'), chunk '2180->2190';
        # other alignment method shows that the delay is -48
        if check[0] == DIFF_NON_ADV and check[1] == 'acc_v' and check[3] == 2180 and check[4] == 2190:
            if check[2][0] == '01' and check[2][1] == '05':
                delay1 = -48
                delay2 = -48
            
            elif check[2][0] == '05' and check[2][1] == '01':
                delay1 = 48
                delay2 = 48
        
        # Same story with 'sim-adv' cars: bar ('01', '05'), chunk '2145->2165'; delay is -5
        elif check[0] == SIM_ADV and check[1] == 'bar' and check[3] == 2145 and check[4] == 2165:
            if check[2][0] == '01' and check[2][1] == '05':
                delay1 = -5
                delay2 = -5
                
            elif check[2][0] == '05' and check[2][1] == '01':
                delay1 = 5
                delay2 = 5
        else:
            print('fine_chunk_alignment: hitting case of abnormally large delays, check "%s"' % (check,))
            delay1 = 0
            delay2 = 0
    
    # If one delay is suspiciously larger than the other is probably the right one
    if int(abs(delay1) / (fp_chunk * fs) * 100) > 20 and int(abs(delay2) / (fp_chunk * fs) * 100) < 20:
        delay1 = delay2
        
    if int(abs(delay2) / (fp_chunk * fs) * 100) > 20 and int(abs(delay1) / (fp_chunk * fs) * 100) < 20:
        delay2 = delay1
        
    # Catching abnormal case from 'diff-adv' cars: bar ('07', '09'), chunk '5185->5205' 
    if check[0] == DIFF_ADV and check[1] == 'bar' and check[3] == 5185 and check[4] == 5205:
        if check[2][0] == '07' and check[2][1] == '09':
            delay1 = -39
            delay2 = -39
            
        elif check[2][0] == '09' and check[2][1] == '07':
            delay1 = 39
            delay2 = 39
    
    # Check if two delays are equal
    if delay1 == delay2:
        a_sig1, a_sig2 = align_two_signals(a_sig1, a_sig2, delay1, fp_chunk * fs)
        delay = delay1
        
    else:    
        # Using alignment with delay1 get sliding RMSE between signals
        rmse_delay1 = align_and_compute_dist(a_sig1, a_sig2, fs, delay1, fp_chunk * fs)

        # Using alignment with delay2 get sliding RMSE between signals
        rmse_delay2 = align_and_compute_dist(a_sig1, a_sig2, fs, delay2, fp_chunk * fs)

        # Check if RMSE computation was successful
        if isinstance(rmse_delay1, int) or isinstance(rmse_delay2, int):
            return

        # Delay is decided based on majority voting, minimizing the sliding RMSE
        rmse_maj_vote = [0, 0]

        # Iterate over sliding RMSE
        for k,v in sorted(rmse_delay1.items()):
            if v < rmse_delay2[k]:
                rmse_maj_vote[0] += 1
            else:
                rmse_maj_vote[1] += 1

        # Align signals based on an optimal delay
        if rmse_maj_vote[0] > rmse_maj_vote[1]:
            a_sig1, a_sig2 = align_two_signals(a_sig1, a_sig2, delay1, fp_chunk * fs)
            delay = delay1
        else:
            a_sig1, a_sig2 = align_two_signals(a_sig1, a_sig2, delay2, fp_chunk * fs)
            delay = delay2

    return a_sig1, a_sig2, delay
