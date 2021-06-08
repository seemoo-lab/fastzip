# Types of sensors supported ('unXXX' means uncalibrated sensor XXX values)
SENSOR_TYPES = ['acc', 'unacc', 'lacc', 'laccW', 'bar', 'grav', 'gyr',
                'gyrW', 'ungyr', 'hum', 'lux', 'mag', 'unmag', 'tmp']

# Sensor types with which we are working
ALLOWED_SENSORS = ['acc_v', 'acc_h', 'gyrW', 'unmag', 'bar']

# Sensor numbers located in Car 1 and 2
CAR1 = ['01', '02', '03', '04', '05']
CAR2 = ['06', '07', '08', '09', '10']

# Sampling rates (in Hz) for sensors
ACC_FS = 100
GYR_FS = 100
MAG_FS = 50
BAR_FS = 10

# Lengths of data chunks to be used for fingerprint generation (in seconds)
FP_ACC_CHUNK = 10
FP_GYR_CHUNK = 10
FP_MAG_CHUNK = 20
FP_BAR_CHUNK = 20

# Addition to a signal in sec to perform xcorr, we want to end up with a fp_chunk lenght
XCORR_ACC = 2
XCORR_GYR = 1
XCORR_BAR = 4

# Bands to split magnetometer data, it depends on the 1) sampling frequency, and 2) it is a design choice;
# to make it uniform for 25 Hz spectrum, while filtering 1 Hz (Earth's magnetic field) we end up with 6 bands
MAG_BANDS = [(1, 6), (5, 10), (9, 14), (13, 18), (17, 22), 22]

# Names of experiments—used for setting experiment specific parameters
SIM_NON_ADV = 'sim-non-adv'
SIM_ADV = 'sim-adv'
DIFF_PARK = 'diff-park'
DIFF_NON_ADV = 'diff-non-adv'
DIFF_ADV = 'diff-adv'

# Paths where the data is stored
SIM_NON_ADV_PATH = '/home/seemoo/car_zip/exp-3/non-adv'
SIM_ADV_PATH = '/home/seemoo/car_zip/exp-3/adv'
DIFF_PARK_PATH = '/home/seemoo/car_zip/exp-4/aligned'
DIFF_NON_ADV_PATH = '/home/seemoo/car_zip/exp-5/non-adv'
DIFF_ADV_PATH = '/home/seemoo/car_zip/exp-5/adv'

# Experiments timeframes
# Similar, non-adv
SIM_PARKING1 = [1210, 2530]          # 2018-12-13 09:49:00 -- 10:11:00
SIM_CAR1_CITY = [2710, 5230]         # 2018-12-13 10:14:00 -- 10:56:00
SIM_CAR1_COUNTRY = [5230, 7090]      # 2018-12-13 10:56:00 -- 11:27:00
SIM_CAR1_HIGHWAY = [7090, -1]        # 2018-12-13 11:27:00 -- 12:15:00

SIM_CAR2_CITY = [3730, 6430]         # 2018-12-13 10:31:00 -- 11:16:00
SIM_CAR2_COUNTRY = [6430, 8230]      # 2018-12-13 11:16:00 -- 11:46:00
SIM_CAR2_HIGHWAY = [8230, -1]        # 2018-12-13 11:46:00 -- 12:40:00

# Similar, adv
SIM_CAR12_COUNTRY = [0, 2240]        # 2018-12-13 13:59:39 -- 14:37:00
SIM_CAR12_CITY1 = [2240, 6920]       # 2018-12-13 14:37:00 -- 15:55:00
SIM_CAR12_HIGHWAY = [6920, 9200]     # 2018-12-13 15:55:00 -- 16:33:00
SIM_CAR12_CITY2 = [9200, 10220]      # 2018-12-13 16:33:00 -- 16:50:00
SIM_PARKING2 = [10880, 11540]        # 2018-12-13 17:01:00 -- 17:12:00

# Different, parking
DIFF_PARKING1 = [530, 1790]          # 2018-12-17 15:35:00 -- 15:56:00
DIFF_PARKING2 = [1910, 2810]         # 2018-12-17 15:58:00 -- 16:13:00
# CAR1_PARKING_EXTRA = [2810, 3050]    # 2018-12-17 16:13:00 -- 16:17:00

# Different, non-adv
DIFF_CAR1_CITY = [905, 3785]         # 2018-12-18 07:29:00 -- 08:17:00
DIFF_CAR1_COUNTRY = [3785, 6065]     # 2018-12-18 08:17:00 -- 08:55:00
DIFF_CAR1_HIGHWAY = [6065, -1]       # 2018-12-18 08:55:00 -- 10:07:00

DIFF_CAR2_CITY = [1685, 4865]        # 2018-12-18 07:42:00 -- 08:35:00
DIFF_CAR2_COUNTRY = [4865, 7025]     # 2018-12-18 08:35:00 -- 09:11:00
DIFF_CAR2_HIGHWAY = [7025, -1]       # 2018-12-18 09:11:00 -- 09:53:00

# Different, adv
DIFF_CAR12_COUNTRY = [0, 1620]       # 2018-12-18 10:37:00 -- 11:04:00
DIFF_CAR12_CITY1 = [1620, 3360]      # 2018-12-18 11:04:00 -- 11:33:00
DIFF_CAR12_HIGHWAY = [3360, 5880]    # 2018-12-18 11:33:00 -- 12:15:00
DIFF_CAR12_CITY2 = [5880, -1]        # 2018-12-18 12:15:00 -- 12:23:50

# Action names to be used when iterating over the data
AR = 'AR'
XCORR = 'xcorr'
KEYS = 'keys'

# Evaluation scenarios, i.e., benign and different type of adversarial
EVAL_SCENARIOS = ['benign', 'baseline', 'replay', 'stalk']

# Subscenarios
SUBSCENARIOS = ['city', 'country', 'highway', 'parking']

# Fusion config
FUSION_CONFIG = [['acc_v', 'acc_h'], ['acc_v', 'gyrW'], ['acc_v', 'bar'], ['acc_h', 'gyrW'], ['acc_h', 'bar'], ['gyrW', 'bar'], 
                ['acc_v', 'acc_h', 'gyrW'], ['acc_v', 'acc_h', 'bar'], ['acc_v', 'gyrW', 'bar'], ['acc_h', 'gyrW', 'bar'], 
                ['acc_v', 'acc_h', 'gyrW', 'bar']]

# Paths to store results
RESULTS_PATH = '/home/seemoo/car_zip/logs'
CACHE_PATH = '/home/seemoo/car_zip/cache'
PLOT_PATH = '/home/seemoo/car_zip/plots'
FP_PATH = '/home/seemoo/car_zip/fps'