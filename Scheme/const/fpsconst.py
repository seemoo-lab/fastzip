# Number of fps per chunk
FPS_PER_CHUNK = 1000

# Similarity thresholds, i.e., how many bits are allowed to be different for co-located devices
SIM_THR_ACC_V = 70
SIM_THR_ACC_H = 75
SIM_THR_GYR = 90
SIM_THR_BAR = 90

# Biases for acc_v and acc_h when median quantization method is applied
BIAS_ACC_V = 0.00015
BIAS_ACC_H = 0.0001
BIAS_GYR = 0
BIAS_BAR = 0

# Number of fingerprint bits generated per modality
BITS_ACC = 24
BITS_GYR = 16
BITS_BAR = 12

# Step (in samples) for generating a corpus of the keys from a chunk using shifting
DELTA_ACC = 10       # 0.1 sec
DELTA_GYR = 25       # 0.25 sec
DELTA_BAR = 5        # 0.5 sec