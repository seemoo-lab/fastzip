import numpy as np
from math import ceil
import secrets
from common.helper import compute_hamming_dist
from const.fpsconst import *

# Get systemRandom class instance out of secrets module
secretsGenerator = secrets.SystemRandom()


# ToDo: sorting is done here, adjust it if necessary
def generate_random_points(chunk_len, n_bits, strat='random'):
    if strat == 'random':
        return np.array([secretsGenerator.randint(0, chunk_len - 1) for x in range(n_bits)])
#         return np.array(sorted([secretsGenerator.randint(0, chunk_len - 1) for x in range(n_bits)]))
    elif strat == 'equid-start':
        # Default values for guard and increment
        guard = 0
        inc = 1
        
        # To cover the most of the signal with points we use a guard interval (start, end) and increment (distance between poitns)
        if n_bits == BITS_ACC:
            guard = 5
        elif n_bits == BITS_GYR:
            guard = 5
            inc = 3
        elif n_bits == BITS_BAR:
            guard = 5
            inc = 0
            
        # Cases for reduced number of bits
        if n_bits == BITS_ACC / 2 + 1:
            return np.arange(8, 1000, 82)
        
        elif n_bits == BITS_GYR / 2 + 1:
            return np.arange(4, 1000, 124)
            
        elif n_bits == BITS_BAR / 2 + 1:
            return np.arange(4, 200, 32)
        
        return np.arange(0 + guard, chunk_len + guard, ceil(chunk_len / n_bits) + inc)
        
    elif strat == 'equid-end':
        return np.arange(chunk_len - 1, 0, -ceil(chunk_len / n_bits))
    elif strat == 'equid-midleft' or strat == 'equid-midright':
        # Generate left and right planes with equidistant points
        left_side = np.arange(int(chunk_len / 2), 0, -ceil(chunk_len / n_bits))
        right_side = np.arange(int(chunk_len / 2), chunk_len, ceil(chunk_len / n_bits))
        
        if strat == 'equid-midleft':
            return construct_equidist_rand_points(left_side, right_side)
        else:
            return construct_equidist_rand_points(right_side, left_side)
        
        
def construct_equidist_rand_points(eqd_points1, eqd_points2):
    # Check that length of equidistant arrays is the same
    if len(eqd_points1) != len(eqd_points2):
        print('construct_equidist_rand_points: input arrays must have the same length!')
        return
    
    # Array storing random points
    random_points = np.zeros(len(eqd_points1) * 2, dtype=int)
    
    # Index to populate random points
    idx = 0
    
    # Iterate over equidistant points
    for i in range(0, len(eqd_points1)):
        # Point in the middle is the same for both equ_point arrays
        if i == 0:
            random_points[idx] = eqd_points1[i]
        else:
            # Take one element from the 1st array and the second element from the second array and put the consequently
            random_points[idx + 1] = eqd_points1[i]
            random_points[idx + 2] = eqd_points2[i]
            
            # Move on with idx
            idx += 2
            
    return random_points


def generate_equidist_rand_points(chunk_len, step, eqd_delta):
    # Equidistant delta cannot be bigger than the step
    if eqd_delta > step:
        print('generate_equidist_rand_points: "eqd_delta" must be smaller than "step"')
        return -1, 0
    
    # Store equidistant random points
    eqd_rand_points = []
    
    # Generate equdistant points
    for i in range(0, ceil(chunk_len / eqd_delta)):
        eqd_rand_points.append(np.arange(0 + eqd_delta * i, chunk_len + eqd_delta * i, step) % chunk_len)
        
    return eqd_rand_points, len(eqd_rand_points)
    
      
def generate_fingerprint(chunk, random_points, qs_thr):
    # Fingerprint to be returned
    fp = []
    
    # Iterate over random points
    for i in range(0, len(random_points)):
        if chunk[random_points[i]] > qs_thr:
            fp.append('1')
        else:
            fp.append('0')
            
    return ''.join(fp)


def generate_fps_corpus(chunk1, chunk1_qs_thr, chunk2, chunk2_qs_thr, n_bits, n_iter=FPS_PER_CHUNK, eqd_delta=-1):
    # Initialize vars for computing a corpus of fingerprints
    fps1 = ''
    fps2 = ''
    sim = []
    rps = []
    eqd_flag = False
    
    # ToDo: this is just for testing, remove it later on
    rps_strat = ['equid-start', 'equid-end', 'equid-midleft', 'equid-midright']

    # Check if equidistant delta is valid
    if isinstance(eqd_delta, int) and eqd_delta > 0:
        
        # Use proper increment (distance between points) for each modality
        if n_bits == BITS_ACC:
            inc = 1
        elif n_bits == BITS_GYR:
            inc = 3
        elif n_bits == BITS_BAR:
            inc = 0
         
        # Cases for reduced number of bits
        if n_bits == BITS_ACC / 2 + 1:
            inc = 0
            eqd_delta = 50
        
        elif n_bits == BITS_GYR / 2 + 1:
            inc = 0
            eqd_delta = 100
            
        elif n_bits == BITS_BAR / 2 + 1:
            inc = 0
            eqd_delta = 20
        
        # Generate a corpus of equidistant random points
        eqd_rand_points, n_iter = generate_equidist_rand_points(len(chunk1), ceil(len(chunk1) / n_bits) + inc, eqd_delta)
        
        # Set equidist flag
        eqd_flag = True
    
    # Generate a number of fingerprtins from a single chunk
    for i in range(0, n_iter):
        # Generate random x-axis points (time)
        if not eqd_flag:
            rand_points = generate_random_points(len(chunk1), n_bits, rps_strat[0])
        else:
            rand_points = eqd_rand_points[i]

        # Generate fps
        fp1 = generate_fingerprint(chunk1, rand_points, chunk1_qs_thr)
        fp2 = generate_fingerprint(chunk2, rand_points, chunk2_qs_thr)

        # Compute similarity between two fingerprints 
        _, fp_sim = compute_hamming_dist(fp1, fp2)

        # Store random points
        rps.append(' '.join(str(x) for x in rand_points.tolist())) 
        
        # Store the similarity between two fingerpritns
        sim.append(fp_sim)
        
        # Append a chunk fingerprint to the corpus of fingerprints
        fps1 += ''.join(fp1)
        fps2 += ''.join(fp2)
        
    return fps1, fps2, np.array(sim), rps


def generate_fps_corpus_chunk(chunk, chunk_qs_thr, n_bits, n_iter=FPS_PER_CHUNK, eqd_delta=-1):
    # Initialize vars for computing a corpus of fingerprints
    fps = []
    rps = []
    eqd_flag = False
    
    # ToDo: this is just for testing, remove it later on
    rps_strat = ['equid-start', 'equid-end', 'equid-midleft', 'equid-midright']
    
    # Check if equidistant delta is valid
    if isinstance(eqd_delta, int) and eqd_delta > 0:
        
        # Use proper increment (distance between poitns) for each modality
        if n_bits == BITS_ACC:
            inc = 1
        elif n_bits == BITS_GYR:
            inc = 3
        elif n_bits == BITS_BAR:
            inc = 0
        
        # Cases for reduced number of bits
        if n_bits == BITS_ACC / 2 + 1:
            inc = 0
            eqd_delta = 50
        
        elif n_bits == BITS_GYR / 2 + 1:
            inc = 0
            eqd_delta = 100
            
        elif n_bits == BITS_BAR / 2 + 1:
            inc = 0
            eqd_delta = 20
        
        # Generate a corpus of equidistant random points
        eqd_rand_points, n_iter = generate_equidist_rand_points(len(chunk), ceil(len(chunk) / n_bits) + inc, eqd_delta)
        
        # Set equidist flag
        eqd_flag = True
    
    # Generate a number of fingerprtins from a single chunk
    for i in range(0, n_iter):
        # Generate random x-axis points (time)
        if not eqd_flag:
            rand_points = generate_random_points(len(chunk), n_bits, rps_strat[0])
        else:
            rand_points = eqd_rand_points[i]
        
        # Generate fp
        fp = generate_fingerprint(chunk, rand_points, chunk_qs_thr)

        # Store random points
        rps.append(' '.join(str(x) for x in rand_points.tolist())) 
        
        # Append a chunk fingerprint to the corpus of fingerprints
        fps.append(''.join(fp))
        
    return fps, rps


def compute_qs_thr(chunk, bias):
    # Make a copy of chunk
    chunk_cpy = np.copy(chunk)
    
    # Sort the chunk
    chunk_cpy.sort()
    
    return np.median(chunk_cpy) + bias
    
    
def compute_zeros_ones_ratio(chunk1_fps, chunk2_fps):
    # Check if both fingerprint corpuses are strings
    if isinstance(chunk1_fps, str) and isinstance(chunk2_fps, str):
        # Fingerprint corpuses must have the same size
        if len(chunk1_fps) != len(chunk2_fps):
            print('compute_zeros_ones_ratio: input fingerprints must have the same length!')
            return 
    else:
        print('compute_zeros_ones_ratio: input fingerprints must be provided as binary stings, e.g., "010011100..."!')
        return 
    
    # Count number of 0s and 1s in a corpus of fingerprints
    n_ones1 = 0
    n_zeros1 = 0
    n_ones2 = 0
    n_zeros2 = 0
    
    # Iterate over fingerprints
    for i in range(0, len(chunk1_fps)):
        # Count number of 0s and 1s
        if chunk1_fps[i] == '1':
            n_ones1 += 1
        else:
            n_zeros1 += 1
        
        if chunk2_fps[i] == '1':
            n_ones2 += 1
        else:
            n_zeros2 += 1
            
    # Compute ratios between 0s and 1s
    r_ones1 = n_ones1 / (n_ones1 + n_zeros1) 
    r_zeros1 = n_zeros1 / (n_ones1 + n_zeros1)
    r_ones2 = n_ones2 / (n_ones2 + n_zeros2) 
    r_zeros2 = n_zeros2 / (n_ones2 + n_zeros2)
    
    return r_ones1, r_zeros1, r_ones2, r_zeros2
