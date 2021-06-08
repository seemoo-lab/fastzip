import numpy as np

# Supported normalization techiques
NORM_TYPES = ['energy', 'zscore', 'meansub', 'minmax']


def normalize_signal(sig, norm):
    # Check if sig is non-zero
    if len(sig) == 0:
        print('normalize_signal: signal must have non-zero length!')
        return
    
    if not isinstance(norm, str) or norm not in NORM_TYPES:
        print('normalize_signal: provide one of the supported normalization methods %s as a string paramter!' % NORM_TYPES)
        return
    
    # Noramlized signal to be returned
    norm_sig = np.copy(sig)
    
    # Check how signal should be normalized
    if norm == 'energy':
        # Perform energy normalization
        norm_sig = norm_sig / np.sqrt(np.sum(norm_sig ** 2))

    elif norm == 'zscore':
        # Perform z-score normalization (also knonw as variance scaling)
        if np.std(norm_sig) != 0:
            norm_sig = (norm_sig - np.mean(norm_sig)) / np.std(norm_sig)
        else:
            print('normalize_signal: cannot perform Z-score normalization, STD is zero --> returning the original signal!')
            return sig
        
    elif norm == 'meansub':
        # Subtract mean from sig
        norm_sig = norm_sig - np.mean(norm_sig)
        
    elif norm == 'minmax':
        # Perform min-max normalization
        if np.amax(norm_sig) - np.amin(norm_sig) != 0:
            norm_sig = (norm_sig - np.amin(norm_sig)) / (np.amax(norm_sig) - np.amin(norm_sig))
        else:
            print('normalize_signal: cannot perform min-max normalization, min == max --> returning the original signal!')
            return sig
    
    return norm_sig
