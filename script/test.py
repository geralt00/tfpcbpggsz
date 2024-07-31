import tensorflow as tf
import numpy as np
import random
import time

def apply_efficiency_mask(arr, weights):
    """Applies a boolean mask while preserving the original array shape (uproot)."""
    max_weight = np.max(weights)
    random.seed(int(time.time()) % 10)
    np_seed = 121
    np.random.seed(np_seed)
    threshold = np.random.uniform(0, max_weight, len(weights))

    mask = weights >= threshold
    
    print(f'Efficiency: {mask.sum() / len(mask)}')
    # Create a new structured array with only the masked elements
    filtered_data = {name: arr[name][mask] for name in arr.dtype.names}
    return np.array(list(zip(*filtered_data.values())), dtype=arr.dtype) 


arr=np.array([1,1.2,1.3,1.4,1.1,1.3,1.4,1,1])
weights=np.array([1,1.2,1.3,1.4,1.1,1.3,1.4,1,1])
c_arr=apply_efficiency_mask(arr, weights)
print(c_arr)
