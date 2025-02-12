import os
import numpy as np

import os
import numpy as np

def process_across_files(directory_path):
    arrays = []

    # Load all files starting with "first_layer_output"
    for filename in sorted(os.listdir(directory_path)):
        if filename.startswith("imagenc_firstlayer_output") and filename.endswith(".npy"):
            file_path = os.path.join(directory_path, filename)
            arr = np.load(file_path).flatten()
            arrays.append(arr)

    # Convert list of arrays into a 3D NumPy array (stack along a new axis)
    stacked_arrays = np.stack(arrays, axis=0)  # Shape: (num_files, array_shape...)

    # Compute variance and standard deviation across files (axis=0)
    variance_across_files = np.var(stacked_arrays, axis=0)

    print(f"Variance shape: {variance_across_files.shape}")

    return variance_across_files

# Example usage
directory_path = "/Users/dannybarash/Code/oct/AE_experiment/data_of_oct"
variance_across_files  = process_across_files(directory_path)
print(f"variance: {variance_across_files.mean()}")
print(variance_across_files)
