

import os
import numpy as np


def generate_input_output_vectors(num_vectors, num_devices, num_dnn, embedding_size):
    """Generate input and output vectors.
    
    Parameters
    ----------
    num_vectors : int
        The number of vectors to generate.
    num_devices : int
        The number of devices.
    num_dnn : int
        The number of Deep Neural Networks per channel.
    embedding_size : int
        The size of the embedding.
    
    Returns
    -------
    input_vectors : numpy.ndarray
        The input vectors.
    output_vectors : numpy.ndarray
        The output vectors.
    """
    # Generate input vectors
    input_vectors = np.random.uniform(
        0, 1, (num_vectors, num_devices, num_dnn, embedding_size + 1)).astype(np.float32)
    output_vectors = np.random.uniform(
        0, 1, (num_vectors, num_devices)).astype(np.float32)
    return input_vectors, output_vectors


def store_input_output_vectors(input_vectors, output_vectors, root_dir):
    """Store the input and output vectors in a npy file.
    
    Parameters
    ----------
    input_vectors : numpy.ndarray
        The input vectors.
    output_vectors : numpy.ndarray
        The output vectors.
    root_dir : str
        The root directory.
    
    """
    for i, (input_vector, output_vector) in enumerate(zip(input_vectors, output_vectors)):
        np.save(os.path.join(root_dir, "mapping", f"{i}.npy"), input_vector)
        np.save(os.path.join(root_dir, "workload", f"{i}.npy"), output_vector)

def make_dirs(root_dir):
    """Make the directories.
    
    Parameters
    ----------
    root_dir : str
        The root directory.
    
    """
    for split_dir in ["train", "valid", "test"]:
        for dir_name in ["mapping", "workload"]:
            os.makedirs(os.path.join(root_dir, split_dir, dir_name), exist_ok=True)

def main():
    num_vectors = [500, 100, 100]
    num_devices = 3
    num_dnn = 11
    embedding_size = 35
    root_dir = "../data/demo/"
    split_dirs = ["train", "valid", "test"]
    make_dirs(root_dir)
    for num_vectors_subset, split_dir in zip(num_vectors, split_dirs):
        input_vectors, output_vectors = generate_input_output_vectors(num_vectors_subset, num_devices, num_dnn, embedding_size)
        store_input_output_vectors(input_vectors, output_vectors, os.path.join(root_dir, split_dir))
    

if __name__ == "__main__":
    main()
