
import os
import json
import yaml
import pickle
import numpy as np
import pandas as pd

from typing import Dict, Tuple, List


def store_stats(stats: Dict[str, str], filename: str = 'stats.json'):
    """Store the experiment stats.

    Parameters
    ----------
    stats : Dict[str, str]
        The experiment stats.
    filename : str, optional
        The filename of the stats. The default is 'stats.json'.
    """
    with open(filename, 'w') as f:
        json.dump(stats, f)


def load_stats(filename: str = 'stats.json') -> Dict[str, str]:
    """Load the experiment stats.

    Parameters
    ----------
    filename : str, optional
        The filename of the stats. The default is 'stats.json'.

    Returns
    -------
    Dict[str, str]
        The experiment stats.
    """
    with open(filename) as f:
        stats = json.load(f)
    return stats


def store_3d_np_array(array: np.ndarray, filename: str = 'emb.xlsx', num_devices: int = 3):
    """Store a 3D numpy array.

    Parameters
    ----------
    array : np.ndarray
        The array to be stored.
    filename : str
        The filename of the array.
    """
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    for i in range(num_devices):
        df = pd.DataFrame(array[i, :, :])
        df.to_excel(writer, sheet_name='dev%d' % i, index=False, header=False)

    writer.save()

    
def load_3d_np_array(emb_shape: Tuple[int], filename: str = 'emb.xlsx') -> np.ndarray:
    """Load a 3D numpy array.

    Parameters
    ----------
    filename : str
        The filename of the array.

    Returns
    -------
    np.ndarray
        The array.
    """
    # to read all sheets to a map
    xls = pd.ExcelFile(filename, engine='openpyxl')
    sheet_to_df_map = np.zeros(tuple(emb_shape))
    for idx, sheet_name in enumerate(xls.sheet_names):
        sheet_to_df_map[idx] = pd.read_excel(xls, sheet_name, index_col=None, header=None)
    return sheet_to_df_map


def read_dims(filename: str = 'dims.txt') -> List[int]:
    """Read the dimensions of the embeddings.

    Parameters
    ----------
    filename : str
        The filename of the embeddings.

    Returns
    -------
    List[int]
        The dimensions of the embeddings.
    """
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
    dims = []
    for line in lines:
        dims.append(int(line.split(' ')[0]))
    return dims


def store_dims(dims: List[int], filename: str = 'dims.txt'):
    """Store the dimensions of the embeddings.

    Parameters
    ----------
    dims : List[int]
        The dimensions of the embeddings.
    filename : str, optional
        The filename of the dimensions. The default is 'dims.txt'.
    """
    with open(filename, 'w') as f:
        for dim in dims:
            f.write(str(dim) + '\n')


# TODO: implement an exporter for the normalized benchmarking data
def benchmarking_export(benchmarkings: np.ndarray, mappings_dir: str, exp_list: List[str]):
    """Export the benchmarkings.

    Parameters
    ----------
    benchmarkings : np.ndarray
        The benchmarkings.
    mappings_dir : str
        The path to the mappings directory.
    """
    for exp_id, bench in zip(exp_list, benchmarkings):
        # create the directory if it doesn't exist
        if not os.path.exists(mappings_dir):
            os.makedirs(mappings_dir)
        with open(f"{mappings_dir}/{exp_id}.npy", "wb") as f:
            np.save(f, bench)


def import_benchmarkings(mappings_dir: str):
    """Import the benchmarkings.

    Parameters
    ----------
    mappings_dir : str
        The path to the mappings directory.

    Returns
    -------
    list
        The benchmarkings.
    """
    benchmarkings = []
    for file in os.listdir(mappings_dir):
        with open(f"{mappings_dir}/{file}", "rb") as f:
            benchmarkings.append(np.load(f))
    return benchmarkings


def save_pickle(filename: str, data: object):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def parse_configs(filepath: str):
    """Parse the config file and return a dictionary of settings.
    
    Parameters
    ----------
    filepath : str
        Path to the config file.
        
    Returns
    -------
    dict
        Dictionary of settings.
    """
    with open(filepath, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def load_npy(filename: str) -> np.ndarray:
    """Load a numpy array from a file.

    Parameters
    ----------
    filename : str
        Path to the file.

    Returns
    -------
    np.ndarray
        Loaded array.
    """
    return np.load(filename)
