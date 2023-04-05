
import os
import numpy as np
import torch.distributed as dist

from pathlib import Path
from typing import List, Tuple

from common.io import load_npy
from common.terminal import colorstr


def info():
    print(f"\n\n"
          f"\t\t           The {colorstr(['red', 'bold'], list(['Estimator']))} is a library that models \n"
          f"\t\t      heterogeneous architectures. The features of the \n"
          f"\t\t      architecture is learned. The goal is to use this\n"
          f"\t\t    library to predict the workload of each device on the\n"
          f"\t\t      working heterogenous platform to affect efficient\n"
          f"\t\t     task mapping using search techniques, such as Monte\n"
          f"\t\t                  Carlo Tree Search agents.\n"
          f"\n")
    

def recursive_items(dictionary):
    """Recursively iterate over a dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary to iterate over.

    Yields
    ------
    tuple
        Key, value pair.

    """
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
        else:
            yield (key, value)


def update_args(args, settings):
    """Update the arguments with the settings.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    settings : dict
        Dictionary of settings.

    Returns
    -------
    argparse.Namespace
        Updated arguments.
    """
    args_dict = vars(args)
    flattened_settings = dict(recursive_items(settings))
    for key, _ in recursive_items(args_dict):
        if key in flattened_settings:
            args.__dict__[key] = flattened_settings[key]
    return args


def is_dist_avail_and_initialized():
    """Check if distributed is available and initialized.

    Returns
    -------
    bool
        True if distributed is available and initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def confirm_paths(args):
    """Find required paths.
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments from the command line.
    
    Raises
    ------
    ValueError
        If any parsed path does not exist.
    """
    if not Path(args.out_data_dir).is_dir():
        raise ValueError(f"Logging directory is invalid. Value parsed {args.out_data_dir}.")
    if not Path(args.dataset_path).is_dir():
        raise ValueError(f"Path to dataset is invalid. Value parsed {args.dataset_path}.")
    if not os.path.isdir(os.path.join(args.dataset_path, "train", "mapping")):
        raise ValueError(f"Path to training image data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset_path, "train", "workload")):
        raise ValueError(f"Path to training label data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset_path, "valid", "mapping")):
        raise ValueError(f"Path to validation image data does not exist.")
    if not os.path.isdir(os.path.join(args.dataset_path, "valid", "workload")):
        raise ValueError(f"Path to validation label data does not exist.")


def get_embeddings(embeddings_directory: str) -> Tuple[np.ndarray, List[str]]:
    """Load the embeddings from a directory.
    
    Parameters
    ----------
    embeddings_directory : str
        Path to the directory.
    normalized : bool
        Whether to load the normalized version of the embeddings.
        If false, load the standardized version. Default: True.
    
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Loaded embeddings and the list with the device order.
    """
    embeddings_file = os.path.join(embeddings_directory, "embeddings_demo.npy")
    embeddings = load_npy(embeddings_file)
    return embeddings
