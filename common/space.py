
import numpy as np

from typing import Tuple, Dict


def get_models():
    return {
        0: "graph_alexnet_mine",
        1: "graph_mobilenet_mine",
        2: "graph_resnet34_mine",
        3: "graph_resnet50_mine",
        4: "graph_resnet101_mine",
        5: "graph_vgg13_mine",
        6: "graph_vgg16_mine",
        7: "graph_vgg19_mine",
        8: "graph_squeezenet_mine",
        9: "graph_inception_v3_mine",
        10: "graph_inception_v4_mine"
    }


def dnn_factory():
    """Generate a DNN mapper.

    Returns
    -------
    numpy.ndarray
        The DNN mapper.
    """
    dnn_mapper = np.array([
        "graph_alexnet_mine",
        "graph_mobilenet_mine",
        "graph_resnet34_mine",
        "graph_resnet50_mine",
        "graph_resnet101_mine",
        "graph_vgg13_mine",
        "graph_vgg16_mine",
        "graph_vgg19_mine",
        "graph_squeezenet_mine",
        "graph_inception_v3_mine",
        "graph_inception_v4_mine"
    ])
    return dnn_mapper


def get_num_layers():
    return {
        "graph_alexnet_mine": 8,
        "graph_mobilenet_mine": 15,
        "graph_resnet34_mine": 12,
        "graph_resnet50_mine": 18,
        "graph_resnet101_mine": 34,
        "graph_vgg13_mine": 13,
        "graph_vgg16_mine": 16,
        "graph_vgg19_mine": 19,
        "graph_squeezenet_mine": 18,
        "graph_inception_v3_mine": 17,
        "graph_inception_v4_mine": 23,
    }


def layer_factory():
    """Define the number of layers of the used models for this dataset.
    
    Returns
    -------
    np.ndarray
        The number of layers of each model.
    """
    layers = np.array([8, 15, 12, 18, 34, 13, 16, 19, 18, 17, 23])
    return layers


def get_devs():
    return {
        0: "L",
        1: "G",
        2: "B",
    }


def device_factory():
    """Generate a device mapper.

    Returns
    -------
    np.ndarray
        The device mapper.
    """
    device_mapper = np.array([
        "L",
        "G",
        "B"
    ])
    return device_mapper


# TODO: move to config file
def dev2dev_mapper() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate a device to device mapper. This 
    mapper is used to map the device names generated
    by the random device order generator to
    the device names used in the simulator.

    Returns
    -------
    Dict[str, str]
        The device to device mapper.
    """
    dev2dev = {
        'L': 'little',
        'G': 'gpu',
        'B': 'big'
    }
    inv_mapper = {v: k for k, v in dev2dev.items()}
    return inv_mapper


def zip_model_layer_dict():
    """Create a dictionary of model names and their corresponding number of layers.

    Returns
    -------
    dict
        The dictionary of model names and their corresponding number of layers.
    """
    model_layer_dict = {}
    for idx, model in enumerate(dnn_factory()):
        model_layer_dict[model] = layer_factory()[idx]
    return model_layer_dict
