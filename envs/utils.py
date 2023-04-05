
import numpy as np
import pandas as pd

from copy import deepcopy

from common.io import load_pickle
from common.space import dnn_factory
from lib.estimator.utils import get_embeddings


def idx2label_mapper(
    idx: np.ndarray, 
    mapping: np.ndarray = np.array([])
) -> np.ndarray:
    """Map the indices of a vector to the corresponding 
    labels.

    Parameters
    ----------
    idx : numpy.ndarray
        The vector with the random indices.
    mapping : numpy.ndarray
        The mapper.

    Returns
    -------
    numpy.ndarray
        The mapped vector.
    """
    return mapping[idx]


# TODO: move to config file
def emb_idx2dnn() -> np.ndarray:
    """Configure the order of the models in the dataset with
    respect to the embedding matrix. The order can be 
    observed in the `raw.csv` file that lies in the `code`
    directory of the embeddings.

    Returns
    -------
    List[str]
        The order of models as configured in the embeddings matrix.
    """
    emb2dnn_mapper = np.array([
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
    return emb2dnn_mapper


def get_swap_indices(a: np.ndarray = dnn_factory(), b: np.ndarray = emb_idx2dnn()):
    """Get the indices of a NumPy array that correspond 
    to the same elements of another NumPy array.

    Returns
    -------
    np.ndarray
        The array with the indices.
    """
    _lst = []
    for _model in b:
        _lst.append(np.argwhere(a == _model))
    return np.asarray(_lst)


def get_obs(
    mappings, partitions, workload,     # these will be configured for each action
    embeddings=None, devices=None,      # preload in hikey `OpenAI Gym Environment`
):
    if embeddings is None:
        embeddings = get_embeddings("../data/demo/")
    if devices is None:
        devices = pd.Index(['LITTLE', 'GPU', 'BIG'])
    return build_embedding(
        embeddings, devices, idx2label_mapper(mappings, devices), 
        partitions, idx2label_mapper(workload, dnn_factory()), 
        emb_idx2dnn()), idx2label_mapper(mappings, devices), idx2label_mapper(
            workload, dnn_factory())


def simulate(
        model,
        sample,
        # std_scaler_path: str = '../../data/demo/StandardScaler.pkl',
        # norm_scaler_path: str = '../../data/demo/MinMaxScaler.pkl',
        overwrite_num_classes: int = None):
    # load standard scaler
    # std_scaler = load_pickle(std_scaler_path)
    # load min-max scaler
    # norm_scaler = load_pickle(norm_scaler_path)

    # inference
    output = model.test(sample)

    # if not overwrite_num_classes:
    #     unscaled_output = norm_scaler.inverse_transform(output)
    #     unscaled_output = std_scaler.inverse_transform(unscaled_output)
    #     return unscaled_output / 1000
    # else:
    #     return output * 10000
    return output


def build_embedding(
    embeddings: np.ndarray,
    devices: pd.Index,
    order: np.ndarray,
    partitions: np.ndarray,
    models: np.ndarray,
    models2idx: np.ndarray
):
    """Build the embedding for the generated dataset sample.

    Parameters
    ----------
    embeddings : np.ndarray
        The embeddings.
    devices : pd.index
        The devices.
    order : np.ndarray
        The order of the devices.
    partitions : np.ndarray
        The partitions.
    models : np.ndarray
        The models.
    models2idx : np.ndarray
        The pre-configured model order inside the 
        embeddings matrix.
    """
    # initialize the embedding matrix
    _embedding = np.zeros(embeddings.shape)
    for idx, model in enumerate(models):
        for device in order[idx]:
            dev_idx = np.argwhere(devices == device)[0][0]
            dev_emb = embeddings[dev_idx]
            checkpoint_a = partitions[idx, 0]
            checkpoint_b = partitions[idx, 1]
            end_flag = checkpoint_b > embeddings.shape[1]
            model_idx = np.argwhere(models2idx == model)[0][0]
            model_emb = deepcopy(dev_emb[:, model_idx])
            if dev_idx == 0:
                _embedding[dev_idx, :, model_idx] = np.pad(
                    model_emb[0:checkpoint_a-1], (0, embeddings.shape[1] - checkpoint_a + 1), 'constant')
            elif dev_idx == 1:
                if not end_flag:
                    _embedding[dev_idx, :, model_idx] = np.pad(
                        model_emb[checkpoint_a-1:checkpoint_b-1],
                        (checkpoint_a-1, embeddings.shape[1] - checkpoint_b + 1 if checkpoint_b <= embeddings.shape[1] else 0), 'constant')
                else:
                    _embedding[dev_idx, :, model_idx] = np.pad(model_emb[checkpoint_a-1:checkpoint_b-1], (checkpoint_a-1, 0), 'constant')
            elif dev_idx == 2:
                if not end_flag:
                    _embedding[dev_idx, :, model_idx] = np.pad(
                        model_emb[checkpoint_b-1:embeddings.shape[1]], (checkpoint_b-1, 0), 'constant')
    return _embedding
