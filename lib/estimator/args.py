
import json
import argparse

from copy import deepcopy


def simulator_options():
    """Example run:
        >>> python train.py --model-id resnet9 --dataset-id 'demo' --use-deterministic-algorithms --dataset-path '../../data/demo/' --name demo --auto-save --info --out-data-dir '../../data/demo/experiments' --use-tensorboard --use-wandb

    Returns
    -------
    argparse.ArgumentParser
        Parser for the Simulator library.
    """
    parser = argparse.ArgumentParser(
        description=f'Parser for the Simulator library.\n'
                    f'For information on the Simulator library, pass `--info`')

    parser.add_argument('--model-id', type=str,
                        help=f'The model name for training. The available '
                             f'models can be found at PyTorch Hub.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help=f'The model name for training. The available '
                             f'models can be found at PyTorch Hub.')
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Default: cuda)")
    parser.add_argument("--load-data", action="store_true",
                        help="Map data to GPU or RAM with respect to the utilized device")
    parser.add_argument("--batch-size", default=16, type=int,
                        help="images per gpu (default: 16)")
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    
    # TODO: Make these variables dynamically computed based on the input vectors
    parser.add_argument("--emb-dim", default=35, type=int,
                        help="number of elements in an embedding model vector (default: 35)")
    parser.add_argument("--sim-num-devices", default=3, type=int,
                        help="number of devices found in simulator platform (default: 3)")
    parser.add_argument("--sim-num-dnn", default=11, type=int,
                        help="number of DNN models mapped during simulation (default: 11)")
    
    parser.add_argument("--weight-decay", default=1e-4,
                        type=float, help="weight decay (default: 1e-4)")

    parser.add_argument("--resume", default="", type=str,
                        help="path of checkpoint")
    parser.add_argument("--use-deterministic-algorithms", action="store_true",
                        help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--clip-grad-norm", default=None, type=float,
                        help="the maximum gradient norm (default None)")

    parser.add_argument("--dataset-path", default=None,
                        type=str, help="the path to the dataset")
    parser.add_argument("--dataset-id", default=None,
                        type=str, help="the alias of the dataset")
    
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="use tensorboard to log the progress for the experiment.")
    parser.add_argument("--use-wandb", action="store_true",
                        help="use wandb to log the progress for the experiment.")
    parser.add_argument("--graph", action="store_true",
                        help="export model graph to writer (tensorboard or wandb).")                  
    
    parser.add_argument("--name", default=None, type=str,
                        help="alias for the running experiment")
    parser.add_argument("--out-data-dir", default=None,
                        type=str, help="path to output log info")

    parser.add_argument("--elite-metric", default="acc", type=str,
                        help=f"the metric used to determine the making "
                             f"of checkpoints (default: acc")
    parser.add_argument("--auto-save", action="store_true",
                        help="compile a checkpoint every epoch that yields better model results.")
    parser.add_argument("--chkpt-interval", default=5, type=int,
                        help="set a static interval for the checkpoint saving.")
    parser.add_argument("--info", action="store_true",
                        help="print abstract software info.")
    parser.add_argument("--seed", default=0, type=int,
                        help="seed used to reproduce an experiment")
    parser.add_argument("--verbose", action="store_true",
                        help="print the model architecture and other information.")
    # This is for the single neuron experiment
    parser.add_argument("--overwrite-num-classes", type=int, default=None,
                        help="This variable overwrites the number of neuron in the output layer of the Simulator-CNN (default: None).")

    return parser


def test_args():
    """Example run:
        >>> python test.py --use-deterministic-algorithms --demo

    Returns
    -------
    argparse.ArgumentParser
        Parser for the Simulator library.
    """
    parser = argparse.ArgumentParser(
        description=f'Parser for the Simulator library.\n'
                    f'These are the arguments for the testing of the simulator library.')
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Default: cuda)")
    parser.add_argument("--emb-dim", default=35, type=int,
                        help="number of elements in an embedding model vector (default: 35)")
    parser.add_argument("--sim-num-devices", default=3, type=int,
                        help="number of devices found in simulator platform (default: 3)")
    parser.add_argument("--sim-num-dnn", default=11, type=int,
                        help="number of DNN models mapped during simulation (default: 11)")
    parser.add_argument("--seed", default=0, type=int,
                        help="seed used to reproduce an experiment")
    parser.add_argument("--verbose", action="store_true",
                        help="print the model architecture and other information.")
    parser.add_argument("--demo", action="store_true",
                        help="launch a demo session.")
    parser.add_argument("--auto-set", action="store_true",
                        help="automatically reconfigure the arguments to match those in the training experiment.")
    parser.add_argument("--resume", default="../../data/demo/experiments/model.pth",
                        type=str, help="path of checkpoint")
    parser.add_argument("--use-deterministic-algorithms", action="store_true",
                        help="Forces the use of deterministic algorithms only.")
    # Options from the training parser
    parser.add_argument("--overwrite-num-classes", type=int, default=None,
                        help="This variable overwrites the number of neuron in the output layer of the Simulator-CNN (default: None).")
    # These options were added during the framework's final development stage to be used for demo purposes.
    parser.add_argument("--workload", type=int, nargs='+', default=(0, 5),
                        help="The neural network workload to be simulated (default: (0, 5)).")
    return parser


def store_args(filename, args):
    """Store the arguments in a file.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Parser for the Simulator library.
    filename : str
        Path to the file where the arguments are stored.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the arguments.
    """
    with open(filename, 'w+') as f:
        json.dump(args.__dict__, f, indent=2)
    
    return args.__dict__


def load_args(filename, args):
    """Load the arguments from a file.

    Parameters
    ----------
    args : argparse.ArgumentParser
        Parser for the Simulator library.
    filename : str
        Path to the file where the arguments are stored.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the Simulator library.
    """
    backup = deepcopy(args)
    with open(filename, 'r') as f:
        args.__dict__ = json.load(f)
    
    keys = backup.__dict__.keys()
    for key in keys:
        if key not in args.__dict__:
            args.__dict__[key] = backup.__dict__[key]

    return args
