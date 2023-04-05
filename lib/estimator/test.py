
import sys

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

import os
import torch
import warnings
import numpy as np

from pathlib import Path

from common.terminal import colorstr
from lib.estimator.inference import Tester
from lib.estimator.nvidia import cuda_check
from lib.estimator.deterministic import set_seed
from lib.estimator.args import test_args, load_args


def driver(args):
    warnings.filterwarnings("ignore")

    if args.auto_set:
        checkpoint_path = Path(args.resume)
        configs_filepath = checkpoint_path.parent.parent.absolute()
        load_args(filename=os.path.join(configs_filepath, 'config.json'), args=args)
        args.resume = checkpoint_path
    
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"Device utilized: {colorstr(options=['red', 'underline'], string_args=list([device]))}.\n")
    
    if device == torch.device('cuda'):
        args.n_devices, cuda_arch = cuda_check()
        print(
            f"Found NVIDIA GPU of "
            f"{colorstr(options=['cyan'], string_args=list([cuda_arch]))} "
            f"Architecture.")
    
    if args.use_deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
        if args.seed is None:
            args.seed = 0

        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    model = Tester(num_devices=args.sim_num_devices, 
                   device=args.device, load_checkpoint=args.resume, 
                   verbose=args.verbose)

    return model
    

def simulate(model):
    # set hyper-parameters
    root_dir = '../../data/demo/test'

    # set input data
    mapping = list(sorted(os.listdir(os.path.join(root_dir, "mapping"))))
    X = [np.load(os.path.join(root_dir, "mapping", f)) for f in mapping]
    
    # inference
    for sample in X:
        output = model.test(sample)
        print(output)


if __name__ == "__main__":
    args = test_args().parse_args()
    model = driver(args=args)
    if args.demo:
        simulate(model)
