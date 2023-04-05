
import sys
sys.path.append('../../')

import time
import numpy as np

from pathlib import Path

from envs.hikey import HikeyEnv
from lib.mcts.core.agent import Agent
from lib.estimator.release import init_model
from lib.estimator.utils import get_embeddings
from common.space import layer_factory, dnn_factory


if __name__ == "__main__":
    """ Run:
    >>> python main.py --use-deterministic-algorithms --demo --emb-dim 35 --sim-num-devices 3 --sim-num-dnn 11 --auto-set --resume '../../data/demo/experiments/model.pth' --seed 33
    """

    model, args = init_model()
    embeddings = get_embeddings("../../data/demo")
    
    export_path = Path(args.resume).parent.parent.absolute() / Path('data')
    
    Path(export_path).mkdir(parents=True, exist_ok=True)

    env = HikeyEnv(
        simulator=model,
        ref_emb=embeddings,
        num_dev=args.sim_num_devices,
        emb_dims=embeddings.shape,
        dnn_ref=np.asarray(args.workload),
        dnn_names=np.asarray(dnn_factory())[np.asarray(args.workload)],
        layers=np.asarray(layer_factory())[np.asarray(args.workload)],
        names=['L', 'G', 'B'],
        export_path=export_path,
        overwrite_num_classes=args.overwrite_num_classes
    )

    # TODO: Experiment on the hyperparameters of the MCTS agent
    agent = Agent(
        env, 
        config=dict(budget=500, temperature=2, max_depth=300)
    )

    # ui = Render(names=env._names, dnns=dnn_factory(),
    #             emb_dim=embeddings.shape[1])

    steps = 0
    done = False
    state = env.reset(verbose=True)
    start = time.time()
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {steps} [Models: {info['models']}\tDevices: "
              f"{info['devices']}\tPartitions: {info['partitions']}"
              f"\tReward: {reward}]")
        
        # pprint(info)
        # ui.render(
        #     cached_workload=next_state, 
        #     export_path=env.export_path, 
        #     epochs=env.epochs, 
        #     verbose=1
        # )

        steps += 1
        state = next_state
    end = time.time()    
    # ui.close()
    env.close()

    print(f"Time: {end - start} seconds")
