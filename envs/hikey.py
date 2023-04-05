
import numpy as np
import seaborn as sns
import torch.nn as nn
import gymnasium as gym

from gymnasium import spaces
from typing import List, Tuple

from common.io import store_3d_np_array
from common.space import dev2dev_mapper, device_factory
from envs.utils import get_obs, simulate, get_swap_indices


class HikeyEnv(gym.Env):
    """Custom Environment that follows gym interface. 
    This environment is fine-tuned for the Hikey platform.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        simulator: nn.Module,
        ref_emb: np.ndarray,
        num_dev: int, 
        emb_dims: Tuple[int],
        dnn_ref: np.ndarray,
        dnn_names: np.ndarray,
        layers: np.ndarray,
        names: List[str] = [], 
        export_path: str = './hikey',
        overwrite_num_classes: int = None
    ):
        super(HikeyEnv, self).__init__()

        """Initialize Seaborn's theme.
        """
        sns.set_theme()
        
        """For our custom environment, the number
        of devices is the action space too.
        """
        self.n_actions = num_dev
        
        """This is in case the simulator CNN
        uses a single output neuron."""
        self.overwrite_num_classes = overwrite_num_classes

        """These are the experiment details in human readable form.
        Here, the framework stores the chosen DNN models to be mapped
        as well as the device order for the mapping."""
        self._task_details = {
            "Models": list(dnn_names)
        }

        """Assign names to the devices
        """
        self._names = names

        """Corresponding model indices between embedding tensor 
        and defined DNN model array."""
        self._mod2emb_mapper = get_swap_indices()
        
        """Export path used to save the rendered observations.
        """
        self.export_path = export_path

        if not self._names:
            self._names = ['Dev-' + str(i) for i in range(num_dev)]

        """This is a `gym.Env` mandatory variable.
        Here we essentially define our action space.
        """
        self.action_space = spaces.Discrete(self.n_actions)

        """This is a `gym.Env` mandatory variable.
        Here we essentially define our observation space.
        """
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(emb_dims), dtype=np.float32)
        
        """This is the core of the environment.
        The simulator module tries to approximate the 
        performance of the target platform to provide 
        accurate information for the MCTS stage."""
        self._simulator = simulator

        """This is the complete embedding vector.
        """
        self._ref_emb = ref_emb
        
        """This is a list with the number of partitions
        for each of the selected DNN models incremented 
        by 1."""
        self._n_layers = layers

        """These are the indices of the selected DNN models.
        """
        self._dnn_ref = dnn_ref

    def seed(self, seed=None):
        pass

    def reset(self, verbose: bool = False):
        """Initialize the observation registry.
        """
        self._registry = []

        """This is an epoch counter
        """
        self.epochs = 0

        """This is a rendering flag. It tells if the rendering function
        was called. If it was called then the `close` function shall
        delete any instance created."""
        self._is_rendering = False

        """Print the task details.
        """
        if verbose:
            print(self._task_details)
        
        """This is a pointer to the platform's workload.
        at current timestep `t`. This tensor is reconfigured 
        during the framework's searching an optimal solution."""
        self.cached_workload = np.zeros(self.observation_space.shape)
        
        """This is the configured device order.
        This list is gradually built up by the 
        MCTS agent."""
        self._dev_order = []
        
        """`dnn pointer`: there are 11 DNNs and they are 
        the columns of the workload tensor."""
        self.dnn_ptr = 0

        """`emb pointer`: there are 35 elements in each 
        embedding vector and they are the rows of the 
        workload tensor."""
        self.emb_ptr = self._n_layers[self.dnn_ptr]

        """Reset the terminal flag, along with
        the winning and the losing indicators.
        """
        self._terminal = False
        self._win = False
        self._loss = False

        """Reset the action trajectory.
        """
        self._trajectory = []

        """Reset the pipeline device order registry.
        """
        self._pipeline_registry = []

        """Reset the partitions matrix.
        """
        self._partitions = np.zeros((
            np.prod(self._dnn_ref.shape), self._ref_emb.shape[0] - 1), dtype=np.int)
        
        self._partitions[:, 0] = self._n_layers - 1
        self._partitions[:, 1] = self._n_layers

        """Reset the device order registry.
        """
        self._dev_order = np.zeros((
            np.prod(self._dnn_ref.shape), self._ref_emb.shape[0]), dtype=np.int)
        
        """Reset the partition pointer.
        """
        self._part_ptr = 0

        """Reset the device order alias registry.
        """
        self._dev_order_alias = np.asarray([device_factory()] * np.prod(
            self._dnn_ref.shape)).reshape(np.prod(self._dnn_ref.shape), self._ref_emb.shape[0])

        """Initialize a dummy device set. This will help to 
        track whether the defined pipeline utilizes a device
        twice. If it does, the episode will terminate.
        """
        self._dev_set = []

        return self.cached_workload
    
    def _register_state(self):
        self._registry.append(list(self.cached_workload))
        self.epochs += 1
    
    def _take_action(self, action: int):
        if action not in range(self.n_actions):
            raise ValueError(
                "Invalid action={} which is out of the defined action space".format(action))
        
        if self._terminal:
            raise RuntimeError("The environment is in terminal state")

        if self.emb_ptr == 1 and self.dnn_ptr < np.prod(self._dnn_ref.shape) - 1:
            self.dnn_ptr += 1
            self._part_ptr = 0
            self._trajectory = []
            self._pipeline_registry = []
            self.emb_ptr = self._n_layers[self.dnn_ptr] - 1
        elif self.emb_ptr > 1:
            self.emb_ptr -= 1
        else:
            self._win = True
        
        self._trajectory.append(action)
        
        if len(self._trajectory) > 1:

            if action != self._trajectory[-2]:
                self._pipeline_registry.append(action)
                _, _pipe_reg_unique_cntr = np.unique(
                    np.asarray(self._pipeline_registry), return_counts=True)

                if np.any(_pipe_reg_unique_cntr > 1):
                    self._loss = True
        else:
            self._pipeline_registry.append(action)

        if len(self._trajectory) > 1:
            if action != self._trajectory[-2]:
                # NOTE: This part of code is customized for the Hikey platform.
                if self._part_ptr == 0:
                    self._partitions[self.dnn_ptr, self._part_ptr] = self.emb_ptr
                    self._part_ptr += 1
                else:
                    _swap_part = self._partitions[self.dnn_ptr, self._part_ptr - 1]
                    self._partitions[self.dnn_ptr, self._part_ptr] = _swap_part
                    self._partitions[self.dnn_ptr, self._part_ptr - 1] = self.emb_ptr
                    
        if self.emb_ptr == 1 and self.dnn_ptr >= np.prod(self._dnn_ref.shape) - 1:
            self._win = True

        dev_order_idx = np.argsort(np.unique(self._trajectory, return_index=True)[1])
        dev_order_idx_flipped = np.flip(dev_order_idx)
        util_devs = np.unique(self._trajectory)
        self._dev_order[self.dnn_ptr] = np.hstack((util_devs[dev_order_idx_flipped], np.setdiff1d(np.arange(
            self._ref_emb.shape[0]), util_devs[dev_order_idx_flipped], assume_unique=False)))

        names_ordered = np.array(self._names)[self._dev_order[self.dnn_ptr]]
        inv_dev_mapper = {v: k for k, v in dev2dev_mapper().items()}
        dev_order_tpl = sorted(inv_dev_mapper.items(), key=lambda pair: list(names_ordered).index(pair[0]))
        dev_order_alias = [id for dev, id in dev_order_tpl]
        self._dev_order_alias[self.dnn_ptr] = np.array(dev_order_alias)
        
        if not self._loss:
            _obs, _devs, _ = get_obs(
                mappings=np.expand_dims(self._dev_order[self.dnn_ptr], axis=0), 
                partitions=np.expand_dims(self._partitions[self.dnn_ptr, :] + 1, axis=0), 
                workload=np.expand_dims(np.asarray(self._dnn_ref[self.dnn_ptr]), axis=0),
                embeddings=self._ref_emb,
                devices=np.asarray(self._names)
            )
            _dev_indices_ordered = get_swap_indices(
                _devs.flatten(), self._names)
            _obs = _obs[_dev_indices_ordered.flatten()]
        else:
            _obs = np.ones((self._ref_emb.shape)) * (-10)
        
        _obs[:, :, [self._dnn_ref[self.dnn_ptr], np.argwhere(self._mod2emb_mapper == self._dnn_ref[self.dnn_ptr])[0][0]]] = _obs[:, :, [
            np.argwhere(self._mod2emb_mapper == self._dnn_ref[self.dnn_ptr])[0][0], self._dnn_ref[self.dnn_ptr]]]

        self.cached_workload[:, :, self._dnn_ref[self.dnn_ptr]] = _obs[:, :, self._dnn_ref[self.dnn_ptr]]

    def _get_reward(self):
        if self._loss:
            return -5000
        else:
            _perf = simulate(model=self._simulator,
                            sample=self.cached_workload,
                            std_scaler_path='../../data/demo/StandardScaler.pkl',
                            norm_scaler_path='../../data/demo/MinMaxScaler.pkl',
                            overwrite_num_classes=self.overwrite_num_classes,)
            return np.sum(_perf) * (self.dnn_ptr + 1)
    
    def _is_terminal(self):
        self._terminal = self._win or self._loss
        return self._terminal
    
    def _get_info(self):
        return {
            "epochs": self.epochs,
            "terminal": self._terminal,
            "trajectory": self._trajectory,
            "partitions": np.asarray(self._partitions).flatten(),
            "devices": np.asarray(self._dev_order_alias).flatten(),
            "models": self._task_details["Models"]
        }
    
    def hard_copy(self):
        store_3d_np_array(self.cached_workload, "./cached_workload-" + str(self.epochs) + ".xlsx")

    def step(self, action):
        self._take_action(action=action)
        self._register_state()

        done = self._is_terminal()
        reward = self._get_reward()
        info = self._get_info()

        return self.cached_workload, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError
        
    def close(self, verbose: bool = False):
        if self._is_rendering:
            self._hikey_app.quit()

        if verbose:
            print(f"Closing the `HikeyEnv` environment.")
        
        return
