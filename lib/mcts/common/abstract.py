
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict

from lib.mcts.common.config import Configurable


class AbstractAgent(Configurable, ABC):
    """An abstract class specifying the interface of a generic agent.
    """

    def __init__(self, config: Dict = None):
        super(AbstractAgent, self).__init__(config)
        self.writer = None  # TensorBoard writer
        self.directory = None  # Run directory

    @abstractmethod
    def record(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, info: Dict):
        """Record a transition of the environment to update the agent.

        Parameters
        ----------
        state : np.ndarray
            The current state of the agent.
        action : int
            The action performed.
        reward : float
            The reward collected.
        next_state : np.ndarray
            The new state of the agent after the action was performed.
        done : bool
            Whether the next state is terminal.
        info : Dict
            Dictionary with additional environment information.

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, state: np.ndarray):
        """Pick an action.

        Parameters
        ----------
        state : np.ndarray
            The current state of the agent.
        
        Returns
        -------
        int
            The action to perform.

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    def plan(self, state: np.ndarray):
        """Plan an optimal trajectory from an initial state.

        Parameters
        ----------
        state : np.ndarray
            The initial state of the agent.

        Returns
        -------
        List[int]
            A sequence of actions to perform.
        """
        return [self.act(state)]

    @abstractmethod
    def reset(self):
        """Reset the agent to its initial internal state.

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    @abstractmethod
    def seed(self, seed: int = None):
        """Seed the agent's random number generator.

        Parameters
        ----------
        seed : int, optional
            The seed to be used to generate random numbers, by default None

        Returns
        -------
        int
            The used seed.

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, filename: str):
        """Save the model parameters to a file.
        
        Parameters
        ----------
        filename : str
            The path of the file to save the model parameters in.

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, filename):
        """Load the model parameters from a file.

        Parameters
        ----------
        filename : str
            The path of the file to load the model parameters from

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    def eval(self):
        """Set to testing mode. Disable any unnecessary exploration.
        """
        pass

    def set_writer(self, writer):
        """Set a TensorBoard writer to log the agent internal variables.

        Parameters
        ----------
        writer : torch.utils.tensorboard.SummaryWriter
            A summary writer.
        """
        self.writer = writer

    def set_directory(self, directory):
        """Set export directory of agent.

        Parameters
        ----------
        directory : str
            The export directory.
        """
        self.directory = directory

    def set_time(self, time):
        """Set a local time, to control the agent 
        internal schedules (e.g. exploration).

        Parameters
        ----------
        time : int
            An integer unique to the 
            internal agent mechanism.
        """
        self.time = time
