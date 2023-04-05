
import numpy as np

from lib.mcts.core.mcts import MonteCarloTreeSearch
from lib.mcts.common.tree import AbstractTreeSearchAgent


class Agent(AbstractTreeSearchAgent):
    """An agent that uses Monte Carlo Tree Search 
    to plan a sequence of action in a Markov
    Decision Process.
    """
    def make_planner(self):
        prior = Agent.policy()
        rollout = Agent.policy()
        return MonteCarloTreeSearch(self.env, prior, rollout, self.config)

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "budget": 100,
            "horizon": None,
            "env_preprocessors": []
         })
        return config

    @staticmethod
    def policy():
        return Agent.availability

    @staticmethod
    def availability(state, observation):
        """Choose actions from a uniform distribution 
        over currently available actions only.
        
        Parameters
        ----------
        state : np.ndarray
            The environment state
        observation : np.ndarray
            The corresponding observation

        Returns
        -------
        Tuple
            A tuple containing the actions and their probabilities
        """
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)
        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities
