
from gymnasium.utils import seeding
from collections import defaultdict

from lib.mcts.common.config import Configurable


class AbstractPlanner(Configurable):
    def __init__(self, config=None):
        super().__init__(config)
        self.np_random = None
        self.root = None
        self.observations = []
        self.reset()
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(budget=500,
                    gamma=0.99,
                    step_strategy="reset")

    def seed(self, seed=None):
        """Seed the planner randomness 
        source, e.g. for rollout policy.

        Parameters
        ----------
        seed : int, optional
            The seed to be used, by default None

        Returns
        -------
        List[int]
            The used seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def plan(self, state, observation):
        """Plan an optimal sequence of actions.

        Parameters
        ----------
        state : np.ndarray
            The initial environment state.
        observation : np.ndarray
            The corresponding state observation.
        
        Returns
        -------
        List[int]
            The actions sequence.

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    def get_plan(self):
        """Get the optimal action sequence of the current 
        tree by recursively selecting the best action 
        within each node with no exploration.

        Returns
        -------
        List[int]
            The list of actions.
        """
        actions = []
        node = self.root
        while node.children:
            action = node.selection_rule()
            actions.append(action)
            node = node.children[action]
        return actions

    def step(self, state, action):
        observation, reward, done, info = state.step(action)
        self.observations.append(observation)
        return observation, reward, done, info

    def get_visits(self):
        visits = defaultdict(int)
        for observation in self.observations:
            visits[str(observation)] += 1
        return visits

    def get_updates(self):
        return defaultdict(int)

    def step_tree(self, actions):
        """Update the planner tree when the agent 
        performs an action.

        Parameters
        ----------
        actions : List[int]
            A sequence of actions to follow from the root node.
        """
        if self.config["step_strategy"] == "reset":
            self.step_by_reset()
        elif self.config["step_strategy"] == "subtree":
            if actions:
                self.step_by_subtree(actions[0])
            else:
                self.step_by_reset()
        else:
            self.step_by_reset()

    def step_by_reset(self):
        """Reset the planner tree to a root 
        node for the new state.
        """
        self.reset()

    def step_by_subtree(self, action):
        """Replace the planner tree by its subtree 
        corresponding to the chosen action.

        Parameters
        ----------
        action : int
            A chosen action from the root node.
        """

        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # The selected action was never explored, start a new tree.
            self.step_by_reset()

    def reset(self):
        raise NotImplementedError
