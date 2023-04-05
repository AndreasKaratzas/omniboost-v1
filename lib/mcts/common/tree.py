
import gymnasium as gym

from typing import Tuple, Dict

from lib.mcts.common.abstract import AbstractAgent
from lib.mcts.utils.factory import preprocess_env


class AbstractTreeSearchAgent(AbstractAgent):
    PLANNER_TYPE: Tuple[gym.Env, Dict] = (None, None)
    NODE_TYPE = None

    def __init__(self,
                 env,
                 config=None):
        """A new Tree Search agent.

        Parameters
        ----------
        env : gym.Env
            The environment
        config : Dict, optional
            The agent configuration, by default None
        """
        super(AbstractTreeSearchAgent, self).__init__(config)
        self.env = env
        self.planner = self.make_planner()
        self.previous_actions = []
        self.remaining_horizon = 0
        self.steps = 0

    @classmethod
    def default_config(cls):
        return {
            "env_preprocessors": [],
            "display_tree": False,
            "receding_horizon": 1,
            "terminal_reward": 0
        }

    def make_planner(self):
        if self.PLANNER_TYPE:
            return self.PLANNER_TYPE(self.env, self.config)
        else:
            raise NotImplementedError()

    def plan(self, observation):
        """Plan an optimal sequence of actions.
        Start by updating the previously found 
        tree with the last action performed.

        Parameters
        ----------
        observation : np.ndarray
            The current state

        Returns
        -------
        List[int]
            The list of actions.
        """
        self.steps += 1
        replanning_required = self.step(self.previous_actions)
        if replanning_required:
            env = preprocess_env(self.env, self.config["env_preprocessors"])
            actions = self.planner.plan(state=env, observation=observation)
        else:
            actions = self.previous_actions[1:]
        
        self.previous_actions = actions
        return actions

    def step(self, actions):
        """Handle receding horizon mechanism.

        Parameters
        ----------
        actions : List[int]
            The list of actions.

        Returns
        -------
        bool
            Whether a replanning is required.
        """
        replanning_required = self.remaining_horizon == 0 or len(actions) <= 1
        if replanning_required:
            self.remaining_horizon = self.config["receding_horizon"] - 1
        else:
            self.remaining_horizon -= 1

        self.planner.step_tree(actions)
        return replanning_required

    def reset(self):
        self.planner.step_by_reset()
        self.remaining_horizon = 0
        self.steps = 0

    def seed(self, seed=None):
        return self.planner.seed(seed)

    def record(self, state, action, reward, next_state, done, info):
        pass

    def act(self, state):
        return self.plan(state)[0]

    def save(self, filename):
        return False

    def load(self, filename):
        return False
