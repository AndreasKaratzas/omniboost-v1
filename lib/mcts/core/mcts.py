
import numpy as np

from lib.mcts.src.node import Node
from lib.mcts.common.plan import AbstractPlanner
from lib.mcts.src.olop import OpenLoopOptimisticPlanning
from lib.mcts.utils.factory import safe_deepcopy_env


class MonteCarloTreeSearch(AbstractPlanner):
    """An implementation of Monte-Carlo Tree 
    Search, with Upper Confidence Tree exploration.
    """
    def __init__(self, env, prior_policy, rollout_policy, config=None):
        """New MonteCarloTreeSearch instance.

        Parameters
        ----------
        env : gym.Env
            The MonteCarloTreeSearch configuration. Use default if None.
        prior_policy : function
            The prior policy used when expanding and selecting nodes.
        rollout_policy : function
            The rollout policy used to estimate the value of a leaf node.
        config : Dict, optional
            Configuration dictionary, by default None
        """
        super().__init__(config)
        self.env = env
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        if not self.config["horizon"]:
            self.config["episodes"], self.config["horizon"] = \
                OpenLoopOptimisticPlanning.allocation(
                    self.config["budget"], self.config["gamma"])

    @classmethod
    def default_config(cls):
        cfg = super(MonteCarloTreeSearch, cls).default_config()
        cfg.update({
            "temperature": 2 / (1 - cfg["gamma"]),
            "closed_loop": False
        })
        return cfg

    def reset(self):
        self.root = MonteCarloTreeSearchNode(parent=None, planner=self)

    def run(self, state, observation):
        """Run an iteration of Monte-Carlo Tree 
        Search, starting from a given state.

        Parameters
        ----------
        state : np.ndarray
            The initial environment state
        observation : np.ndarray
            The corresponding observation
        """
        node = self.root
        total_reward = 0
        depth = 0
        terminal = False
        state.seed(np.random.randint(2**30))
        while depth < self.config['horizon'] and node.children and not terminal:
            action = node.sampling_rule(temperature=self.config['temperature'])
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** depth * reward
            node_observation = observation if self.config["closed_loop"] else None
            node = node.get_child(action, observation=node_observation)
            depth += 1

        if not node.children \
                and depth < self.config['horizon'] \
                and (not terminal or node == self.root):
            node.expand(self.prior_policy(state, observation))

        if not terminal:
            total_reward = self.evaluate(state, observation, total_reward, depth=depth)
        node.update_branch(total_reward)

    def evaluate(self, state, observation, total_reward=0, depth=0):
        """Run the rollout policy to yield a 
        sample of the value of being in a 
        given state.

        Parameters
        ----------
        state : np.ndarray
            The leaf state.
        observation : np.ndarray
            The corresponding observation.
        total_reward : int, optional
            The initial total reward accumulated until now, by default 0
        depth : int, optional
            The initial simulation depth, by default 0

        Returns
        -------
        int
            The total reward of the rollout trajectory
        """
        for h in range(depth, self.config["horizon"]):
            actions, probabilities = self.rollout_policy(state, observation)
            action = self.np_random.choice(actions, 1, p=np.array(probabilities))[0]
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** h * reward
            if np.all(terminal):
                break
        return total_reward

    def plan(self, state, observation):
        for i in range(self.config['episodes']):
            self.run(safe_deepcopy_env(state), observation)
        return self.get_plan()

    def step_planner(self, action):
        if self.config["step_strategy"] == "prior":
            self.step_by_prior(action)
        else:
            super().step_planner(action)

    def step_by_prior(self, action):
        """Replace the Monte Carlo Tree Search tree 
        by its subtree corresponding to the chosen action, 
        but also convert the visit counts to prior 
        probabilities and before resetting them.
        
        Parameters
        ----------
        action : int
            A chosen action from the root node
        """
        self.step_by_subtree(action)
        self.root.convert_visits_to_prior_in_branch()


class MonteCarloTreeSearchNode(Node):
    K = 1.0
    """The value function first-order filter gain.
    """

    def __init__(self, parent, planner, prior=1):
        super(MonteCarloTreeSearchNode, self).__init__(parent, planner)
        self.value = 0
        self.prior = prior

    def selection_rule(self):
        if not self.children:
            return None
        # Tie best counts by best value
        actions = list(self.children.keys())
        counts = Node.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

    def sampling_rule(self, temperature=None):
        """Select an action from the node.
            * If exploration is wanted with some 
                temperature, follow the selection strategy.
            - Else, select the action with 
                maximum visit count.

        Parameters
        ----------
        temperature : float, optional
            The exploration parameter, positive or zero, by default None

        Returns
        -------
        int
            The selected action.
        """
        if self.children:
            actions = list(self.children.keys())
            # Randomly tie best candidates with respect to selection strategy
            indexes = [self.children[a].selection_strategy(temperature) for a in actions]
            return actions[self.random_argmax(indexes)]
        else:
            return None

    def expand(self, actions_distribution):
        """Expand a leaf node by creating a new 
        child for each available action.

        Parameters
        ----------
        actions_distribution : Tuple[int, float]
            The list of available actions and 
                their prior probabilities.
        """
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = type(self)(self, self.planner, probabilities[i])

    def update(self, total_reward):
        """Update the visit count and value of this 
        node, given a sample of total reward.

        Parameters
        ----------
        total_reward : float
            The total reward obtained through 
                a trajectory passing by this node.
        """
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)

    def update_branch(self, total_reward):
        """Update the whole branch from this node to the 
        root with the total reward of the corresponding 
        trajectory.

        Parameters
        ----------
        total_reward : float
            The total reward obtained through a 
                trajectory passing by this node.
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)

    def get_child(self, action, observation=None):
        child = self.children[action]
        if observation is not None:
            if str(observation) not in child.children:
                child.children[str(observation)] = MonteCarloTreeSearchNode(
                    parent=child, planner=self.planner, prior=0)
            child = child.children[str(observation)]
        return child

    def selection_strategy(self, temperature):
        """Select an action according to its 
        value, prior probability and visit count.

        Parameters
        ----------
        temperature : float
            The exploration parameter, 
                positive or zero.

        Returns
        -------
        int
            The selected action with maximum 
                value and exploration bonus.
        """
        if not self.parent:
            return self.get_value()

        return self.get_value() + temperature * len(self.parent.children) * self.prior/(self.count+1)

    def convert_visits_to_prior_in_branch(self, regularization=0.5):
        """For any node in the subtree, convert the 
        distribution of all children visit counts 
        to prior probabilities, and reset the visit 
        counts.

        Parameters
        ----------
        regularization : float in [0, 1], optional, By default 0.5
            Used to add some probability mass 
            to all children:
                * When 0, the prior is a Boltzmann distribution of visit counts
                * When 1, the prior is a uniform distribution
        """
        self.count = 0
        total_count = sum([(child.count+1) for child in self.children.values()])
        for child in self.children.values():
            child.prior = (1 - regularization)*(child.count+1) / \
                total_count + regularization/len(self.children)
            child.convert_visits_to_prior_in_branch()

    def get_value(self):
        return self.value
