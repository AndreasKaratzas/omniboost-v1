
import numpy as np

from collections import defaultdict

from lib.mcts.utils.factory import safe_deepcopy_env


class Node(object):
    """A tree node.
    """

    def __init__(self, parent, planner):
        """New node.

        :param parent: its parent node
        :param planner: the planner using the node

        Parameters
        ----------
        parent : Node
            Its parent node.
        planner : AbstractPlanner
            The planner using the node.
        """

        self.parent = parent
        self.planner = planner

        # Dict of children nodes, indexed by action labels.
        self.children = {}

        # Number of times the node was visited.
        self.count = 0

    def get_value(self) -> float:
        """Evaluate the node return.

        Returns
        -------
        float
            An estimate of the node value.

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    def expand(self, branching_factor):
        """Expand the node and discover children.

        Parameters
        ----------
        branching_factor : int
            The number of the node's children.
        """
        for a in range(branching_factor):
            self.children[a] = type(self)(self, self.planner)

    def selection_rule(self):
        """A selection criterion.

        Raises
        ------
        NotImplementedError
            This function is abstract and must be defined separately 
            in each agent that inherits this class.
        """
        raise NotImplementedError()

    @staticmethod
    def breadth_first_search(root, operator=None, condition=None, condition_blocking=True):
        """Breadth-first search of all paths to nodes that meet a given condition.

        Parameters
        ----------
        root : Node
            Starting node.
        operator : bool, optional
            Will be applied to all traversed nodes, by default None
        condition : function, optional
            Nodes meeting that condition will be returned, by default None
        condition_blocking : bool, optional
            Do not explore a node which met the condition, by default True

        Yields
        ------
        List
            List of paths to nodes that met the condition.
        """
        queue = [(root, [])]
        while queue:
            (node, path) = queue.pop(0)
            if (condition is None) or condition(node):
                returned = operator(node, path) if operator else (node, path)
                yield returned
            if (condition is None) or not condition_blocking or not condition(node):
                for next_key, next_node in node.children.items():
                    queue.append((next_node, path + [next_key]))

    def is_leaf(self):
        return not self.children

    def path(self):
        """Computes the path of action labels from the root to the node.

        Returns
        -------
        List[Node]
            Sequence of action labels from the root to the node.
        """
        node = self
        path = []
        while node.parent:
            for a in node.parent.children:
                if node.parent.children[a] == node:
                    path.append(a)
                    break
            node = node.parent
        return reversed(path)

    def sequence(self):
        """Computes the path from the root to the node.

        Returns
        -------
        List[Node]
            A sequence of nodes from the root to the node.
        """
        node = self
        path = [node]
        while node.parent:
            path.append(node.parent)
            node = node.parent
        return reversed(path)

    @staticmethod
    def all_argmax(x):
        """Returns the non-zero elements of a np.ndarray like 
        structure which are the row-wise maximum values of that
        structure.

        Parameters
        ----------
        x : np.ndarray
            The numpy.array-like structure.

        Returns
        -------
        np.ndarray
            The list of indexes of all maximums of `x`.
        """
        m = np.amax(x)
        return np.nonzero(x == m)[0]

    def random_argmax(self, x):
        """Randomly tie-breaking `argmax`.
        
        Parameters
        ----------
        x : np.ndarray
            An array

        Returns
        -------
        int
            A random index among the maximums.
        """
        indices = Node.all_argmax(x)
        return self.planner.np_random.choice(indices)

    def __str__(self):
        return "{} (n:{}, v:{:.2f})".format(list(self.path()), self.count, self.get_value())

    def __repr__(self):
        return '<node {}>'.format(id(self))

    def get_trajectories(self, full_trajectories=True, include_leaves=True):
        """Get a list of visited nodes corresponding to the node subtree.

        Parameters
        ----------
        full_trajectories : bool, optional
            Return a list of observation sequences, else a list of observations, by default True
        include_leaves : bool, optional
            Include leaves or only expanded nodes, by default True

        Returns
        -------
        List
            The list of trajectories.
        """
        trajectories = []
        if self.children:
            for action, child in self.children.items():
                child_trajectories = child.get_trajectories(
                    full_trajectories, include_leaves)
                if full_trajectories:
                    trajectories.extend(
                        [[self] + trajectory for trajectory in child_trajectories])
                else:
                    trajectories.extend(child_trajectories)
            if not full_trajectories:
                trajectories.append(self)
        elif include_leaves:
            trajectories = [[self]] if full_trajectories else [self]
        return trajectories

    def get_obs_visits(self, state=None):
        """Get number of visits given an observation.

        Parameters
        ----------
        state : np.ndarray, optional
            The given observation, by default None

        Returns
        -------
        Tuple[int, int]
            The number of visits.
        """
        visits = defaultdict(int)
        updates = defaultdict(int)
        if hasattr(self, "observation"):
            for node in self.get_trajectories(full_trajectories=False,
                                              include_leaves=False):
                if hasattr(node, "observation"):
                    visits[str(node.observation)] += 1
                    if hasattr(node, "updates_count"):
                        updates[str(node.observation)] += node.updates_count
        else:
            # Replay required
            for node in self.get_trajectories(full_trajectories=False,
                                              include_leaves=False):
                replay_state = safe_deepcopy_env(state)
                for action in node.path():
                    observation, _, _, _ = replay_state.step(action)
                visits[str(observation)] += 1
        return visits, updates
