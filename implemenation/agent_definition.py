from override_arithmentic_opt import Interval
import numpy as np

# =============================================================================
# Agent Classes and Dummy Implementations
# =============================================================================

max_index = np.finfo(np.float32).max
epsilon = 1e-9


class Agent:

    index = 0
    previous_output = Interval(0, 0)
    _current_output_ = Interval(0, 0)
    function = None
    inverted_function = None
    isOutput = False
    input = None
    num_agents = 0

    @property
    def current_output(self):
        return Interval(
            lower=self._current_output_.lower, upper=self._current_output_.upper
        )

    @current_output.setter
    def current_output(self, new_current):
        self._current_output_ = self.watch_overflow(result=new_current)

    def __init__(
        self,
        X,
        index,
        function,
        inverted_function,
        isOutput=True,
        num_agents=0,
    ):
        self.isOutput = isOutput
        self.index = index
        self.function = function
        self.inverted_function = inverted_function
        self.num_agents = num_agents

    def init_input(self, X):
        if self.isOutput:
            return

        self.current_output = X[self.index]

    def FoG(self, P, agents):
        """
        Dummy function-of-G (FoG) calculation.
        """

        self.previous_output = self.current_output
        r = self.watch_overflow(result=self.function(self.G(P=P, agents=agents)))

        if np.isnan(r.lower):
            r.lower = max_index

        if np.isnan(r.upper):
            r.upper = max_index

        self.current_output = r

    def get_connected_agents(self, adjacency_matrix):
        """
        Dummy function: return a list of agents connected to the given agent.
        """

        i = self.index + 1
        connected_agents = []
        while i < self.num_agents:
            connected_agents.append(adjacency_matrix[i])
            i += 1

        return connected_agents

    def calculate_preconditions(self, agent, adjacency_matrix):
        """
        For each agent connected to 'agent', compute a precondition value.
        """
        preconditions = []
        connected_agents = self.get_connected_agents(adjacency_matrix)
        for connected_agent in connected_agents:
            prev_val = connected_agent.previous_output
            curr_val = connected_agent.current_output
            max_val = max(abs(prev_val), abs(curr_val))
            precondition = abs(prev_val - curr_val) / max_val
            preconditions.append(precondition)
        return preconditions

    def G(self, P, agents):
        """
        Dummy function G. Here we simply return the inputs.
        """
        self.input = {"P": P, "X": agents}

        result = Interval(0, 0)

        for index, agent in enumerate(agents):
            if self.index == index:
                continue
            if np.isnan(agent.current_output.lower):
                agent.current_output.lower = max_index

            if np.isnan(agent.current_output.upper):
                agent.current_output.upper = max_index

            result += agent.current_output * Interval(
                np.float32(P[self.index + index]), np.float32(P[self.index + index])
            )
            result.lower = np.float32(result.lower)  # convert to float32
            result.upper = np.float32(result.upper)  # convert to float32

            if np.isnan(result.lower) or np.isposinf(
                result.lower
            ):  # truncate lower bound value
                result.lower = max_index

            if np.isneginf(result.lower):
                result.lower = -max_index

            if np.isnan(result.upper) or np.isposinf(
                result.upper
            ):  # truncate value of upper bound
                result.upper = max_index

            if np.isneginf(result.upper):
                result.upper = max_index

        return self.watch_overflow(result=result)

    def inverse_function(self, Y):
        """
        Dummy inverse function (F^{-1}). In a real implementation, this would invert the mapping F.
        """
        return self.inverted_function(Y)

    def watch_overflow(self, result):

        result.lower = np.float32(result.lower)
        result.upper = np.float32(result.upper)

        r = np.clip(np.array([result.lower, result.upper]), -max_index, max_index)
        result.lower = r.min()
        result.upper = r.max()

        return result
