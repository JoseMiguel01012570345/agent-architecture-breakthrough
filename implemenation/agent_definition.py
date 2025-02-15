# =============================================================================
# Agent Classes and Dummy Implementations
# =============================================================================
from override_arithmentic_opt import Interval

import sys

max_index = sys.maxsize
epsilon = 1e-9


class Agent:

    index = 0
    previous_output = Interval(0, 0)
    current_output = Interval(0, 0)
    function = None
    inverted_function = None
    isOutput = False
    input = None
    num_agents = 0

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
        self.init_input(X=X)

    def init_input(self, X):
        if self.isOutput:
            return

        result = Interval(0, 0)
        for x in X:
            result += x

        self.current_output = self.function(result)

    def FoG(self, P, agents):
        """
        Dummy function-of-G (FoG) calculation.
        """

        self.previous_output = self.current_output
        self.current_output = self.function(self.G(P=P, agents=agents))

        if self.current_output.lower == float("inf"):
            self.current_output.lower = max_index
        if self.current_output.upper == float("inf"):
            self.current_output.upper = max_index

        return self.current_output

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
            result = result + agent.current_output * Interval(
                float(P[self.index + index]), float(P[self.index + index])
            )

        return result

    def inverse_function(self, Y):
        """
        Dummy inverse function (F^{-1}). In a real implementation, this would invert the mapping F.
        """
        return self.inverted_function(Y)

    def reset_default(self, X):
        self.init_input(X=X)
