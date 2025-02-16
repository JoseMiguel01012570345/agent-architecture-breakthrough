from intervalar_functions import Interval
import os
import numpy as np

max_index = np.finfo(np.float32).max


# =============================================================================
# Corrector Agent
# =============================================================================
class Corrector:

    coordinator = None

    def __init__(self, coordinator):

        self.coordinator = coordinator
        pass

    def adjust_arcs(self, adjacency_matrix, agents, Y):
        """
        Given an error, the current adjacency matrix, and the agents,
        compute adjustments for the arcs (i.e. connection weights).
        """
        adjustments = [element for element in adjacency_matrix]
        # Loop over each output agent
        agent_output_index = -1
        # os.system("cls")
        for i, agent in enumerate(agents):
            # In the pseudocode the target is computed as F^{-1}(Y_i).
            # Here we simply use the agent’s inverse_function
            if not agent.isOutput:
                continue

            agent_output_index += 1
            target = agent.inverse_function(Y[agent_output_index])
            # Compute the current output via function G
            current = agent.G(P=agent.input["P"], agents=agent.input["X"])
            residual = target - current

            # Get a sorted list of input agents (dummy sort by output value)
            connected_agents = self.get_connected_agents(
                agents=agents,
                agent_index=agent.index,
            )  # sorted least to most

            cumulative_effect = current

            for index, connected_agent in enumerate(connected_agents):

                new_cumulative_effect = cumulative_effect + agent.current_output

                if self.watch_overflow(
                    (new_cumulative_effect.lower + new_cumulative_effect.upper) / 2
                ) >= self.watch_overflow((residual.lower + residual.upper) / 2):
                    if cumulative_effect == residual:
                        break

                    remaining = residual - cumulative_effect
                    # Adjust the weight of the current (last) agent
                    adjusted_weight = remaining / agent.current_output
                    r = adjusted_weight * connected_agent.current_output
                    adjustments[agent.index + connected_agent.index] = (
                        self.watch_overflow((r.lower + r.upper) / 2)
                    )

                    print(
                        {
                            "adjusted_weight.lower=": adjusted_weight.lower,
                            "adjusted_weight.upper=": adjusted_weight.upper,
                        }
                    )
                    break

                cumulative_effect += agent.current_output
                adjustments[agent.index + connected_agent.index] = np.float32(1)

        # Return a new adjacency matrix based on the adjustments
        return adjustments

    def get_connected_agents(self, agents, agent_index):
        connected_agents = []
        for agent in agents:
            if agent_index == agent.index:
                continue
            connected_agents.append(agent)

        return connected_agents

    def initialize_input_agents_with_X(self, agents, P):
        """
        Initialize each input agent with the input X.
        """

        for agent in agents:
            agent.FoG(P, agents)

    def agent_reset(self, agent, X):
        agent.init_input(X=X)

    def correction_phase(self, agents, stack_edges, Y, X):
        """
        Implements the correction phase.
        """
        arc_adjustments = []
        # First loop: iterate over stored states (stack_edges)
        outputs_list = []
        adj_matrix_list = []
        for adjacency_matrix, agents in stack_edges:
            outputs = [agent.current_output for agent in agents]

            outputs_list.append(outputs)
            adj_matrix_list.append(
                self.adjust_arcs(adjacency_matrix=adjacency_matrix, agents=agents, Y=Y)
            )

        i = 0
        # Second loop: iterate over stored states again
        while i < len(stack_edges):

            adjacency_matrix, agents = stack_edges[i]

            for agent in agents:
                self.agent_reset(agent=agent, X=X)

            adjacency_matrix = self.coordinator.generate_arcs(agents)
            self.initialize_input_agents_with_X(agents=agents, P=adjacency_matrix)

            outputs = [agent.current_output for agent in agents]

            outputs_list.append(outputs)
            adj_matrix_list.append(
                self.adjust_arcs(adjacency_matrix, agents=agents, Y=Y)
            )

            i += 1

        arc_adjustments.append(outputs_list)
        arc_adjustments.append(adj_matrix_list)
        return arc_adjustments

    def watch_overflow(self, t):
        if np.isposinf(t):
            t = max_index

        return t
