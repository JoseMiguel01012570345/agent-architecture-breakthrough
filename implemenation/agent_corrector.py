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

                    adjusted_weight.lower = self.watch_overflow(adjusted_weight.lower)
                    adjusted_weight.upper = self.watch_overflow(adjusted_weight.upper)

                    adjustments[agent.index + connected_agent.index] = (
                        self.watch_overflow(
                            (adjusted_weight.lower + adjusted_weight.upper) / 2
                        )
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


# print(
#     (
#         18.574908018112183
#         + 19.393550872802734
#         + 20.560311555862427
#         + 20.005969285964966
#         + 20.206825971603394
#         + 63.99463486671448
#         + 27.006182432174683
#         + 22.296238660812378
#         + 22.087427139282227
#         + 24.89205551147461
#         + 21.74910259246826
#         + 21.747734785079956
#         + 21.758198261260986
#         + 21.867995023727417
#         + 21.54519033432007
#         + 25.197240591049194
#         + 24.225059986114502
#         + 21.772318840026855
#         + 22.538645029067993
#         + 21.838814735412598
#         + 22.642179489135742
#         + 21.794848442077637
#         + 22.629486083984375
#         + 40.84949064254761
#         + 69.05083465576172
#         + 70.77893543243408
#         + 68.64086985588074
#         + 69.55621004104614
#         + 68.76409888267517
#         + 46.827701568603516
#         + 66.16608023643494
#         + 66.17698836326599
#         + 68.97046852111816
#         + 70.6016161441803
#         + 73.11069393157959
#         + 73.31677579879761
#         + 73.3014566898346
#         + 72.18992447853088
#         + 72.12830233573914
#         + 70.47216629981995
#         + 71.26956605911255
#         + 70.17894434928894
#         + 74.99951434135437
#         + 73.28455877304077
#         + 69.69242215156555
#         + 70.38070154190063
#         + 70.53236293792725
#         + 70.24864625930786
#         + 70.47219967842102
#         + 70.62185764312744
#         + 69.42634892463684
#         + 69.85171675682068
#         + 67.40596795082092
#         + 74.41848516464233
#         + 68.95937967300415
#         + 67.82097220420837
#         + 66.82605290412903
#         + 7.82528614997864
#         + 67.24937963485718
#         + 67.70036149024963
#         + 67.13477540016174
#         + 66.26250004768372
#         + 67.70340061187744
#         + 67.86571621894836
#         + 67.71271967887878
#         + 2879.1645448207855
#         + 20.736681699752808
#         + 19.298786163330078
#     )
#     / 60
# )
