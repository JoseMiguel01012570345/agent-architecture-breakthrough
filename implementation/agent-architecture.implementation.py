from agent_definition import Agent

# =============================================================================
# Global constants
# =============================================================================
ACTIVATION_THRESHOLD = 0.5  # Example threshold for activation


def no_arcs_exist(adjacency_matrix):
    """
    Return True if the given adjacency_matrix indicates that no arcs exist.
    """
    if adjacency_matrix is None:
        return True
    if isinstance(adjacency_matrix, list) and len(adjacency_matrix) == 0:
        return True
    return False


def calculate_max_difference(target_outputs, current_outputs):
    """
    Calculate the maximum absolute difference between target and current outputs.
    """
    differences = [abs(t - c) for t, c in zip(target_outputs, current_outputs)]
    return max(differences) if differences else 0


def calculate_error(outputs, Y):
    """
    Compute error as the absolute difference between each output and its target Y.
    """
    return [abs(o - y) for o, y in zip(outputs, Y)]


def get_connected_agents(agent: Agent, adjacency_matrix):
    """
    Dummy function: return a list of agents connected to the given agent.
    In a full implementation, this would examine the adjacency_matrix.
    """
    # For simplicity, we return an empty list.
    return adjacency_matrix[agent.index]


def calculate_preconditions(agent, adjacency_matrix):
    """
    For each agent connected to 'agent', compute a precondition value.
    """
    preconditions = []
    connected_agents = get_connected_agents(agent, adjacency_matrix)
    for connected_agent in connected_agents:
        prev_val = connected_agent.previous_output
        curr_val = connected_agent.current_output
        max_val = max(abs(prev_val), abs(curr_val)) + 1e-6
        precondition = abs(prev_val - curr_val) / max_val
        preconditions.append(precondition)
    return preconditions


def check_activation_conditions(preconditions):
    """
    Return True if all preconditions meet or exceed the activation threshold.
    """
    return all(p >= ACTIVATION_THRESHOLD for p in preconditions)


def collect_outputs(output_agents):
    """
    Collect outputs from the output agents.
    """
    return [agent.output for agent in output_agents]


def initialize_input_agents():
    """
    Create and return a list of input agents.
    """
    # For example, create 3 input agents.
    return [Agent() for _ in range(3)]


def initialize_output_agents():
    """
    Create and return a list of output agents.
    """
    # For example, create 2 output agents.
    return [Agent() for _ in range(2)]


def initialize_coordinator_agent():
    """
    Create and return a Coordinator agent with a dummy model.
    """
    model = DummyModel()
    return Coordinator(model)


def initialize_corrector_agent():
    """
    Create and return a Corrector agent.
    """
    return Corrector()


def initialize_input_agents_with_X(input_agents, X):
    """
    Initialize each input agent with the input X.
    """
    for agent in input_agents:
        agent.inputs = X


def correction_phase(input_agents, output_agents, corrector, stack_edges, Y):
    """
    Implements the correction phase.
    """
    arc_adjustments = []
    # First loop: iterate over stored states (stack_edges)
    for adjacency_matrix, input_agents_saved, output_agents_saved in stack_edges:
        outputs = [agent.output for agent in output_agents_saved]
        error = calculate_error(outputs, Y)
        arc_adjustments.append(
            corrector.adjust_arcs(error, adjacency_matrix, input_agents, output_agents)
        )

    # Reset agents’ defaults
    for agent in input_agents:
        agent.reset_default_outputs()
    for agent in output_agents:
        agent.reset_default_inputs()

    start = 0
    # Second loop: iterate over stored states again
    while start < len(stack_edges):
        adjacency_matrix, input_agents_saved, output_agents_saved = stack_edges[start]
        outputs = [agent.output for agent in output_agents_saved]
        error = calculate_error(outputs, Y)
        arc_adjustments.append(
            corrector.adjust_arcs(error, adjacency_matrix, input_agents, output_agents)
        )
        # Update input agents with current output from output agents
        for i, agent in enumerate(input_agents):
            if i < len(output_agents):
                agent.inputs = output_agents[i].output
        start += 1
    return arc_adjustments


# =============================================================================
# Main Process
# =============================================================================
def main_process(dataset, max_iterations):
    """
    Main process that iterates over the dataset.

    Args:
        dataset: a list of (X, Y) pairs.
        max_iterations: maximum number of iterations per sample.

    Returns:
        results: collected outputs from the output agents.
    """
    # Initialize agents
    input_agents = initialize_input_agents()
    output_agents = initialize_output_agents()
    coordinator = initialize_coordinator_agent()
    corrector = initialize_corrector_agent()
    convergence_threshold = 0.01

    results = []
    # For each (X, Y) pair in the dataset:
    for X, Y in dataset:
        initialize_input_agents_with_X(input_agents, X)
        iteration = 1
        converged = False
        stack_edges = []  # to store states (arc matrices and agents' states)

        while iteration <= max_iterations and not converged:
            # Step 1: Coordinator determines arcs
            adjacency_matrix = coordinator.generate_arcs(X)
            # Save current state (using shallow copies for demonstration)
            stack_edges.append(
                (adjacency_matrix, list(input_agents), list(output_agents))
            )

            # Step 2: Check termination condition
            if no_arcs_exist(adjacency_matrix):
                converged = True
                break

            # Step 3: Agent computation phase
            all_agents = input_agents + output_agents
            for agent in all_agents:
                agent.preconditions = calculate_preconditions(agent, adjacency_matrix)
                if check_activation_conditions(agent.preconditions):
                    agent.output = agent.FoG(agent.preconditions, agent.inputs)

            # Step 4: Check convergence
            # Here we assume the target outputs are provided as Y (a list)
            current_outputs = [agent.output for agent in output_agents]
            delta = calculate_max_difference(Y, current_outputs)
            if delta < convergence_threshold:
                converged = True
                break

            # Update each agent's inputs to its current output
            for agent in all_agents:
                agent.inputs = agent.output

            iteration += 1

        # Final output collection
        current_results = collect_outputs(output_agents)
        results.append(current_results)

        # If not converged, run the correction phase and update the coordinator
        if not converged:
            arc_adjustments = correction_phase(
                input_agents, output_agents, corrector, stack_edges, Y
            )
            coordinator.update(arc_adjustments)

    return results


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Create a dummy dataset of (X, Y) pairs.
    # Here, X might be a number (or a vector) and Y is a list of target outputs.
    dataset = [(10, [5, 7]), (20, [15, 17])]
    max_iterations = 5

    final_results = main_process(dataset, max_iterations)
    print("Final results:", final_results)
