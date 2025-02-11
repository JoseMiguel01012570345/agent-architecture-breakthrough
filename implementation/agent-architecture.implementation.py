from agent_definition import Agent
import intervalar_functions
import coordinator as coord
import agent_corrector as aget_correct
import compartimental_models
import model
from scipy.interpolate import RBFInterpolator

# =============================================================================
# Global constants
# =============================================================================


def no_arcs_exist(adjacency_matrix, umbral=0.02):
    """
    Return True if the given adjacency_matrix indicates that no arcs exist.
    """
    if adjacency_matrix is None:
        return True
    if any([element > 0 for element in adjacency_matrix]):
        return True
    return False


def calculate_max_difference(target_outputs, current_outputs):
    """
    Calculate the maximum absolute difference between target and current outputs.
    """
    differences = [
        abs(t - c) / len(current_outputs)
        for t, c in zip(target_outputs, current_outputs)
    ]
    return sum(differences) if differences else 0


def initialize_agents(number_agents, number_input_agents):
    """
    Create and return a list of input agents.
    """
    import random

    list_of_input_agents = []
    for i in range(number_input_agents):  # determine output agents
        rand_index = random.randint(0, number_agents)
        while rand_index in list_of_input_agents:
            rand_index = random.randint(0, number_agents)

        list_of_input_agents.append(rand_index)

    i = 0
    intervalar_Functions = []
    intervalar_inverted_functions = []
    while i < number_agents:

        intervalar_Functions.append(
            intervalar_functions.intervalar_functions_avaliable[
                i % len(intervalar_functions.intervalar_functions_avaliable)
            ][0]
        )

        intervalar_inverted_functions.append(
            intervalar_functions.intervalar_functions_avaliable[
                i % len(intervalar_functions.intervalar_functions_avaliable)
            ][1]
        )
        i += 1

    agents = []
    for index, function in enumerate(intervalar_Functions):

        agent = None
        if index in list_of_input_agents:
            agent = Agent(
                index=index,
                function=function,
                inverted_function=intervalar_inverted_functions[index],
                isOutput=True,
            )
        else:
            agent = Agent(
                index=index,
                function=function,
                inverted_function=intervalar_inverted_functions[index],
            )

        agents.append(agent)

    return agents


def initialize_coordinator_agent(model, X_init, Y_init):
    """
    Create and return a Coordinator agent with a dummy model.
    """

    coordinator = coord.Coordinator(model=model)
    coordinator.update([X_init, Y_init])
    return coordinator


def initialize_corrector_agent(coordinator):
    """
    Create and return a Corrector agent.
    """
    return aget_correct.Corrector(coordinator=coordinator)


def initialize_input_agents_with_X(agents, X, P):
    """
    Initialize each input agent with the input X.
    """

    for agent in agents:
        if not agent.isOutput:
            agent.FoG(X, P)


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
    n_in = len(dataset[0]["X"])
    n_out = len(dataset[0]["Y"])
    agents = initialize_agents(
        number_agents=n_in + n_out,
        number_input_agents=n_in,
    )

    rbf_interpolator_model = model.model(Ninput=n_in, Noutput=n_in**2)

    Y_init = [
        [dataset[0]["X"] for x in dataset[0]["X"]],
        [dataset[1]["X"] for x in dataset[1]["X"]],
    ]

    coordinator = initialize_coordinator_agent(
        model=rbf_interpolator_model,
        X_init=[dataset[0]["X"], dataset[1]["X"]],
        Y_init=Y_init,
    )
    corrector = initialize_corrector_agent(coordinator=coordinator)
    convergence_threshold = 0.01

    results = []
    # For each (X, Y) pair in the dataset:
    for data in dataset:

        X = data["X"]
        Y = data["Y"]

        iteration = 1
        converged = False
        stack_edges = []

        # Reset agents’ defaults
        for agent in agents:
            agent.reset_default_outputs()

        while iteration <= max_iterations and not converged:

            # Step 1: Coordinator determines arcs
            adjacency_matrix = coordinator.generate_arcs(X)

            initialize_input_agents_with_X(agents=agents, X=X, P=adjacency_matrix)

            # Save current state (using shallow copies for demonstration)
            stack_edges.append((adjacency_matrix, agents))

            # Step 2: Check termination condition
            if no_arcs_exist(adjacency_matrix):
                converged = True
                break

            # Step 3: Check convergence
            # Here we assume the target outputs are provided as Y (a list)
            current_outputs = [
                agent.current_output for agent in agents if agent.isOutput
            ]
            delta = calculate_max_difference(Y, current_outputs)
            if delta < convergence_threshold:
                converged = True
                break

            iteration += 1

        # Final output collection

        results.append(current_outputs)

        # If not converged, run the correction phase and update the coordinator
        if not converged:
            arc_adjustments = corrector(agents, stack_edges, Y)
            coordinator.update(arc_adjustments)

    return results


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Create a dummy dataset of (X, Y) pairs.
    # Here, X might be a number (or a vector) and Y is a list of target outputs.
    dataset = compartimental_models.dataset_generator()
    max_iterations = 10000

    final_results = main_process(dataset, max_iterations)
    print("Final results:", final_results)
