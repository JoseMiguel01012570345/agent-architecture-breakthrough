from agent_definition import Agent
import intervalar_functions
import coordinator as coord
import agent_corrector as aget_correct
import compartimental_models
import model
from intervalar_functions import Interval
import os
import numpy as np
from copy import deepcopy

os.system("cls")

# =============================================================================
# Global constants
# =============================================================================
max_index = np.finfo(np.float32).max
epsilon = 1e-9


def no_arcs_exist(adjacency_matrix, umbral=0.02):
    """
    Return True if the given adjacency_matrix indicates that no arcs exist.
    """
    if adjacency_matrix is None:
        return True
    if not any([element > epsilon for element in adjacency_matrix]):
        return True
    return False


def calculate_max_difference(target_outputs, current_outputs):
    """
    Calculate the maximum absolute difference between target and current outputs.
    """
    differences = []
    for t, c in zip(target_outputs, current_outputs):
        m = t - c
        two = Interval(2, 2)
        differences.append(m**two)

    sum_difference = Interval(0, 0)
    for diff in differences:
        sum_difference += diff

    return sum_difference


def initialize_agents(number_agents, number_input_agents, X):
    """
    Create and return a list of input/output agents.
    """
    import random

    list_of_input_agents = list(range(number_agents))
    for i in range(number_agents - number_input_agents):
        rand = random.randint(0, len(list_of_input_agents) - 1)
        list_of_input_agents.pop(rand)

    i = 0
    intervalar_Functions = []
    intervalar_inverted_functions = []

    output_intervalar_Functions = []
    output_intervalar_inverted_functions = []
    while i < number_agents:
        if i in list_of_input_agents:
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
        else:

            output_intervalar_Functions.append(
                intervalar_functions.output_intervalar_functions_avaliable[
                    i % len(intervalar_functions.output_intervalar_functions_avaliable)
                ][0]
            )
            output_intervalar_inverted_functions.append(
                intervalar_functions.output_intervalar_functions_avaliable[
                    i % len(intervalar_functions.output_intervalar_functions_avaliable)
                ][1]
            )

        i += 1
    agents = []
    output_agents = []
    agent_output_count = -1
    agent_input_count = -1
    for index in range(len(intervalar_Functions) + len(output_intervalar_Functions)):

        agent = None
        if index in list_of_input_agents:
            agent_input_count += 1
            agent = Agent(
                X=X,
                index=index,
                function=intervalar_Functions[agent_input_count],
                inverted_function=intervalar_inverted_functions[agent_input_count],
                isOutput=False,
                num_agents=number_agents,
            )
        else:
            agent_output_count += 1
            agent = Agent(
                X=X,
                index=index,
                function=output_intervalar_Functions[agent_output_count],
                inverted_function=output_intervalar_inverted_functions[
                    agent_output_count
                ],
                num_agents=number_agents,
            )
            output_agents.append(agent)

        agents.append(agent)

    return agents, output_agents


def initialize_coordinator_agent(model, X_init, Y_init):
    """
    Create and return a Coordinator agent with a dummy model.
    """

    coordinator = coord.Coordinator(model=model)
    middle_interval_point_Y = []
    for y in Y_init:

        middle_interval_point_Y.append((y.lower + y.upper) / 2)

    Y_init.clear()
    Y_init = middle_interval_point_Y
    coordinator.update([[X_init], [Y_init]])
    return coordinator


def initialize_corrector_agent(coordinator):
    """
    Create and return a Corrector agent.
    """
    return aget_correct.Corrector(coordinator=coordinator)


def initialize_input_agents_with_X(agents, P):
    """
    Initialize each input agent with the input X.
    """

    for index, agent in enumerate(agents):
        agent.FoG(P=P, agents=agents)

    return


def watch_overflow(a):

    a = np.float32(a)

    r = np.clip(np.array([a]), -max_index, max_index)
    return a


def show(
    current_output,
    epoch,
    time,
    finish_time=None,
    expected_data=[Interval(0, 0), Interval(0, 0)],
):

    show_finish_time = ""
    if finish_time is not None:
        show_finish_time = f"<<_______FINISH TIME: {finish_time} seconds_________>>"

    os.system("cls")
    file = open(f"./epochs/epoch_{epoch+1}.txt", "a")
    file.write(
        f"____________________________________________TIME:{time} seconds____>>> \n"
    )
    for index, variable in enumerate(current_output):

        file.write(f"Variable {index}:\n")
        file.write(
            f"lower_bound: {variable.lower} \n upper_bound: {variable.upper} \n ___expected value____>>: \n expected_value_{index} lower_bound: {expected_data[index].lower} \n expected_value_{index} upper_bound: {expected_data[index].upper} \n \n {show_finish_time}"
        )


def normalize_agents(agents):
    max_val = 0
    for agent in agents:  # pick max value
        max_agent_value = max(
            abs(agent.current_output.lower), abs(agent.current_output.upper)
        )
        if max_agent_value > max_val:
            max_val = max_agent_value

    if max_val == 0:
        return agents

    for agent in agents:
        agent.current_output /= Interval(max_val, max_val)

    return agents


# =============================================================================
# Main Process
# =============================================================================
def main_process(dataset, max_iterations, epochs=1):
    """
    Main process that iterates over the dataset.

    Args:
        dataset: a list of (X, Y) pairs.
        max_iterations: maximum number of iterations per sample.

    Returns:
        results: collected outputs from the output agents.
    """
    # Initialize agents
    n_out = len(dataset[0]["Y"])
    n_in = len(dataset[0]["X"]) - n_out
    agents, output_agent = initialize_agents(
        number_agents=n_in + n_out, number_input_agents=n_in, X=dataset[0]["X"]
    )

    rbf_interpolator_model = model.model(Ninput=n_in, Noutput=n_in**2)

    X_init = dataset[0]["X"]
    Y_init = []
    for i in X_init:
        Y_init.extend(X_init)

    coordinator = initialize_coordinator_agent(
        model=rbf_interpolator_model,
        X_init=X_init,
        Y_init=Y_init,
    )
    corrector = initialize_corrector_agent(coordinator=coordinator)
    convergence_threshold = 0.01

    results = []
    # For each (X, Y) pair in the dataset:
    # prediction_time_avg = []
    agent_input = []

    import time

    start_time = time.time()
    converged = False
    for epoch in range(epochs):

        try:
            file = open(f"./epochs/epoch_{epoch + 1}.txt", "w")
            file.write("")
            file.close()
        except:
            pass

        for index, data in enumerate(dataset):

            X = data["X"]
            Y = data["Y"]
            agent_input = deepcopy(X)
            iteration = 1
            stack_edges = []
            # Reset agents’ defaults
            for agent in agents:
                agent.init_input(X=X)

            ellapse = time.time()
            while iteration <= max_iterations and not converged:

                # Step 1: Coordinator determines arcs
                adjacency_matrix = coordinator.generate_arcs(agent_input)

                agents = normalize_agents(agents=agents)
                initialize_input_agents_with_X(agents=agents, P=adjacency_matrix)

                # Save current state (using shallow copies for demonstration)
                stack_edges.append((adjacency_matrix, agents))

                # Step 2: Check termination condition
                if no_arcs_exist(adjacency_matrix):
                    converged = True
                    break

                # Step 3: Check convergence
                # Here we assume the target outputs are provided as Y (a list)
                current_outputs = []
                agent_input.clear()
                for agent in agents:
                    if agent.isOutput:
                        current_outputs.append(
                            Interval(
                                lower=agent.current_output.lower,
                                upper=agent.current_output.upper,
                            )
                        )

                    agent_input.append(agent.current_output)

                delta = calculate_max_difference(Y, current_outputs)
                if delta.lower < 0:
                    delta += Interval(lower=abs(delta.lower), upper=abs(delta.lower))

                results.append(current_outputs)
                # input()
                iteration += 1

                if delta.upper < convergence_threshold:
                    converged = True
                    break

            show(
                current_output=current_outputs,
                epoch=epoch,
                time=time.time() - ellapse,
                expected_data=data["Y"],
            )
            # Final output collection

            # If not converged, run the correction phase and update the coordinator
            if not converged:
                arc_adjustments = corrector.correction_phase(
                    agents=agents, stack_edges=stack_edges, Y=Y, X=X
                )
                coordinator.update(arc_adjustments)
            else:
                os.system("cls")
                print("Ajustment has converged.")
                break

        if epoch == epochs - 1:
            show(
                current_output=current_outputs,
                epoch=epoch,
                time=time.time() - ellapse,
                finish_time=time.time() - start_time,
            )

        if delta.upper < convergence_threshold:
            converged = True
            break


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Create a dummy dataset of (X, Y) pairs.
    # Here, X might be a number (or a vector) and Y is a list of target outputs.
    dataset = compartimental_models.dataset_generator(epidemic_number=30)
    max_iterations = 5

    main_process(dataset, max_iterations, epochs=10)
