import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import override_arithmentic_opt as interval_opt
import random
import math
import sys

max_index = 1e50


def sir_model_equations(y, t, beta, gamma):
    """
    Evaluate the derivatives of the SIR model at time t.

    Parameters:
        y : list or array
            Current values of the state variables [S, I, R] where:
              S = number of susceptible individuals,
              I = number of infected individuals,
              R = number of recovered individuals.
        t : float
            Current time (not used explicitly in this autonomous system,
            but required by the ODE solver).
        beta : float
            Transmission rate.
        gamma : float
            Recovery rate.

    Returns:
        dydt : list
            Derivatives [dS/dt, dI/dt, dR/dt] of the SIR model.
    """
    S, I, R = y
    N = S + I + R  # Total population
    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]


def modelo_SI(S, I, beta, kargs):
    dSdt = -beta * S * I
    dIdt = beta * S * I
    return [dSdt, dIdt]


def plot(
    upper_S,
    upper_I,
    upper_R,
    lower_S,
    lower_I,
    lower_R,
    days,
    gamma,
    beta,
    index=0,
    fake_beta=0,
    fake_gamma=0,
    data_fake=[],
):
    t = np.linspace(0, days, days)
    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.text(
        0.98,
        0.02 + 0 * 0.05,
        gamma,
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
    )
    plt.text(
        0.98,
        0.02 + 1 * 0.05,
        beta,
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
    )

    # plt.legend()
    # plt.savefig(f"image/plot{index}.png")
    # Datos para los gráficos
    x1 = np.linspace(0, 10, 100)
    y1 = np.sin(x1)

    x2 = np.linspace(0, 10, 100)
    y2 = np.cos(x2)

    fig, axs = plt.subplots(1, 2)

    # Plotear datos en cada subplot (dos columnas)
    # for i in range(len(axs)):
    #     if i == 0:
    #         axs[i].plot(
    #             x1[:50], y1[:50]
    #         )  # Solo primeros valores para demostrar diferencias visuales.

    plt.plot(t, lower_S, "b", label="Susceptible_lower_bound")
    plt.plot(t, lower_I, "r", label="Infected_lower_bound")
    plt.plot(t, lower_R, "g", label="Recovered_lower_bound")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of Individuals")
    plt.title("SIR Model Simulation")
    plt.legend()
    plt.grid(True)

    plt.plot(t, upper_S, "b", label="Susceptible_upper_bound")
    plt.plot(t, upper_I, "r", label="Infected_upper_bound")
    plt.plot(t, upper_R, "g", label="Recovered_upper_bound")

    # else:
    #     axs[i].plot(t, lower_S, "b", label="Susceptible_lower_bound")
    #     axs[i].plot(t, lower_I, "r", label="Infected_lower_bound")
    #     axs[i].plot(t, lower_R, "g", label="Recovered_lower_bound")
    #     axs[i].xlabel("Time (days)")
    #     axs[i].ylabel("Number of Individuals")
    #     axs[i].title("SIR Model Simulation Generated")
    #     axs[i].legend()
    #     axs[i].grid(True)

    plt.tight_layout()

    plt.show()
    # plt.show()


def plot2(sol):
    plt.plot(sol.t, sol.y[0], label="Susceptibles (S)")
    plt.plot(sol.t, sol.y[1], label="Infectados (I)")
    plt.xlabel("Tiempo")
    plt.ylabel("Población")
    plt.title("Modelo SI")
    plt.legend()
    plt.grid()
    plt.show()


def sir_model(S0, I0, R0, days, beta, gamma, index=0, fake_beta=0, fake_gamma=0):

    y0 = [S0, I0, R0]
    t = np.linspace(0, days, days)

    solution = odeint(sir_model_equations, y0, t, args=(beta, gamma))
    # solution = odeint(
    #     modelo_SI,
    #     S0,
    #     I0,
    #     args=beta,
    # )
    # solution_fake = odeint(modelo_SI, y0, t, args=(fake_beta, fake_gamma))

    error = math.ceil(np.abs(math.ceil(solution.mean())) * 2.5)
    error = np.random.randint(0, error, size=(3, days))

    # lower_error = np.random.randint(0, np.abs(error / 2) - 1, size=(3, days))

    # S_fake, I_fake, R_fake = solution_fake.T

    lower_bound = solution.T - error
    upper_bound = solution.T + error

    lower_bound /= np.abs(upper_bound.max())
    upper_bound /= np.abs(upper_bound.max())

    lower_S, lower_I, lower_R = lower_bound
    upper_S, upper_I, upper_R = upper_bound

    # plot(
    #     upper_S,
    #     upper_I,
    #     upper_R,
    #     lower_S,
    #     lower_I,
    #     lower_R,
    #     days,
    #     gamma=gamma,
    #     beta=beta,
    #     index=index,
    #     fake_beta=fake_beta,
    #     fake_gamma=fake_beta,
    #     # data_fake=[S_fake, I_fake],
    # )

    interval_solution = [
        interval_opt.Interval(lower=np.float32(low), upper=np.float32(up))
        for low, up in zip(lower_S, upper_S)
    ]

    interval_solution.extend(
        [
            interval_opt.Interval(lower=np.float32(low), upper=np.float32(up))
            for low, up in zip(lower_I, upper_I)
        ]
    )

    interval_solution.extend(
        [
            interval_opt.Interval(lower=np.float32(low), upper=np.float32(up))
            for low, up in zip(lower_R, upper_R)
        ]
    )

    return interval_solution


def dataset_generator():
    """
    Returns { "upper_bound": np.array , "lower_bound": np.array }

    Shape of np.array = (days,)
    """

    sir_dataset = []
    days = 120
    maximal_param = max_index
    generated_beta = 0
    generated_gramma = 0
    for i in range(150):

        # Initial conditions:
        S0 = random.randint(1, 10000)  # initial number of susceptible individuals
        I0 = random.randint(1, S0)  # initial number of infected individuals
        R0 = 0  # initial number of recovered individuals
        gamma = random.randrange(1, int(maximal_param)) / maximal_param
        beta = random.randrange(1, int(maximal_param)) / maximal_param

        maximal_param = 3 * random.randint(1, int(maximal_param / 2))
        generated_beta = random.randint(0, int(maximal_param))
        generated_gramma = random.randint(0, int(maximal_param))

        # print(
        #     {
        #         "gamma": gamma,
        #         "beta": beta,
        #         "generated_gramma": generated_gramma,
        #         "generated_beta": generated_beta,
        #     }
        # )

        sir_dataset.append(
            {
                "X": sir_model(
                    S0=S0, I0=I0, R0=R0, days=days, gamma=gamma, beta=beta, index=i
                ),
                "Y": (
                    interval_opt.Interval(gamma, gamma),
                    interval_opt.Interval(beta, beta),
                ),
            }
        )

    return sir_dataset


# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# maximal_param = max_index
# generated_beta = 0

# gen_betas = []
# for i in range(150):
#     # Parámetros del modelo
#     beta = random.randint(0, 10) / 1000  # Tasa de transmisión
#     maximal_param = 3 * random.randint(1, int(maximal_param / 2))
#     generated_beta = random.randint(0, int(maximal_param))

#     gen_betas.append(abs(beta - generated_beta))
#     # N = 1000  # Población total
#     # I0 = 1  # Número inicial de infectados
#     # S0 = N - I0  # Número inicial de susceptibles

#     # # Ecuaciones del modelo SI
#     # def modelo_SI(t, y):
#     #     S, I = y
#     #     dSdt = -beta * S * I
#     #     dIdt = beta * S * I
#     #     return [dSdt, dIdt]

#     # # Condiciones iniciales
#     # y0 = [S0, I0]

#     # # Intervalo de tiempo
#     # t_span = (0, 10)
#     # t_eval = np.linspace(0, 10, 1000)

#     # # Resolver el sistema de ecuaciones diferenciales
#     # sol = solve_ivp(modelo_SI, t_span, y0, t_eval=t_eval)

#     # # Graficar los resultados
#     # plt.plot(sol.t, sol.y[0], label="Susceptibles (S)")
#     print(
#         {
#             "beta": beta,
#             "generated_beta": generated_beta,
#         }
#     )
# plt.plot(range(150), gen_betas, label="Infectados (I)")
# plt.xlabel("Generation")
# plt.ylabel("Error")
# plt.title("SI Model")
# plt.legend()
# plt.grid()
# plt.savefig(f"./image/plt{i}2.jpg")
# # plt.show()
