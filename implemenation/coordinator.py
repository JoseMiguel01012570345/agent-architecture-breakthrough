from agent_definition import Agent
from intervalar_functions import Interval
import numpy as np

max_index = np.finfo(np.float32).max


# =============================================================================
# Coordinator
# =============================================================================
class Coordinator:

    training_data = []

    def __init__(self, model):
        self.model = model

    def generate_arcs(self, X):
        """
        Given an input X, generate an adjacency matrix.
        """
        middle_interval_point = []
        for x in X:
            if isinstance(x, Interval):

                middle_interval_point.append(
                    self.watch_overflow((x.lower + x.upper) / 2)
                )  # watch overflow

            if isinstance(x, Agent):

                t = np.float32((x.current_output.lower + x.current_output.upper) / 2)
                if np.isposinf(t):
                    t = max_index
                middle_interval_point.append(t)

        return self.model.predict(middle_interval_point)

    def update(self, adjustment):
        """
        Update the coordinator (e.g. update the ML model) based on the adjustments.
        """
        self.training_data.append(adjustment)

        middle_interval_point_X = []
        for X in adjustment[0]:
            row = []
            for x in X:
                row.append(
                    self.watch_overflow((x.lower + x.upper) / 2)
                )  # watch overflow

            middle_interval_point_X.append(row)

        middle_interval_point_Y = []
        for Y in adjustment[1]:
            row = []
            for y in Y:
                row.append(y)

            middle_interval_point_Y.append(row)

        self.model.update(middle_interval_point_X, middle_interval_point_Y)

    def retrain_model(self, adjustment):
        """
        Dummy retraining routine.
        """
        print("Retraining model with new training data...")

        middle_interval_point_X = []
        for X in adjustment[0]:
            row = []
            for x in X:
                row.append(
                    self.watch_overflow((x.lower + x.upper) / 2)
                )  # watch overflow

            middle_interval_point_X.append(row)

        middle_interval_point_Y = []
        for Y in adjustment[1]:
            for y in Y:
                if isinstance(y, Interval):
                    middle_interval_point_Y.append(
                        self.watch_overflow((y.lower + y.upper) / 2)
                    )
                else:
                    middle_interval_point_Y.append(y)

        middle_interval_point_Y = [middle_interval_point_Y]
        self.model.update(middle_interval_point_X, middle_interval_point_Y)

    def watch_overflow(self, t):
        if np.isposinf(t):
            t = max_index

        return t
