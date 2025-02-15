from agent_definition import Agent
from intervalar_functions import Interval
import numpy as np

import sys

max_index = sys.maxsize


# =============================================================================
# Coordinator
# =============================================================================
class DummyModel:
    def predict(self, X):
        """
        Dummy prediction method that returns an "adjacency matrix".
        In a real implementation, this would be a learned model's prediction.
        """
        # For example, return a 2x2 dummy matrix (or any structure needed)
        return [[1, 0], [0, 1]]


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
                t = np.float32((x.lower + x.upper) / 2)
                if t == np.float32("inf"):
                    t = max_index
                middle_interval_point.append(t)

            if isinstance(x, Agent):
                if x.isOutput:
                    continue

                t = np.float32((x.current_output.lower + x.current_output.upper) / 2)
                if t == np.float32("inf"):
                    t = max_index
                middle_interval_point.append(t)

        return self.model.predict(middle_interval_point)

    def update(self, adjustments):
        """
        Update the coordinator (e.g. update the ML model) based on the adjustments.
        """
        self.training_data.append(adjustments)
        self.retrain_model(adjustments)

    def retrain_model(self, adjustment):
        """
        Dummy retraining routine.
        """
        print("Retraining model with new training data...")

        middle_interval_point_X = [(x.lower + x.upper) / 2 for x in adjustment[0]]
        middle_interval_point_X = [middle_interval_point_X]

        middle_interval_point_Y = []
        for Y in adjustment[1]:
            for y in Y:
                if isinstance(y, Interval):
                    middle_interval_point_Y.append((y.lower + y.upper) / 2)
                else:
                    middle_interval_point_Y.append(y)

        middle_interval_point_Y = [middle_interval_point_Y]
        self.model.update(middle_interval_point_X, middle_interval_point_Y)
