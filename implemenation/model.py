import numpy as np

# from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class general_model:

    X = []
    Y = []

    def __init__():
        pass

    def update():
        pass

    def predict():
        pass


class model(general_model):

    # Example data
    X = []
    Y = []
    model = None
    NInput = 0
    Noutput = 0

    def __init__(self, Ninput, Noutput):
        self.NInput = Ninput
        self.Noutput = Noutput
        self.model = RandomForestRegressor()

    def update(self, X, Y):

        Y = np.array(Y)
        X = np.array(X)

        self.model.fit(X=X, y=Y)

        return

    def predict(self, X):

        X = np.array([X])

        y_pred = self.model.predict(X=X)

        y_pred = y_pred.flatten()

        return y_pred
