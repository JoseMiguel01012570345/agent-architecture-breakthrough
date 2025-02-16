import numpy as np

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

max_index = np.finfo(np.float32).max


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

        Y = np.clip(Y, -max_index, max_index)
        X = np.clip(X, -max_index, max_index)

        max_x = np.abs(X).max()
        max_y = np.abs(Y).max()
        if max_x == 0:
            max_x = 1
        if max_y == 0:
            max_y = 1

        X /= max_x
        Y /= max_y

        self.model.fit(X=X, y=Y)

        return

    def predict(self, X):

        X = np.array([X])
        X = np.float32(X)
        X /= np.abs(X).max()

        y_pred = self.model.predict(X=X)

        y_pred = y_pred.flatten()

        return y_pred
