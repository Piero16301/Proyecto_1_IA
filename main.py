from NoLineal import NoLineal
from MSE import MSE

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_temp = np.arange(0, 1, 0.05)
    y_temp = [np.sin(2 * i * np.pi) + np.random.normal(0, 0.2) for i in x_temp]
    x = np.array([x_temp])
    y = np.array([y_temp])
    plt.plot(x[0], y[0], "*", color="red")

    w = np.array([np.random.rand(4)])

    error = MSE()
    model = NoLineal(error, x, y, w, 0.7, 10000, 1, "MSE")
    model.execute()
    y_l1, y_l2 = model.calcularY()
    plt.plot(x[0], y_l1, color="blue", label="MSE L1")
    plt.plot(x[0], y_l2, color="green", label="MSE L2")
    plt.legend()
    plt.show()
