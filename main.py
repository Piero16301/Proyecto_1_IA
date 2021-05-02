from NoLineal import NoLineal
from MSE import MSE
from MAE import MAE

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_temp = np.arange(0, 1, 0.05)
    y_temp = [np.sin(2 * i * np.pi) + np.random.normal(0, 0.2) for i in x_temp]
    x = np.array([x_temp])
    y = np.array([y_temp])
    plt.plot(x[0], y[0], "*", color="red")

    w = np.array([np.random.rand(4)])

    errorMSE = MSE()
    errorMAE = MAE()

    modelMSE = NoLineal(errorMSE, x, y, w, 0.7, 10000, 1, "MSE")
    modelMAE = NoLineal(errorMAE, x, y, w, 0.7, 10000, 1, "MAE")

    modelMSE.execute()
    modelMAE.execute()

    MSE_y_l1, MSE_y_l2 = modelMSE.calcularY()
    MAE_y_l1, MAE_y_l2 = modelMAE.calcularY()

    plt.plot(x[0], MSE_y_l1, color="blue", label="MSE L1")
    plt.plot(x[0], MSE_y_l2, color="green", label="MSE L2")
    plt.plot(x[0], MAE_y_l1, color="orange", label="MAE L1")
    plt.plot(x[0], MAE_y_l2, color="purple", label="MAE L2")
    plt.legend()
    plt.show()
