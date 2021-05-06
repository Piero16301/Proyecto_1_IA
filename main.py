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

    w = np.array([np.random.rand(4)])

    errorMSE = MSE()
    errorMAE = MAE()

    # MSE y MAE sin regularizacion
    modelMSE = NoLineal(errorMSE, x, y, w, 0.7, 10000, 0, "MSE")
    modelMAE = NoLineal(errorMAE, x, y, w, 0.7, 10000, 0, "MAE")

    modelMSE.execute()
    modelMAE.execute()

    MSE_y_l1, MSE_y_l2 = modelMSE.calcularY()
    MAE_y_l1, MAE_y_l2 = modelMAE.calcularY()

    plt.plot(x[0], y[0], "*", color="red")
    plt.plot(x[0], MSE_y_l1, color="blue", label="MSE sin REG")
    plt.legend()
    plt.savefig("MSE_sin_REG.png")
    plt.show()
    plt.close()

    plt.plot(x[0], y[0], "*", color="red")
    plt.plot(x[0], MAE_y_l1, color="orange", label="MAE sin REG")
    plt.legend()
    plt.savefig("MAE_sin_REG.png")
    plt.show()
    plt.close()

    # MSE y MAE con regularizacion
    modelMSE = NoLineal(errorMSE, x, y, w, 0.7, 10000, 0.6, "MSE")
    modelMAE = NoLineal(errorMAE, x, y, w, 0.7, 10000, 0.6, "MAE")

    modelMSE.execute()
    modelMAE.execute()

    MSE_y_l1, MSE_y_l2 = modelMSE.calcularY()
    MAE_y_l1, MAE_y_l2 = modelMAE.calcularY()

    plt.plot(x[0], y[0], "*", color="red")
    plt.plot(x[0], MSE_y_l1, color="blue", label="MSE con REG L1")
    plt.legend()
    plt.savefig("MSE_con_REG_L1.png")
    plt.show()
    plt.close()

    plt.plot(x[0], y[0], "*", color="red")
    plt.plot(x[0], MSE_y_l2, color="orange", label="MSE con REG L2")
    plt.legend()
    plt.savefig("MSE_con_REG_L2.png")
    plt.show()
    plt.close()

    plt.plot(x[0], y[0], "*", color="red")
    plt.plot(x[0], MAE_y_l1, color="green", label="MAE con REG L1")
    plt.legend()
    plt.savefig("MAE_con_REG_L1.png")
    plt.show()
    plt.close()

    plt.plot(x[0], y[0], "*", color="red")
    plt.plot(x[0], MAE_y_l2, color="purple", label="MAE con REG L2")
    plt.legend()
    plt.savefig("MAE_con_REG_L2.png")
    plt.show()
    plt.close()
