from NoLineal import NoLineal
from MSE import MSE
from MAE import MAE

import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    
    x_temp = []
    y_temp = []
    with open('x_ds.txt') as file:
        for row in file:
            line = row.strip()
            x_temp.append(float(line))

    with open('y_ds.txt') as file:
        for row in file:
            line = row.strip()
            y_temp.append(float(line))

    x = np.array([x_temp])
    y = np.array([y_temp])
    #plt.plot(x[0], y[0], "*", color="red")

    #Grado
    p = 3
    w = np.array([np.random.rand(p+1)])

    errorMSE = MSE()
    errorMAE = MAE()

    l_rate = 0.7
    lamb = 0.01

    """
    modelMSE_l1 = NoLineal(errorMSE, x, y, w, l_rate, 10000, lamb, "L1")
    modelMAE_l1 = NoLineal(errorMAE, x, y, w, l_rate, 10000, lamb, "L1")
    modelMAE_l2 = NoLineal(errorMAE, x, y, w, l_rate, 10000, lamb, "L2")
    modelMSE = NoLineal(errorMSE, x, y, w, l_rate, 10000, 0, "No_reg")
    modelMAE = NoLineal(errorMAE, x, y, w, l_rate, 10000, 0, "No_reg")"""

    modelMSE_l2 = NoLineal(errorMSE, x, y, w, l_rate, 10000, lamb, "L2")
    modelMom = NoLineal(errorMSE, x, y, w, l_rate, 10000, lamb, "L2") 
    modelAdagrad = NoLineal(errorMSE, x, y, w, l_rate, 10000, lamb, "L2")
    modelAdam = NoLineal(errorMSE, x, y, w, l_rate, 10000, lamb, "L2") 
    modelAdadelta = NoLineal(errorMSE, x, y, w, l_rate, 10000, lamb, "L2")
    
    """
    modelMSE_l1.execute()
    print("\n")
    modelMAE_l1.execute()
    print("\n")
    modelMSE_l2.execute()
    print("\n")
    modelMAE_l2.execute()
    print("\n")
    modelMSE.execute()
    modelMAE.execute()"""
    
    modelMSE_l2.execute()
    print("\n")
    modelMom.execute_Momentum()
    print("\n")
    modelAdagrad.execute_AdaGrad()
    print("\n")
    modelAdam.execute_Adam()
    print("\n")
    modelAdadelta.execute_Adadelta()

    """
    MSE_y_l1 = modelMSE_l1.calcularY()
    MSE_y_l2 = modelMSE_l2.calcularY()
    MAE_y_l1 = modelMAE_l1.calcularY()
    MAE_y_l2 = modelMAE_l2.calcularY()
    MSE_y = modelMSE.calcularY()
    MAE_y = modelMAE.calcularY()"""
    
    """
    MSE_y_l2 = modelMSE_l2.calcularY()
    MSE_y_MOM = modelMom.calcularY() 
    MSE_y_Adam = modelAdam.calcularY()
    MSE_y_Adagrad = modelAdagrad.calcularY()
    MSE_y_Adadelta = modelAdadelta.calcularY() """

    """Curvas sin opti
    plt.plot(x[0], MSE_y_l1, color="black", label="MSE L1")
    plt.plot(x[0], MSE_y_l2, color="green", label="MSE L2")
    plt.plot(x[0], MAE_y_l1, color="orange", label="MAE L1")
    plt.plot(x[0], MAE_y_l2, color="yellow", label="MAE L2")
    plt.plot(x[0], MSE_y, color="blue", label="MSE")
    plt.plot(x[0], MAE_y, color="grey", label="MAE")"""

    
    """ERRORS
    x_err = [ i for i in range(len(modelMSE_l1.arr_err))] 
    plt.plot(x_err, modelMSE_l1.arr_err, color="black", label="MSE L1")
    plt.plot(x_err, modelMSE_l2.arr_err, color="green", label="MSE L2")
    plt.plot(x_err, modelMAE_l1.arr_err, color="orange", label="MAE L1")
    plt.plot(x_err, modelMAE_l2.arr_err, color="yellow", label="MAE L2")
    plt.plot(x_err, modelMSE.arr_err, color="blue", label="MSE")
    plt.plot(x_err, modelMAE.arr_err, color="grey", label="MAE")  """
    
    plt.plot([ i for i in range(len(modelMSE_l2.arr_err))], modelMSE_l2.arr_err, color="green", label="MSE L2")
    plt.plot([ i for i in range(len(modelMom.arr_err))], modelMom.arr_err, color="black", label="Momentum")
    plt.plot([ i for i in range(len(modelAdagrad.arr_err))], modelAdagrad.arr_err, color="orange", label="Adagrad")
    plt.plot([ i for i in range(len(modelAdadelta.arr_err))], modelAdadelta.arr_err, color="yellow", label="Adadelta")
    plt.plot([ i for i in range(len(modelAdam.arr_err))], modelAdam.arr_err, color="blue", label="Adam")
 
    
    """Curvas OPT
    plt.plot(x[0], MSE_y_l2, color="green", label="MSE L2")
    plt.plot(x[0], MSE_y_MOM, color="blue", label="MSE L2 Momentum")
    plt.plot(x[0], MSE_y_Adagrad, color="red", label="MSE L2 Adagrad")
    plt.plot(x[0], MSE_y_Adam, color="purple", label="MSE L2 Adam")
    plt.plot(x[0], MSE_y_Adadelta, color="brown", label="MSE L2 Adadelta")"""
    plt.legend()
    plt.show()
