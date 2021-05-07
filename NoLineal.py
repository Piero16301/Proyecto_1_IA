import numpy as np


class NoLineal:
    def __init__(self, error, x, y, w, alpha, itera, lamb, norma):
        self.error = error
        self.x = x
        self.y = y
        self.w_L1 = w.copy()
        self.w_L2 = w.copy()
        self.alpha = alpha
        self.itera = itera
        self.lamb = lamb
        self.norma = norma

    @staticmethod
    def model(w, x_val):
        suma = w[0][0]
        for i in range(1, len(w[0])):
            suma += (w[0][i] * (x_val ** i))
        return suma

    def execute(self):
        epoch = 1
        while self.itera:
            self.itera -= 1

            # [[w1,w2,w3]]
            g_l1 = np.array([self.error.gradienteREG_L1(self.model, self.x, self.y, self.w_L1, self.lamb)])
            self.w_L1 = self.w_L1 - self.alpha * g_l1

            # [[w1,w2,w3]]
            g_l2 = np.array([self.error.gradienteREG_L2(self.model, self.x, self.y, self.w_L2, self.lamb)])
            self.w_L2 = self.w_L2 - self.alpha * g_l2
            if epoch % 1000 == 0:
                print("epoch:", epoch, "\t", self.norma + "_REG_L1:", round(self.error.errorREG_L1(self.model, self.x,
                                                                                                   self.y, self.w_L1,
                                                                                                   self.lamb), 5), "\t",
                      self.norma + "_REG_L2:", round(self.error.errorREG_L2(
                        self.model, self.x, self.y, self.w_L1, self.lamb), 5))

            epoch += 1

    def calcularY(self):
        y_l1 = [self.model(self.w_L1, i) for i in self.x[0]]
        y_l2 = [self.model(self.w_L2, i) for i in self.x[0]]
        return y_l1, y_l2
