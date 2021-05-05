import numpy as np


class NoLineal:
    def __init__(self, error, x, y, w, alpha, itera, lamb, norma):
        self.error = error
        self.x = x
        self.y = y
        self.w = w.copy()
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
          g = 0 
          if self.norma == "L1":  
            g = np.array([self.error.gradienteREG_L1(self.model,self.x, self.y, self.w, self.lamb)])

            self.w = self.w - self.alpha * g
            if epoch % 1000 == 0:
              print("epoch:", epoch, "\t",self.error.name,"norma", self.norma + ":", round(self.error.errorREG_L1(self.model, self.x,self.y, self.w,self.lamb), 5))
            
          if self.norma == "L2":  
            g = np.array([self.error.gradienteREG_L2(self.model, self.x, self.y, self.w, self.lamb)])
            self.w = self.w - self.alpha * g
            
            if epoch % 1000 == 0:
              print("epoch:", epoch, "\t",self.error.name,"norma", self.norma + ":", round(self.error.errorREG_L2(self.model, self.x,self.y, self.w,self.lamb), 5))
                     
          epoch += 1

    def calcularY(self):
        y = [self.model(self.w, i) for i in self.x[0]]
        return y
