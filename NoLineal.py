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
        self.arr_err = []
        
    
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
          
          if self.norma == "L1" :  
            g = np.array([self.error.gradienteREG_L1(self.model,self.x, self.y, self.w, self.lamb)])

            self.w = self.w - self.alpha * g
            err = round(self.error.errorREG_L1(self.model, self.x,self.y, self.w,self.lamb), 7)
            self.arr_err.append(err)              
            if epoch % 1000 == 0:
              print("epoch:", epoch, "\t",self.error.name,"norma", self.norma + ":",err)
            
          if self.norma == "L2" : 
            g = np.array([self.error.gradienteREG_L2(self.model, self.x, self.y, self.w, self.lamb)])

            self.w = self.w - self.alpha * g
            err = round(self.error.errorREG_L2(self.model, self.x,self.y, self.w,self.lamb), 7)
            self.arr_err.append(err)
            if epoch % 1000 == 0:
              print("epoch:", epoch, "\t",self.error.name,"norma", self.norma + ":",err)
          
          if self.norma == "No_reg" : 
            g = np.array([self.error.gradienteREG_L2(self.model, self.x, self.y, self.w, self.lamb)])

            self.w = self.w - self.alpha * g
            err = round(self.error.errorREG_L2(self.model, self.x,self.y, self.w,self.lamb), 7)
            self.arr_err.append(err)
            if epoch % 1000 == 0:
               
              print("epoch:", epoch, "\t",self.error.name,"norma", self.norma + ":", err)
                    
        
          epoch+=1
          

    def execute_Momentum(self):
        epoch = 1
        gamma = 0.9
        v = np.zeros((self.w.shape))
        
        while self.itera:
          self.itera -= 1
          g = 0 
             
          g = np.array([self.error.gradienteREG_L1(self.model,self.x, self.y, self.w, self.lamb)])

          v = gamma*v-self.alpha*g

          self.w = self.w + v

          err = round(self.error.errorREG_L2(self.model, self.x,self.y, self.w,self.lamb), 7)
          if err<1:
            self.arr_err.append(err)
          if epoch % 1000 == 0:
            print("epoch:", epoch, "\t","Momentum",self.error.name,"norma", self.norma + ":",err)
          epoch += 1


    def execute_AdaGrad(self):
        #Initialization
        cache = np.zeros((self.w.shape))
        eps = 1e-4

        epoch = 1
        while self.itera:
          self.itera -= 1
          g = 0 
             
          g = np.array([self.error.gradienteREG_L2(self.model,self.x, self.y, self.w, self.lamb)])
          
          cache += g**2 
          self.w = self.w-(self.alpha * g)/( (np.sqrt(cache)+eps))
          
          err = round(self.error.errorREG_L2(self.model, self.x,self.y, self.w,self.lamb), 7)
          if err<1:
            self.arr_err.append(err)
          if epoch % 1000 == 0:
            print("epoch:", epoch, "\t","Adagrad",self.error.name,"norma", self.norma + ":",err)
          epoch += 1

    def execute_Adadelta(self):
        epoch = 1
        gamma = 0.9
        g_0 = np.zeros((self.w.shape)) 
        eps = 1e-4
        E_g = np.zeros((self.w.shape))
        E_w = np.zeros((self.w.shape))
        while self.itera:
          self.itera -= 1
         
          g = np.array([self.error.gradienteREG_L2(self.model,self.x, self.y, self.w, self.lamb)])

          #E = gamma*E + (1-gamma)*np.array([g[0]**2]) 
          E_g = gamma*E_g + (1-gamma)*(g**2)
          RMSg_t = (np.sqrt(E_g)+eps)
          delta_t = ((self.alpha * g)/ RMSg_t)
          E_w = gamma*E_w + (1-gamma)*(delta_t**2)
          RMSw_t = (np.sqrt(E_w)+eps)   

          self.w = self.w - (RMSw_t/RMSg_t)*g 

          err = round(self.error.errorREG_L2(self.model, self.x,self.y, self.w,self.lamb), 7)
          if err<1:
            self.arr_err.append(err)
          if epoch % 1000 == 0:
            print("epoch:", epoch, "\t","Adadelta",self.error.name,"norma", self.norma + ":",err)
          epoch += 1
            
       



    def execute_Adam(self):
        epoch = 1
        beta1 = 0.9
        beta2 = 0.999
        v = np.zeros((self.w.shape))
        m = np.zeros((self.w.shape))
        eps = 1e-8

        while self.itera:
          self.itera -= 1
          g = 0 
          
          g = np.array([self.error.gradienteREG_L2(self.model,self.x, self.y, self.w, self.lamb)])

          m = beta1*m + (1-beta1)*g
          v = beta2*v + (1-beta2)*(g**2)
          self.w = self.w - (self.alpha * m)/ (np.sqrt(v) + eps)

          
          err = round(self.error.errorREG_L2(self.model, self.x,self.y, self.w,self.lamb), 7)
          if err<1:
            self.arr_err.append(err)
          if epoch % 1000 == 0:
            print("epoch:", epoch, "\t","Adam",self.error.name,"norma", self.norma + ":",err)
          epoch += 1
          
          


    def calcularY(self):
        y = [self.model(self.w, i) for i in self.x[0]]
        return y
