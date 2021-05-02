class MAE:
    @staticmethod
    def errorREG_L1(model, x, y, w, lamb):
        suma = 0
        for j in range(len(w[0])):
            for i in range(len(x[0])):
                suma += abs(y[0][i] - model(w, x[0][i])) / (len(y[0]))

        w2 = abs(w[0])
        return suma + (1/ (2*len(y[0])) ) * lamb * sum(w2)

    @staticmethod
    def errorREG_L2(model, x, y, w, lamb):
        suma = 0
        for j in range(len(w[0])):
            for i in range(len(x[0])):
                suma += abs(y[0][i] - model(w, x[0][i])) / (len(y[0]))

        w2 = w[0]**2
        return suma + (1/ (2*len(y[0])) ) * lamb * sum(w2)

    @staticmethod
    def gradienteREG_L1(model, x, y, w, lamb):
        w_fin = []
        for j in range(len(w[0])):
            suma = 0
            for i in range(len(x[0])):
                suma += (((y[0][i] - model(w, x[0][i])) / (abs(y[0][i] - model(w, x[0][i])))) * (-(x[0][i]) ** j))

            suma = (suma / len(y[0])) + (1/(2*len(y[0]))) * lamb * (w[0][j]/abs(w[0][j]))
            w_fin.append(suma)

        return w_fin

    @staticmethod
    def gradienteREG_L2(model, x, y, w, lamb):
        w_fin = []
        for j in range(len(w[0])):
            suma = 0
            for i in range(len(x[0])):
                suma += (((y[0][i] - model(w, x[0][i])) / (abs(y[0][i] - model(w, x[0][i])))) * (-(x[0][i]) ** j))

            suma = (suma / len(y[0])) + (1/len(y[0])) * lamb * w[0][j]
            w_fin.append(suma)

        return w_fin
