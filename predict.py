import numpy as np 

class ModeloRegresionLineal:
    def __init__(self, X, y):
        """
        Constructor de la clase.

        Args:
            X (np.array): Matriz de características (en este caso, publicidad).
            y (np.array): Vector de valores objetivo (en este caso, ventas).
        """
        self.X = X
        self.y = y
        self.n = len(X)
        self.beta = self._calcular_betas()

    def _calcular_betas(self):
        """
        Calcula los valores óptimos de los parámetros Beta utilizando el método de mínimos cuadrados.

        Returns:
            np.array: Vector de coeficientes (Beta_0 y Beta_1).
        """
        X_mean = np.mean(self.X)
        y_mean = np.mean(self.y)
        numerador = np.sum((self.X - X_mean) * (self.y - y_mean))
        denominador = np.sum((self.X - X_mean) ** 2)
        beta_1 = numerador / denominador
        beta_0 = y_mean - beta_1 * X_mean
        return np.array([beta_0, beta_1])

    def predecir(self, x):
        """
        Realiza una predicción para un nuevo valor de X.

        Args:
            x (float): Valor de la característica (publicidad).

        Returns:
            float: Valor predicho.
        """
        return self.beta[0] + self.beta[1] * x

    def ecuacion_regresion(self):
        """
        Devuelve la ecuación de regresión en formato de cadena.

        Returns:
            str: Ecuación de regresión.
        """
        return f"y = {self.beta[0]:.2f} + {self.beta[1]:.2f}x"

# Datos del problema
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])

# Crear el modelo
modelo = ModeloRegresionLineal(X, y)

# Imprimir la ecuación de regresión
print("Ecuación de regresión:", modelo.ecuacion_regresion())

# Hacer una predicción
nuevo_valor_x = float(input("Ingrese el valor de publicidad para predecir las ventas: "))
prediccion = modelo.predecir(nuevo_valor_x)
print("Predicción de ventas:", prediccion)
