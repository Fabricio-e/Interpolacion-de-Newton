import numpy as np
import matplotlib.pyplot as plt

def newton_divided_diff(x, y):
    """ Calcula la tabla de diferencias divididas de Newton """
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y  # Primera columna es y

    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])

    return coef[0, :]

def newton_interpolation(x_data, y_data, x):
    """ Evalúa el polinomio de Newton en los puntos x """
    coef = newton_divided_diff(x_data, y_data)
    n = len(x_data)

    y_interp = np.zeros_like(x)
    for i in range(len(x)):
        term = coef[0]
        product = 1
        for j in range(1, n):
            product *= (x[i] - x_data[j - 1])
            term += coef[j] * product
        y_interp[i] = term

    return y_interp, coef

# Datos del ejercicio
V_data = np.array([10, 20, 30, 40, 50, 60])
Cd_data = np.array([0.32, 0.30, 0.28, 0.27, 0.26, 0.25])

# 1. Obtén el polinomio interpolador de Newton para estos datos.
coefficients = newton_divided_diff(V_data, Cd_data)

def polynomial_equation(V):
    """ Evalúa el polinomio interpolador para una velocidad V """
    n = len(V_data)
    result = coefficients[0]
    product = 1
    for i in range(1, n):
        product *= (V - V_data[i - 1])
        result += coefficients[i] * product
    return result

print("1. Polinomio interpolador de Newton:")
equation_str = f"{coefficients[0]:.4f}"
product_terms = ""
for i in range(1, len(V_data)):
    product_terms += f"(V - {V_data[i-1]})"
    equation_str += f" + {coefficients[i]:.4f} * {product_terms}"
print(f"Cd(V) ≈ {equation_str}")

# 2. Estima el coeficiente de arrastre a una velocidad de 35 m/s.
V_estimate = 35
Cd_estimate = polynomial_equation(V_estimate)
print(f"\n2. Coeficiente de arrastre estimado a V = {V_estimate} m/s: {Cd_estimate:.4f}")

# 3. Genera una gráfica comparando los valores interpolados con los datos reales.
V_vals = np.linspace(min(V_data), max(V_data), 100)
Cd_interp_vals = polynomial_equation(V_vals)

plt.figure(figsize=(8, 6))
plt.plot(V_data, Cd_data, 'ro', label='Datos experimentales')
plt.plot(V_vals, Cd_interp_vals, 'b-', label='Interpolación de Newton')
plt.scatter(V_estimate, Cd_estimate, color='green', marker='x', s=100, label=f'Estimación en V={V_estimate} m/s')
plt.xlabel('Velocidad del aire (V) [m/s]')
plt.ylabel('Coeficiente de arrastre ($C_d$)')
plt.legend()
plt.title('Interpolación de Newton del coeficiente de arrastre en función de la velocidad')
plt.grid(True)
plt.savefig("interpolacion_coeficiente_arrastre.png")
plt.show()