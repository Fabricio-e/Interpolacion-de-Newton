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
T_data = np.array([200, 250, 300, 350, 400])
efficiency_data = np.array([30, 35, 40, 46, 53])

# 1. Construye el polinomio interpolador de Newton
coefficients = newton_divided_diff(T_data, efficiency_data)

def polynomial_equation(T):
    """ Evalúa el polinomio interpolador para una temperatura T """
    n = len(T_data)
    result = coefficients[0]
    product = 1
    for i in range(1, n):
        product *= (T - T_data[i - 1])
        result += coefficients[i] * product
    return result

print("1. Polinomio interpolador de Newton:")
equation_str = f"{coefficients[0]:.4f}"
product_terms = ""
for i in range(1, len(T_data)):
    product_terms += f"(T - {T_data[i-1]})"
    equation_str += f" + {coefficients[i]:.4f} * {product_terms}"
print(f"Eficiencia(T) ≈ {equation_str}")

# 2. Predice la eficiencia del motor para T = 275 °C
T_predict = 275
efficiency_predict = polynomial_equation(T_predict)
print(f"\n2. Eficiencia predicha para T = {T_predict}°C: {efficiency_predict:.2f}%")

# 3. Representa gráficamente la interpolación junto con los datos experimentales
T_vals = np.linspace(min(T_data), max(T_data), 100)
efficiency_interp_vals = polynomial_equation(T_vals)

plt.figure(figsize=(8, 6))
plt.plot(T_data, efficiency_data, 'ro', label='Datos experimentales')
plt.plot(T_vals, efficiency_interp_vals, 'b-', label='Interpolación de Newton')
plt.scatter(T_predict, efficiency_predict, color='green', marker='x', s=100, label=f'Predicción en T={T_predict}°C')
plt.xlabel('Temperatura de entrada (T) [°C]')
plt.ylabel('Eficiencia [%]')
plt.legend()
plt.title('Interpolación de Newton de la eficiencia de un motor térmico')
plt.grid(True)
plt.savefig("interpolacion_eficiencia_motor.png")
plt.show()