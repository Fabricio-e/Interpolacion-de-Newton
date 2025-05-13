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

    # Construcción de la ecuación polinómica (opcional)
    polynomial_equation = f"{coef[0]:.4f}"
    product_terms = ""
    for i in range(1, n):
        product_terms += f"(x - {x_data[i-1]})"
        polynomial_equation += f" + {coef[i]:.4f} * {product_terms}"

    y_interp = np.zeros_like(x)
    for i in range(len(x)):
        term = coef[0]
        product = 1
        for j in range(1, n):
            product *= (x[i] - x_data[j - 1])
            term += coef[j] * product
        y_interp[i] = term

    return y_interp, polynomial_equation

# Datos del ejercicio
F_data = np.array([50, 100, 150, 200])
epsilon_data = np.array([0.12, 0.35, 0.65, 1.05])

# 1. Obtener la ecuación de interpolación de Newton
epsilon_interp_function, polynomial_equation = newton_interpolation(F_data, epsilon_data, np.array([0]))
print("1. Ecuación de interpolación de Newton:")
print(f"epsilon(F) = {polynomial_equation}")

# 2. Estimar la deformación para una carga de 125 N
F_estimate = 125
epsilon_estimate, _ = newton_interpolation(F_data, epsilon_data, np.array([F_estimate]))
print("\n2. Estimación de la deformación para F = 125 N:")
print(f"epsilon({F_estimate}) ≈ {epsilon_estimate[0]:.4f} mm")

# 3. Generar una gráfica de la interpolación y comparar con los datos originales
F_vals = np.linspace(min(F_data), max(F_data), 100)
epsilon_interp_vals, _ = newton_interpolation(F_data, epsilon_data, F_vals)

plt.figure(figsize=(8, 6))
plt.plot(F_data, epsilon_data, 'ro', label='Datos experimentales')
plt.plot(F_vals, epsilon_interp_vals, 'b-', label='Interpolación de Newton')
plt.scatter(F_estimate, epsilon_estimate, color='green', marker='x', s=100, label=f'Estimación en F={F_estimate} N')
plt.xlabel('Carga aplicada (F) [N]')
plt.ylabel('Deformación ($\\epsilon$) [mm]')
plt.legend()
plt.title('Interpolación de Newton de la deformación en función de la carga')
plt.grid(True)
plt.savefig("interpolacion_deformacion.png")
plt.show()