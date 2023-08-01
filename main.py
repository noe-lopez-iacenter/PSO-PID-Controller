''' This Python Script uses PSO to find the best params for PID controller '''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pyswarm import pso

# Modelo del sistema masa-resorte-amortiguador
def mass_spring_damper(t, z, m, k, c, u):
    dzdt = [z[1], (u - c * z[1] - k * z[0]) / m]
    return dzdt

# Función objetivo: Error cuadrático medio (MSE)
def mse(x, *args):
    kp, ki, kd = x
    m, k, c, u_target, y0 = args

    def control_law(t, y):
        error = u_target - y[0]
        integral = np.trapz(y[0], t)
        derivative = np.gradient(y[0], t)
        u = kp * error + ki * integral + kd * derivative
        return u

    t_span = (0, t[-1])
    sol = solve_ivp(mass_spring_damper, t_span, y0, args=(m, k, c, control_law), t_eval=t)
    y = sol.y
    u = control_law(sol.t, y)

    return np.mean((u_target - u) ** 2)

# Parámetros del sistema y controlador
m = 1.0  # Masa del sistema
k = 10.0  # Constante del resorte
c = 1.0  # Coeficiente de amortiguamiento
u_target = np.sin(np.linspace(0, 10, 100))  # Señal de referencia
y0 = [0.0, 0.0]  # Condiciones iniciales

# Rango de valores para los parámetros del controlador PID
lower_bound = [0.0, 0.0, 0.0]  # [kp, ki, kd]
upper_bound = [10.0, 5.0, 2.0]

# Optimización utilizando PSO
best_params, _ = pso(mse, lower_bound, upper_bound, args=(m, k, c, u_target, y0))

# Resultados y visualización
print("Mejores parámetros del controlador PID (kp, ki, kd):", best_params)

# Simulación del sistema con los parámetros ajustados
t_span = np.linspace(0, 10, 100)
sol = solve_ivp(mass_spring_damper, (0, t_span[-1]), y0, args=(m, k, c, lambda t, y: best_params[0] * (u_target - y[0]) + best_params[1] * np.cumsum(u_target - y[0]) + best_params[2] * np.gradient(u_target - y[0], t_span)), t_eval=t_span)
y_simulated = sol.y[0]

# Visualización de los resultados
plt.figure(figsize=(10, 6))
plt.plot(t_span, u_target, label='Referencia')
plt.plot(t_span, y_simulated, label='Respuesta simulada')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Sistema masa-resorte-amortiguador con controlador PID ajustado por PSO')
plt.legend()
plt.grid(True)
plt.show()
