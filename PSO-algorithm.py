import numpy as np
import matplotlib.pyplot as plt


# Definición del sistema masa-resorte-amortiguador
class MassSpringDamper:
    def __init__(self, mass, spring_constant, damping_coefficient):
        self.mass = mass
        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.position = 0.0
        self.velocity = 0.0

    def update(self, force, dt):
        acceleration = (
            force
            - self.damping_coefficient * self.velocity
            - self.spring_constant * self.position
        ) / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt


# Controlador PID
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def control(self, setpoint, feedback, dt):
        error = setpoint - feedback
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


# Función de evaluación (fitness) para el PSO
def evaluate_fitness(Kp, Ki, Kd):
    # Parámetros del sistema
    mass = 1.0
    spring_constant = 1.0
    damping_coefficient = 0.1

    # Crear el sistema y el controlador
    system = MassSpringDamper(mass, spring_constant, damping_coefficient)
    controller = PIDController(Kp, Ki, Kd)

    # Simulación
    setpoint = 1.0
    max_time = 10.0
    dt = 0.01

    time_points = np.arange(0, max_time, dt)
    positions = []
    for time in time_points:
        feedback = system.position
        control_signal = controller.control(setpoint, feedback, dt)
        system.update(control_signal, dt)
        positions.append(system.position)

    # Calcular el error cuadrático medio (MSE) entre la posición y el setpoint deseado
    mse = np.mean((setpoint - np.array(positions)) ** 2)
    return mse


# Implementación del algoritmo PSO
def pso_optimization(max_iterations, num_particles, bounds):
    num_dimensions = len(bounds)
    particles = np.random.rand(num_particles, num_dimensions)

    # Escalamiento de los parámetros al rango permitido
    for i in range(num_particles):
        for d in range(num_dimensions):
            particles[i, d] = bounds[d][0] + particles[i, d] * (
                bounds[d][1] - bounds[d][0]
            )

    # Mejor posición local de cada partícula
    personal_best = particles.copy()
    personal_best_fitness = np.array(
        [evaluate_fitness(*particle) for particle in personal_best]
    )

    # Mejor posición global de todas las partículas
    global_best_idx = np.argmin(personal_best_fitness)
    global_best = personal_best[global_best_idx]
    global_best_fitness = personal_best_fitness[global_best_idx]

    # Coeficientes de inercia
    w = 0.5
    c1 = 1.5
    c2 = 1.5

    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Actualizar la velocidad y posición de cada partícula
            r1, r2 = np.random.rand(2)
            velocity = (
                w * particles[i]
                + c1 * r1 * (personal_best[i] - particles[i])
                + c2 * r2 * (global_best - particles[i])
            )
            particles[i] += velocity

            # Escalar nuevamente los parámetros al rango permitido
            for d in range(num_dimensions):
                particles[i, d] = np.clip(particles[i, d], bounds[d][0], bounds[d][1])

            # Evaluar la nueva posición
            fitness = evaluate_fitness(*particles[i])

            # Actualizar la mejor posición local de la partícula
            if fitness < personal_best_fitness[i]:
                personal_best[i] = particles[i]
                personal_best_fitness[i] = fitness

            # Actualizar la mejor posición global
            if fitness < global_best_fitness:
                global_best = particles[i]
                global_best_fitness = fitness

    return global_best, global_best_fitness


if __name__ == "__main__":
    # Parámetros para el algoritmo PSO
    max_iterations = 100
    num_particles = 30
    parameter_bounds = [
        (0.1, 10.0),
        (0.01, 5.0),
        (0.01, 2.0),
    ]  # Rango permitido para Kp, Ki y Kd

    # Ejecutar el algoritmo PSO para ajustar los parámetros del controlador PID
    best_params, best_fitness = pso_optimization(
        max_iterations, num_particles, parameter_bounds
    )

    print("Mejores parámetros encontrados: Kp={}, Ki={}, Kd={}".format(*best_params))
    print("Valor de fitness (MSE) asociado: {}".format(best_fitness))

    # Evaluar el desempeño del controlador PID con los parámetros ajustados
    setpoint = 1.0
    dt = 0.01
    system = MassSpringDamper(mass=1.0, spring_constant=1.0, damping_coefficient=0.1)
    controller = PIDController(Kp=best_params[0], Ki=best_params[1], Kd=best_params[2])

    time_points = np.arange(0, 10, dt)
    positions = []
    for time in time_points:
        feedback = system.position
        control_signal = controller.control(setpoint, feedback, dt)
        system.update(control_signal, dt)
        positions.append(system.position)

    # Graficar la respuesta del sistema con el controlador PID ajustado
    plt.plot(time_points, positions, label="Posición del sistema")
    plt.axhline(y=setpoint, color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Tiempo")
    plt.ylabel("Posición")
    plt.legend()
    plt.grid(True)
    plt.show()
