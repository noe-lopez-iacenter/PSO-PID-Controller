import numpy as np
import matplotlib.pyplot as plt


# Definición del sistema masa-resorte-amortiguador (igual que en el script anterior)
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


# Controlador PID (igual que en el script anterior)
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


# Función de evaluación (fitness) para el Algoritmo Genético
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


# Operadores genéticos
def crossover(parent1, parent2):
    # Cruza de dos controladores PID
    child = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]
    return child


def mutate(individual, mutation_rate):
    # Mutación de un controlador PID
    mutated_individual = [
        (param + np.random.normal(0, 0.1))
        if np.random.rand() < mutation_rate
        else param
        for param in individual
    ]
    return mutated_individual


# Implementación del Algoritmo Genético
def genetic_algorithm(num_generations, population_size, mutation_rate, bounds):
    num_dimensions = len(bounds)
    population = [
        np.random.uniform(bounds[d][0], bounds[d][1], num_dimensions)
        for d in range(num_dimensions)
    ]

    for generation in range(num_generations):
        # Evaluación del fitness para cada individuo en la población
        fitness_values = [evaluate_fitness(*individual) for individual in population]

        # Selección basada en el fitness (elitismo: seleccionar los 2 mejores individuos directamente)
        sorted_indices = np.argsort(fitness_values)
        selected_parents = [population[i] for i in sorted_indices[:2]]

        # Crear una nueva generación mediante cruza y mutación
        new_population = [crossover(selected_parents[0], selected_parents[1])]

        for _ in range(population_size - 1):
            mutated_child = mutate(new_population[-1], mutation_rate)
            new_population.append(mutated_child)

        population = new_population

    # Seleccionar el mejor individuo de la última generación
    best_idx = np.argmin([evaluate_fitness(*individual) for individual in population])
    best_params = population[best_idx]
    best_fitness = evaluate_fitness(*best_params)

    return best_params, best_fitness


if __name__ == "__main__":
    # Parámetros para el Algoritmo Genético
    num_generations = 100
    population_size = 30
    mutation_rate = 0.3
    parameter_bounds = [
        (0.1, 10.0),
        (0.01, 5.0),
        (0.01, 2.0),
    ]  # Rango permitido para Kp, Ki y Kd

    # Ejecutar el Algoritmo Genético para ajustar los parámetros del controlador PID
    best_params, best_fitness = genetic_algorithm(
        num_generations, population_size, mutation_rate, parameter_bounds
    )

    print("Mejores parámetros encontrados: Kp={}, Ki={}, Kd={}".format(*best_params))
    print("Valor de fitness (MSE) asociado: {}".format(best_fitness))

    # Evaluar el desempeño del controlador PID con los parámetros ajustados (igual que en el script anterior)
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

    # Graficar la respuesta del sistema con el controlador PID ajustado (igual que en el script anterior)
    plt.plot(time_points, positions, label="Posición del sistema")
    plt.axhline(y=setpoint, color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Tiempo")
    plt.ylabel("Posición")
    plt.legend()
    plt.grid(True)
    plt.show()
