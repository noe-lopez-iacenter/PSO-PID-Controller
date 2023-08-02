import numpy as np
import matplotlib.pyplot as plt


# Definición del sistema masa-resorte-amortiguador
class MassSpringDamper:
    """System mass-spring-damper"""

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
    """PID Controller to stabilize system"""

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


# Implementación de la comparación entre PSO y Algoritmo Genético
def compare_pso_vs_genetic(
    pso_max_iterations,
    pso_num_particles,
    genetic_num_generations,
    genetic_population_size,
    genetic_mutation_rate,
    bounds,
):
    # Ejecutar el algoritmo PSO para ajustar los parámetros del controlador PID
    pso_best_params, pso_best_fitness = pso_optimization(
        pso_max_iterations, pso_num_particles, bounds
    )

    # Ejecutar el algoritmo genético para ajustar los parámetros del controlador PID
    genetic_best_params, genetic_best_fitness = genetic_algorithm(
        genetic_num_generations, genetic_population_size, genetic_mutation_rate, bounds
    )

    # Comparar el desempeño del sistema con los parámetros ajustados por ambos algoritmos
    setpoint = 1.0
    dt = 0.01
    system = MassSpringDamper(mass=1.0, spring_constant=1.0, damping_coefficient=0.1)
    pso_controller = PIDController(
        Kp=pso_best_params[0], Ki=pso_best_params[1], Kd=pso_best_params[2]
    )
    genetic_controller = PIDController(
        Kp=genetic_best_params[0], Ki=genetic_best_params[1], Kd=genetic_best_params[2]
    )

    time_points = np.arange(0, 10, dt)
    pso_positions = []
    genetic_positions = []

    for time in time_points:
        feedback = system.position

        # Controlador PID ajustado por PSO
        pso_control_signal = pso_controller.control(setpoint, feedback, dt)
        system.update(pso_control_signal, dt)
        pso_positions.append(system.position)

        # Controlador PID ajustado por Algoritmo Genético
        genetic_control_signal = genetic_controller.control(setpoint, feedback, dt)
        system.update(genetic_control_signal, dt)
        genetic_positions.append(system.position)

    # Mostrar los resultados de desempeño de ambos algoritmos
    print("Resultados de PSO:")
    print(
        "Mejores parámetros encontrados: Kp={}, Ki={}, Kd={}".format(*pso_best_params)
    )
    print("Valor de fitness (MSE) asociado: {}".format(pso_best_fitness))

    print("\nResultados del Algoritmo Genético:")
    print(
        "Mejores parámetros encontrados: Kp={}, Ki={}, Kd={}".format(
            *genetic_best_params
        )
    )
    print("Valor de fitness (MSE) asociado: {}".format(genetic_best_fitness))

    # Graficar la respuesta del sistema con el controlador PID ajustado por PSO
    plt.subplot(2, 1, 1)
    plt.plot(time_points, pso_positions, label="PSO Controlador PID")
    plt.axhline(y=setpoint, color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Tiempo")
    plt.ylabel("Posición")
    plt.legend()
    plt.grid(True)
    plt.title("Respuesta del sistema con controlador PID ajustado por PSO")

    # Graficar la respuesta del sistema con el controlador PID ajustado por el Algoritmo Genético
    plt.subplot(2, 1, 2)
    plt.plot(time_points, genetic_positions, label="Algoritmo Genético Controlador PID")
    plt.axhline(y=setpoint, color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Tiempo")
    plt.ylabel("Posición")
    plt.legend()
    plt.grid(True)
    plt.title(
        "Respuesta del sistema con controlador PID ajustado por Algoritmo Genetico"
    )

    # Ajustar el diseño de las gráficas para evitar superposiciones
    plt.tight_layout()

    plt.show()

    """# Graficar la respuesta del sistema con ambos controladores PID ajustados
    plt.plot(time_points, pso_positions, label="PSO Controlador PID")
    plt.plot(time_points, genetic_positions, label="Algoritmo Genético Controlador PID")
    plt.axhline(y=setpoint, color="r", linestyle="--", label="Setpoint")
    plt.xlabel("Tiempo")
    plt.ylabel("Posición")
    plt.legend()
    plt.grid(True)
    plt.title("Comparación PSO vs. Algoritmo Genético")
    plt.show()"""


if __name__ == "__main__":
    # Parámetros para los algoritmos PSO y Genético
    pso_max_iterations = 100
    pso_num_particles = 30
    genetic_num_generations = 100
    genetic_population_size = 30
    genetic_mutation_rate = 0.3
    parameter_bounds = [
        (0.1, 10.0),
        (0.01, 5.0),
        (0.01, 2.0),
    ]  # Rango permitido para Kp, Ki y Kd

    # Comparar el desempeño de PSO y Algoritmo Genético
    compare_pso_vs_genetic(
        pso_max_iterations,
        pso_num_particles,
        genetic_num_generations,
        genetic_population_size,
        genetic_mutation_rate,
        parameter_bounds,
    )
