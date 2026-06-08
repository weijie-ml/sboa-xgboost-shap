import numpy as np
from scipy.special import gamma

# Secretary Bird Optimization Algorithm (SBOA)

def sboa(pop_size, max_iter, lower_bounds, upper_bounds, fitness_func, print_log=True):

    dim = len(lower_bounds)
    lower_bounds = np.array(lower_bounds, np.float32)
    upper_bounds = np.array(upper_bounds, np.float32)

    # Initialization
    positions = np.random.rand(pop_size, dim) * (upper_bounds - lower_bounds) + lower_bounds
    fitness_values = [fitness_func(*position) for position in positions]
    
    
    # Main loop
    best_position = None
    optim_curve = np.zeros(max_iter)
    
    for iter_num in range(max_iter):
        # CF decreases with iteration progress
        cf = (1 - iter_num / max_iter) ** (2 * iter_num / max_iter)
        
        # Update the global best
        current_best_score = np.min(fitness_values)
        best_location = np.argmin(fitness_values)
        if iter_num == 0:
            best_position = positions[best_location, :]
            fbest = current_best_score
        elif current_best_score < fbest:
            fbest = current_best_score
            best_position = positions[best_location, :]

        # Secretary bird's predation strategy
        for i in range(pop_size):
            if iter_num < max_iter / 3:  # Search prey stage
                num_birds = positions.shape[0]
                random_bird_1 = np.random.randint(0, num_birds)
                random_bird_2 = np.random.randint(0, num_birds)
                rand_factor_1 = np.random.rand(dim)
                new_position = positions[i, :] + (positions[random_bird_1, :] - positions[random_bird_2, :]) * rand_factor_1
                new_position = np.clip(new_position, lower_bounds, upper_bounds)
            elif iter_num > max_iter / 3 and iter_num < 2 * max_iter / 3:  # Approaching prey stage
                rand_bird = np.random.randn(dim)
                new_position = best_position + (iter_num / max_iter) ** 4 * (rand_bird - 0.5) * (best_position - positions[i, :])
                new_position = np.clip(new_position, lower_bounds, upper_bounds)
            else:  # Attack prey stage
                levy_step = 0.5 * levy(dim)
                new_position = best_position + cf * positions[i, :] * levy_step
                new_position = np.clip(new_position, lower_bounds, upper_bounds)

            new_fitness = fitness_func(*new_position)
            if new_fitness <= fitness_values[i]:
                positions[i, :] = new_position
                fitness_values[i] = new_fitness

        # Secretary Bird's escape strategy
        rand_value = np.random.rand()
        random_bird_index = np.random.randint(0, pop_size)
        random_position = positions[random_bird_index, :]
        for i in range(pop_size):
            if rand_value < 0.5:
                # C1: Secretary birds use their environment to hide from predators
                rand_bird = np.random.rand(dim)
                new_position = best_position + (1 - iter_num / max_iter) ** 2 * (2 * rand_bird - 1) * positions[i, :]
                new_position = np.clip(new_position, lower_bounds, upper_bounds)
            else:
                # C2: Secretary birds fly or run away from the predator
                random_factor = np.random.randint(1, 2)
                rand_factor_2 = np.random.rand(dim)
                new_position = positions[i, :] + rand_factor_2 * (random_position - random_factor * positions[i, :])
                new_position = np.clip(new_position, lower_bounds, upper_bounds)

            new_fitness = fitness_func(*new_position)
            if new_fitness <= fitness_values[i]:
                positions[i, :] = new_position
                fitness_values[i] = new_fitness

        if print_log:
            print(f"[ SBOA Optimization ] Iteration: {iter_num+1:>4}/{max_iter:<4}, Fitness: {fbest:>8.4f}")

        optim_curve[iter_num] = fbest

    return best_position, optim_curve


# Levy flight function
def levy(dim):
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / np.abs(v) ** (1 / beta)
    return step



