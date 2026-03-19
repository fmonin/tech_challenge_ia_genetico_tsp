

import random
import math
import copy 
from typing import List, Tuple

default_problems = {
5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
10:[(470, 169), (602, 202), (754, 239), (476, 233), (468, 301), (522, 29), (597, 171), (487, 325), (746, 232), (558, 136)],
12:[(728, 67), (560, 160), (602, 312), (712, 148), (535, 340), (720, 354), (568, 300), (629, 260), (539, 46), (634, 343), (491, 135), (768, 161)],
15:[(512, 317), (741, 72), (552, 50), (772, 346), (637, 12), (589, 131), (732, 165), (605, 15), (730, 38), (576, 216), (589, 381), (711, 387), (563, 228), (494, 22), (787, 288)]
}

def generate_random_population(cities_location: List[Tuple[float, float]], population_size: int) -> List[List[Tuple[float, float]]]:
    """
    Gera uma população aleatória de rotas para um conjunto de cidades.

    Parâmetros:
    - cities_location (List[Tuple[float, float]]): Lista de coordenadas das cidades.
    - population_size (int): Tamanho da população.

    Retorna:
    List[List[Tuple[float, float]]]: Lista de rotas, cada rota é uma lista de coordenadas de cidades.
    """
    return [random.sample(cities_location, len(cities_location)) for _ in range(population_size)]


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calcula a distância Euclidiana entre dois pontos.

    Parâmetros:
    - point1 (Tuple[float, float]): Coordenadas do primeiro ponto.
    - point2 (Tuple[float, float]): Coordenadas do segundo ponto.

    Retorna:
    float: Distância Euclidiana entre os pontos.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """
    Calcula a aptidão de um caminho com base na distância Euclidiana total.

    Parâmetros:
    - path (List[Tuple[float, float]]): Lista de coordenadas do caminho.

    Retorna:
    float: Distância total do caminho.
    """
    distance = 0
    n = len(path)
    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])

    return distance


def order_crossover(parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Realiza crossover de ordem (OX) entre duas sequências de pais para gerar um filho.

    Parâmetros:
    - parent1 (List[Tuple[float, float]]): Primeiro pai.
    - parent2 (List[Tuple[float, float]]): Segundo pai.

    Retorna:
    List[Tuple[float, float]]: Filho gerado pelo crossover.
    """
    length = len(parent1)

    # Escolhe dois índices aleatórios para o crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Inicializa o filho com uma cópia do segmento de parent1
    child = parent1[start_index:end_index]

    # Preenche as posições restantes com genes de parent2
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child

### demonstration: crossover test code
# Example usage:
# parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
# parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# # parent1 = [1, 2, 3, 4, 5, 6]
# # parent2 = [6, 5, 4, 3, 2, 1]


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]



# TODO: implementar intensidade de mutação e inverter segmentos em vez de apenas trocar dois vizinhos.
def mutate(solution:  List[Tuple[float, float]], mutation_probability: float) ->  List[Tuple[float, float]]:
    """
    Mutação de uma solução invertendo um segmento da sequência com probabilidade definida.

    Parâmetros:
    - solution (List[int]): Sequência a ser mutada.
    - mutation_probability (float): Probabilidade de mutação.

    Retorna:
    List[int]: Solução mutada.
    """
    mutated_solution = copy.deepcopy(solution)

    # Verifica se deve aplicar mutação
    if random.random() < mutation_probability:
        
        # Garante ao menos duas cidades para trocar
        if len(solution) < 2:
            return solution
    
        # Seleciona índice aleatório (excluindo o último) para troca
        index = random.randint(0, len(solution) - 2)
        
        # Troca duas cidades vizinhas
        mutated_solution[index], mutated_solution[index + 1] = solution[index + 1], solution[index]   
        
    return mutated_solution

### Demonstration: mutation test code    
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(population: List[List[Tuple[float, float]]], fitness: List[float]) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (List[List[Tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (List[float]): The corresponding fitness values for each solution in the population.

    Returns:
    Tuple[List[List[Tuple[float, float]]], List[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combina população e aptidão em pares
    combined_lists = list(zip(population, fitness))

    # Ordena pelo valor de aptidão
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separa novamente em listas ordenadas
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return sorted_population, sorted_fitness


if __name__ == '__main__':
    N_CITIES = 10
    
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [(random.randint(0, 100), random.randint(0, 100))
              for _ in range(N_CITIES)]
    
    # CREATE INITIAL POPULATION
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Lists to store best fitness and generation for plotting
    best_fitness_values = []
    best_solutions = []
    
    for generation in range(N_GENERATIONS):
  
        
        population_fitness = [calculate_fitness(individual) for individual in population]    
        
        population, population_fitness = sort_population(population,  population_fitness)
        
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
           
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)    

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        new_population = [population[0]]  # Keep the best individual: ELITISM
        
        while len(new_population) < POPULATION_SIZE:
            
            # SELECTION
            parent1, parent2 = random.choices(population[:10], k=2)  # Select parents from the top 10 individuals
            
            # CROSSOVER
            child1 = order_crossover(parent1, parent2)
            
            ## MUTATION
            child1 = mutate(child1, MUTATION_PROBABILITY)
            
            new_population.append(child1)
            
    
        print('generation: ', generation)
        population = new_population
    


