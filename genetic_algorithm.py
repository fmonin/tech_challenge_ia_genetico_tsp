

import random
import math
import copy 
from typing import List, Tuple

# Problemas de referência com coordenadas de cidades (casos de teste fixos)
default_problems = {
    5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
    10: [(470, 169), (602, 202), (754, 239), (476, 233), (468, 301), (522, 29), (597, 171), (487, 325), (746, 232), (558, 136)],
    12: [(728, 67), (560, 160), (602, 312), (712, 148), (535, 340), (720, 354), (568, 300), (629, 260), (539, 46), (634, 343), (491, 135), (768, 161)],
    15: [(512, 317), (741, 72), (552, 50), (772, 346), (637, 12), (589, 131), (732, 165), (605, 15), (730, 38), (576, 216), (589, 381), (711, 387), (563, 228), (494, 22), (787, 288)]
}

def generate_random_population(cities_location: List[Tuple[float, float]], population_size: int) -> List[List[Tuple[float, float]]]:
    """Gera uma população inicial de rotas aleatórias.

    Cada indivíduo é uma permutação completa das cidades (cada cidade aparece apenas uma vez).
    """
    return [random.sample(cities_location, len(cities_location)) for _ in range(population_size)]


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calcula a distância Euclidiana entre dois pontos 2D."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """Calcula a fitness de um caminho como a soma das distâncias de viagem.

    A fitness é menor para soluções melhores (menor distância total).
    """
    distance = 0
    n = len(path)
    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])
    return distance


def order_crossover(parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Realiza crossover do tipo Order Crossover (OX) para rotas TSP.

    Preserva a propriedade de permutação (sem cidades repetidas).

    1) Seleciona um segmento contíguo de parent1.
    2) Copia esse segmento para o filho.
    3) Preenche as posições restantes com as cidades de parent2 na ordem,
       ignorando cidades já copiadas.
    """
    length = len(parent1)
    if length <= 1:
        return parent1.copy()

    # Escolhe dois pontos de corte para o crossover
    start_index = random.randint(0, length - 2)
    end_index = random.randint(start_index + 1, length - 1)

    # Inicializa o filho com valores vazios
    child = [None] * length
    child[start_index:end_index + 1] = parent1[start_index:end_index + 1]

    parent2_index = 0
    for i in range(length):
        if child[i] is None:
            while parent2[parent2_index] in child:
                parent2_index += 1
            child[i] = parent2[parent2_index]
            parent2_index += 1

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



# TODO: implementar intensidade de mutação e inversão de blocos (2-opt late) para evolução mais robusta.
def two_opt_improve(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Aplica um passo de melhoria 2-opt para refinar um caminho.

    Testa trocas de arestas (i,j) e aplica a melhor inversão local.
    """
    n = len(path)
    best_distance = calculate_fitness(path)
    best_path = path

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
            new_distance = calculate_fitness(new_path)
            if new_distance < best_distance:
                best_distance = new_distance
                best_path = new_path
    return best_path


def mutate(solution:  List[Tuple[float, float]], mutation_probability: float) ->  List[Tuple[float, float]]:
    """Mutação com swap/inversão para melhorar exploração do espaço de soluções.

    Com probabilidade de mutação:
    - 50% troca duas cidades aleatórias (swap)
    - 50% inverte um segmento contínuo (inversion)
    """
    mutated_solution = copy.deepcopy(solution)
    if len(solution) < 2:
        return solution

    if random.random() < mutation_probability:
        if random.random() < 0.5:
            i, j = random.sample(range(len(solution)), 2)
            mutated_solution[i], mutated_solution[j] = mutated_solution[j], mutated_solution[i]
        else:
            i, j = sorted(random.sample(range(len(solution)), 2))
            mutated_solution[i:j+1] = list(reversed(mutated_solution[i:j+1]))

    return mutated_solution


def tournament_selection(population: List[List[Tuple[float, float]]], fitness: List[float], k: int = 5) -> List[Tuple[float, float]]:
    """Seleciona um pai por torneio de tamanho k.

    Retorna o melhor entre k indivíduos escolhidos aleatoriamente.
    """
    selected = random.sample(list(zip(population, fitness)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

### Demonstração: código de teste da função de mutação  
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(population: List[List[Tuple[float, float]]], fitness: List[float]) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """Ordena a população pelo valor de fitness (menor é melhor)."""
    combined_lists = list(zip(population, fitness))
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)
    return sorted_population, sorted_fitness


if __name__ == '__main__':
    N_CITIES = 10
    
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [(random.randint(0, 100), random.randint(0, 100))
              for _ in range(N_CITIES)]
    
    # CRIA POPULAÇÃO INICIAL
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Listas para armazenar os melhores fitness por geração
    best_fitness_values = []
    best_solutions = []
    
    for generation in range(N_GENERATIONS):
        population_fitness = [calculate_fitness(individual) for individual in population]
        population, population_fitness = sort_population(population,  population_fitness)

        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        print(f"Geração {generation}: Melhor fitness = {best_fitness}")

        # ELITISMO: mantém o melhor indivíduo para a próxima geração
        new_population = [population[0]]
        
        while len(new_population) < POPULATION_SIZE:
            # SELEÇÃO de pais via amostragem simples do top 10
            parent1, parent2 = random.choices(population[:10], k=2)
            # CROSSOVER (OX) preserva permutação
            child1 = order_crossover(parent1, parent2)
            # MUTAÇÃO com swap/inversão
            child1 = mutate(child1, MUTATION_PROBABILITY)
            new_population.append(child1)

        print('geração: ', generation)
        population = new_population
    


