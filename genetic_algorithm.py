import random
import math
import copy
from typing import Any, Callable, List, Sequence, Tuple

# Problemas pequenos que continuam aqui só para manter a ideia do projeto original.
default_problems = {
    5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
    10: [(470, 169), (602, 202), (754, 239), (476, 233), (468, 301), (522, 29), (597, 171), (487, 325), (746, 232), (558, 136)],
    12: [(728, 67), (560, 160), (602, 312), (712, 148), (535, 340), (720, 354), (568, 300), (629, 260), (539, 46), (634, 343), (491, 135), (768, 161)],
    15: [(512, 317), (741, 72), (552, 50), (772, 346), (637, 12), (589, 131), (732, 165), (605, 15), (730, 38), (576, 216), (589, 381), (711, 387), (563, 228), (494, 22), (787, 288)]
}


def generate_random_population(items: Sequence[Any], population_size: int) -> List[List[Any]]:
    """Cria várias soluções aleatórias.

    Aqui cada indivíduo é só uma permutação dos itens.
    Isso funciona bem para TSP e também para rota de entregas.
    """
    items = list(items)
    return [random.sample(items, len(items)) for _ in range(population_size)]


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calcula a distância euclidiana entre 2 pontos.

    Deixei essa função porque ela já existia e ainda ajuda nos exemplos menores.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """Fitness simples do TSP clássico.

    Menor valor significa rota melhor.
    """
    if not path:
        return 0.0

    distance = 0.0
    for i in range(len(path)):
        distance += calculate_distance(path[i], path[(i + 1) % len(path)])
    return distance


def order_crossover(parent1: List[Any], parent2: List[Any]) -> List[Any]:
    """Crossover OX.

    Eu mantive esse operador porque ele é um dos mais usados para rota.
    Ele mistura os pais sem repetir cidade.
    """
    length = len(parent1)
    if length <= 1:
        return parent1.copy()

    start_index = random.randint(0, length - 2)
    end_index = random.randint(start_index + 1, length - 1)

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


def mutate(solution: List[Any], mutation_probability: float) -> List[Any]:
    """Mutação simples e didática.

    Em metade dos casos eu troco 2 posições.
    Na outra metade eu inverto um pedaço da rota.
    """
    mutated_solution = copy.deepcopy(solution)

    if len(solution) < 2:
        return mutated_solution

    if random.random() < mutation_probability:
        if random.random() < 0.5:
            i, j = random.sample(range(len(solution)), 2)
            mutated_solution[i], mutated_solution[j] = mutated_solution[j], mutated_solution[i]
        else:
            i, j = sorted(random.sample(range(len(solution)), 2))
            mutated_solution[i:j + 1] = list(reversed(mutated_solution[i:j + 1]))

    return mutated_solution


def tournament_selection(population: List[List[Any]], fitness: List[float], k: int = 5) -> List[Any]:
    """Escolhe 1 pai por torneio.

    Eu gosto dessa seleção porque é simples, estável e fácil de explicar.
    """
    k = max(2, min(k, len(population)))
    selected = random.sample(list(zip(population, fitness)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]


def sort_population(population: List[List[Any]], fitness: List[float]) -> Tuple[List[List[Any]], List[float]]:
    """Ordena da melhor solução para a pior."""
    combined_lists = list(zip(population, fitness))
    combined_lists.sort(key=lambda x: x[1])
    sorted_population = [item[0] for item in combined_lists]
    sorted_fitness = [item[1] for item in combined_lists]
    return sorted_population, sorted_fitness


def two_opt_improve(path: List[Any], fitness_function: Callable[[List[Any]], float] | None = None) -> List[Any]:
    """Pequena melhoria local usando 2-opt.

    Eu deixei isso aqui porque ajuda a dar uma refinada na rota
    sem complicar demais o projeto.
    """
    if len(path) < 4:
        return path.copy()

    evaluator = fitness_function if fitness_function is not None else calculate_fitness
    best_path = path.copy()
    best_distance = evaluator(best_path)

    for i in range(1, len(path) - 2):
        for j in range(i + 1, len(path) - 1):
            new_path = best_path[:i] + best_path[i:j + 1][::-1] + best_path[j + 1:]
            new_distance = evaluator(new_path)
            if new_distance < best_distance:
                best_distance = new_distance
                best_path = new_path

    return best_path


if __name__ == '__main__':
    random.seed(42)

    n_cities = 10
    population_size = 50
    n_generations = 30
    mutation_probability = 0.25

    cities_locations = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n_cities)]
    population = generate_random_population(cities_locations, population_size)

    best_fitness_values = []

    for generation in range(n_generations):
        population_fitness = [calculate_fitness(individual) for individual in population]
        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        print(f'Geração {generation + 1}: melhor fitness = {best_fitness:.2f}')

        new_population = [best_solution]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, population_fitness, k=5)
            parent2 = tournament_selection(population, population_fitness, k=5)
            child = order_crossover(parent1, parent2)
            child = mutate(child, mutation_probability)
            child = two_opt_improve(child)
            new_population.append(child)

        population = new_population