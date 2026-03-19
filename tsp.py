import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, default_problems, two_opt_improve, tournament_selection
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
import pygame
from benchmark_att48 import *


# Definir valores constantes
# Configurações da tela do Pygame
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

# Configurações do algoritmo genético
N_CITIES = 20
POPULATION_SIZE = 200
N_GENERATIONS = None
MUTATION_PROBABILITY = 0.25

# Definição de cores RGB usadas na visualização
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# Inicializar problema de TSP
# Geração aleatória de cidades (comentada atualmente)
cities_locations = [(random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS), random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
                    for _ in range(N_CITIES)]

# Usar problema fixo predefinido (5 cidades) para testes e visualização rápida
# WIDTH, HEIGHT = 800, 400
# cities_locations = default_problems[15]


# Opção de benchmark att48 (deixe comentado se usar problema predefinido)
# WIDTH, HEIGHT = 1500, 800
# att_cities_locations = np.array(att_48_cities_locations)
# max_x = max(point[0] for point in att_cities_locations)
# max_y = max(point[1] for point in att_cities_locations)
# scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
# scale_y = HEIGHT / max_y
# cities_locations = [(int(point[0] * scale_x + PLOT_X_OFFSET),
#                      int(point[1] * scale_y)) for point in att_cities_locations]
# target_solution = [cities_locations[i-1] for i in att_48_cities_order]
# fitness_target_solution = calculate_fitness(target_solution)
# print(f"Melhor solução de referência: {fitness_target_solution}")
# ----- Benchmark att48


# Inicializar Pygame e janela do jogo
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver usando Pygame")
clock = pygame.time.Clock()
# Contador de gerações usado para imprimir e mostrar evolução
generation_counter = itertools.count(start=1)

# Criar população inicial aleatória
# TODO: melhorar inicialização usando heurística (e.g. vizinho mais próximo, convex hull)
population = generate_random_population(cities_locations, POPULATION_SIZE)
best_fitness_values = []
best_solutions = []


# Main game loop
running = True
paused = False
step_once = False

print("Controles: Q=fechar, P=pausar/continuar, Espaço=passo, +/= aumentar mutação, -/_ diminuir mutação")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_p:
                paused = not paused
                print(f"Pausado: {paused}")
            elif event.key == pygame.K_SPACE:
                if paused:
                    step_once = True
                else:
                    paused = True
                    step_once = True
                print("Passo manual: próxima geração")
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                MUTATION_PROBABILITY = min(1.0, MUTATION_PROBABILITY + 0.05)
                print(f"Mutação: {MUTATION_PROBABILITY:.2f}")
            elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                MUTATION_PROBABILITY = max(0.0, MUTATION_PROBABILITY - 0.05)
                print(f"Mutação: {MUTATION_PROBABILITY:.2f}")

    if paused and not step_once:
        # apenas processa eventos enquanto pausado
        pygame.display.flip()
        clock.tick(FPS)
        continue

    if step_once:
        step_once = False

    generation = next(generation_counter)

    screen.fill(WHITE)

    population_fitness = [calculate_fitness(individual) for individual in population]
    population, population_fitness = sort_population(population, population_fitness)

    best_fitness = calculate_fitness(population[0])
    best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distância (pxls)")

    # Desenhar cidades e caminhos no Pygame
    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    if len(population) > 1:
        draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

    print(f"Geração {generation}: Melhor fitness = {round(best_fitness, 2)}")

    # Elitismo: mantém o melhor da geração anterior na nova população
    new_population = [population[0]]

    # Gera novos filhos até completar a população
    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population, population_fitness, k=7)
        parent2 = tournament_selection(population, population_fitness, k=7)
        child1 = order_crossover(parent1, parent2)
        child1 = mutate(child1, MUTATION_PROBABILITY)
        child1 = two_opt_improve(child1)
        new_population.append(child1)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)

# TODO: save the best individual in a file if it is better than the one saved.

# finaliza o jogo e encerra
pygame.quit()
sys.exit()
