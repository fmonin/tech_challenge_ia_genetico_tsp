import itertools
import math
import random
import sys
from typing import Dict, List, Tuple

import pygame

from genetic_algorithm import (
    generate_random_population,
    mutate,
    order_crossover,
    sort_population,
    tournament_selection,
    two_opt_improve,
)
from draw_functions import (
    draw_cities,
    draw_paths,
    draw_plot,
    draw_route_summary,
    draw_text,
    draw_vehicle_legend,
    save_fitness_chart,
)

# ------------------------------
# Configurações gerais
# ------------------------------
WIDTH, HEIGHT = 1600, 900
MAP_LEFT = 20
MAP_TOP = 20
MAP_WIDTH = 980
MAP_HEIGHT = 860
RIGHT_PANEL_X = 1030
FPS = 30
POPULATION_SIZE = 90
N_GENERATIONS = 180
MUTATION_PROBABILITY = 0.22
ELITISM = 2
TOURNAMENT_SIZE = 6
NODE_RADIUS = 5
BACKGROUND = (14, 18, 28)
MAP_BACKGROUND = (24, 30, 45)
GRID_COLOR = (50, 58, 82)
GRAY_ROUTE = (150, 150, 150)
BLUE_ROUTE = (70, 150, 255)
WHITE = (240, 240, 240)
SOFT_TEXT = (205, 210, 220)
YELLOW = (240, 210, 90)

# Aqui eu corrigi um detalhe do enunciado.
# Foi pedido 90 cidades, mas a divisão 20 + 35 + 45 soma 100.
# Para manter as 90 cidades, usei 20 + 25 + 45.
ROUTE_SIZES = [20, 25, 45]

# Deixei seed para facilitar reproduzir a execução na apresentação.
random.seed(42)

# ------------------------------
# Base de cidades reais aproximadas
# ------------------------------
# Cada tupla tem:
# grupo da rota, nome da cidade, latitude, longitude, demanda e prioridade.
RAW_CITIES = [
    # Rota 1 - 20 cidades
    (1, 'São Paulo', -23.55, -46.63, 18, 'critica'),
    (1, 'Guarulhos', -23.45, -46.53, 12, 'regular'),
    (1, 'Osasco', -23.53, -46.79, 10, 'regular'),
    (1, 'Barueri', -23.51, -46.88, 9, 'regular'),
    (1, 'Carapicuíba', -23.52, -46.84, 8, 'regular'),
    (1, 'Jandira', -23.53, -46.90, 7, 'regular'),
    (1, 'Itapevi', -23.55, -46.93, 8, 'regular'),
    (1, 'Cotia', -23.60, -46.92, 12, 'critica'),
    (1, 'Taboão da Serra', -23.62, -46.79, 9, 'regular'),
    (1, 'Embu das Artes', -23.65, -46.85, 8, 'regular'),
    (1, 'Itapecerica da Serra', -23.72, -46.85, 11, 'critica'),
    (1, 'Santana de Parnaíba', -23.44, -46.92, 8, 'regular'),
    (1, 'Cajamar', -23.36, -46.88, 9, 'regular'),
    (1, 'Franco da Rocha', -23.32, -46.73, 10, 'regular'),
    (1, 'Caieiras', -23.36, -46.74, 7, 'regular'),
    (1, 'Francisco Morato', -23.28, -46.74, 8, 'regular'),
    (1, 'Mairiporã', -23.32, -46.59, 8, 'regular'),
    (1, 'Diadema', -23.69, -46.62, 10, 'regular'),
    (1, 'São Caetano do Sul', -23.62, -46.55, 9, 'regular'),
    (1, 'Santo André', -23.66, -46.53, 11, 'critica'),

    # Rota 2 - 25 cidades
    (2, 'São Bernardo do Campo', -23.69, -46.56, 14, 'critica'),
    (2, 'Mauá', -23.67, -46.46, 10, 'regular'),
    (2, 'Ribeirão Pires', -23.71, -46.41, 8, 'regular'),
    (2, 'Rio Grande da Serra', -23.74, -46.40, 7, 'regular'),
    (2, 'Suzano', -23.54, -46.31, 11, 'regular'),
    (2, 'Poá', -23.53, -46.35, 7, 'regular'),
    (2, 'Ferraz de Vasconcelos', -23.54, -46.37, 8, 'regular'),
    (2, 'Itaquaquecetuba', -23.49, -46.35, 10, 'regular'),
    (2, 'Arujá', -23.40, -46.32, 9, 'regular'),
    (2, 'Santa Isabel', -23.31, -46.22, 8, 'regular'),
    (2, 'Mogi das Cruzes', -23.52, -46.19, 13, 'critica'),
    (2, 'Biritiba-Mirim', -23.57, -46.04, 7, 'regular'),
    (2, 'Salesópolis', -23.53, -45.85, 7, 'regular'),
    (2, 'Guararema', -23.42, -46.04, 8, 'regular'),
    (2, 'Jacareí', -23.30, -45.97, 12, 'regular'),
    (2, 'São José dos Campos', -23.19, -45.88, 15, 'critica'),
    (2, 'Caçapava', -23.10, -45.71, 8, 'regular'),
    (2, 'Taubaté', -23.03, -45.56, 10, 'regular'),
    (2, 'Pindamonhangaba', -22.92, -45.46, 8, 'regular'),
    (2, 'Tremembé', -22.96, -45.55, 7, 'regular'),
    (2, 'Guaratinguetá', -22.82, -45.19, 11, 'critica'),
    (2, 'Aparecida', -22.85, -45.23, 9, 'regular'),
    (2, 'Lorena', -22.73, -45.12, 9, 'regular'),
    (2, 'Cruzeiro', -22.58, -44.96, 8, 'regular'),
    (2, 'Cachoeira Paulista', -22.67, -45.01, 7, 'regular'),

    # Rota 3 - 45 cidades
    (3, 'Campinas', -22.90, -47.06, 17, 'critica'),
    (3, 'Valinhos', -22.97, -46.99, 9, 'regular'),
    (3, 'Vinhedo', -23.03, -46.98, 8, 'regular'),
    (3, 'Louveira', -23.09, -46.95, 7, 'regular'),
    (3, 'Jundiaí', -23.19, -46.88, 12, 'regular'),
    (3, 'Várzea Paulista', -23.21, -46.83, 8, 'regular'),
    (3, 'Campo Limpo Paulista', -23.21, -46.78, 8, 'regular'),
    (3, 'Jarinu', -23.10, -46.73, 7, 'regular'),
    (3, 'Atibaia', -23.12, -46.55, 10, 'regular'),
    (3, 'Bom Jesus dos Perdões', -23.14, -46.46, 6, 'regular'),
    (3, 'Bragança Paulista', -22.95, -46.54, 9, 'regular'),
    (3, 'Joanópolis', -22.93, -46.27, 6, 'regular'),
    (3, 'Piracaia', -23.05, -46.36, 7, 'regular'),
    (3, 'Nazaré Paulista', -23.18, -46.40, 7, 'regular'),
    (3, 'Igaratá', -23.20, -46.16, 6, 'regular'),
    (3, 'Salto', -23.20, -47.29, 8, 'regular'),
    (3, 'Itu', -23.26, -47.30, 10, 'regular'),
    (3, 'Sorocaba', -23.50, -47.46, 15, 'critica'),
    (3, 'Votorantim', -23.54, -47.44, 8, 'regular'),
    (3, 'Mairinque', -23.55, -47.18, 7, 'regular'),
    (3, 'São Roque', -23.53, -47.14, 8, 'regular'),
    (3, 'Alumínio', -23.53, -47.25, 7, 'regular'),
    (3, 'Araçariguama', -23.44, -47.06, 7, 'regular'),
    (3, 'Boituva', -23.28, -47.67, 8, 'regular'),
    (3, 'Porto Feliz', -23.21, -47.52, 7, 'regular'),
    (3, 'Tietê', -23.11, -47.72, 7, 'regular'),
    (3, 'Indaiatuba', -23.09, -47.22, 10, 'regular'),
    (3, 'Hortolândia', -22.85, -47.21, 9, 'regular'),
    (3, 'Sumaré', -22.82, -47.27, 9, 'regular'),
    (3, 'Paulínia', -22.76, -47.15, 8, 'regular'),
    (3, 'Americana', -22.74, -47.33, 8, 'regular'),
    (3, 'Nova Odessa', -22.78, -47.29, 7, 'regular'),
    (3, "Santa Bárbara d'Oeste", -22.75, -47.41, 8, 'regular'),
    (3, 'Limeira', -22.56, -47.40, 10, 'regular'),
    (3, 'Piracicaba', -22.73, -47.65, 11, 'critica'),
    (3, 'Rio Claro', -22.41, -47.56, 9, 'regular'),
    (3, 'Santos', -23.96, -46.33, 14, 'critica'),
    (3, 'São Vicente', -23.96, -46.39, 8, 'regular'),
    (3, 'Praia Grande', -24.01, -46.41, 10, 'regular'),
    (3, 'Cubatão', -23.89, -46.42, 9, 'regular'),
    (3, 'Guarujá', -23.99, -46.26, 10, 'regular'),
    (3, 'Bertioga', -23.85, -46.14, 8, 'regular'),
    (3, 'Mongaguá', -24.10, -46.62, 7, 'regular'),
    (3, 'Itanhaém', -24.18, -46.79, 8, 'regular'),
    (3, 'Peruíbe', -24.31, -47.00, 8, 'regular'),
]

# Os 3 veículos disponíveis.
# Eu deixei nomes simples e números coerentes para a apresentação.
VEHICLES = [
    {
        'id': 1,
        'label': 'Veículo 1',
        'name': 'Pequeno',
        'capacity': 190,
        'max_distance': 280.0,
        'operational_cost': 1.00,
        'fixed_cost': 15.0,
        'critical_bonus': 8.0,
        'color': (89, 163, 255),
    },
    {
        'id': 2,
        'label': 'Veículo 2',
        'name': 'Médio',
        'capacity': 320,
        'max_distance': 520.0,
        'operational_cost': 1.12,
        'fixed_cost': 25.0,
        'critical_bonus': 14.0,
        'color': (70, 210, 170),
    },
    {
        'id': 3,
        'label': 'Veículo 3',
        'name': 'Grande',
        'capacity': 520,
        'max_distance': 900.0,
        'operational_cost': 1.28,
        'fixed_cost': 35.0,
        'critical_bonus': 18.0,
        'color': (255, 165, 80),
    },
]


# ------------------------------
# Funções utilitárias de geografia
# ------------------------------
def latlon_to_xy(
    lat: float,
    lon: float,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> Tuple[int, int]:
    """Transforma latitude e longitude em posição de tela.

    Não é mapa real com projeção completa.
    Aqui a ideia é só posicionar as cidades de forma coerente para a visualização.
    """
    usable_width = MAP_WIDTH - 40
    usable_height = MAP_HEIGHT - 40

    x_ratio = (lon - min_lon) / (max_lon - min_lon)
    y_ratio = (max_lat - lat) / (max_lat - min_lat)

    x = MAP_LEFT + 20 + int(x_ratio * usable_width)
    y = MAP_TOP + 20 + int(y_ratio * usable_height)
    return x, y


def haversine_km(city_a: Dict, city_b: Dict) -> float:
    """Calcula a distância aproximada em km entre 2 cidades."""
    radius = 6371.0
    lat1 = math.radians(city_a['lat'])
    lon1 = math.radians(city_a['lon'])
    lat2 = math.radians(city_b['lat'])
    lon2 = math.radians(city_b['lon'])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def route_distance_km(route: List[Dict], depot: Dict) -> float:
    """Calcula a distância total da rota saindo e voltando para o depósito."""
    if not route:
        return 0.0

    total = haversine_km(depot, route[0])
    for i in range(len(route) - 1):
        total += haversine_km(route[i], route[i + 1])
    total += haversine_km(route[-1], depot)
    return total


def build_city_objects() -> Tuple[List[Dict], Dict[int, List[Dict]], Dict]:
    """Monta os objetos das cidades e separa por grupo de rota."""
    min_lat = min(item[2] for item in RAW_CITIES)
    max_lat = max(item[2] for item in RAW_CITIES)
    min_lon = min(item[3] for item in RAW_CITIES)
    max_lon = max(item[3] for item in RAW_CITIES)

    depot = {
        'id': 0,
        'group': 0,
        'name': 'Centro de Distribuição - São Paulo',
        'lat': -23.55,
        'lon': -46.63,
        'demand': 0,
        'priority': 'depot',
        'screen_pos': latlon_to_xy(-23.55, -46.63, min_lat, max_lat, min_lon, max_lon),
    }

    all_cities = []
    groups = {1: [], 2: [], 3: []}

    for city_id, raw in enumerate(RAW_CITIES, start=1):
        group, name, lat, lon, demand, priority = raw
        city = {
            'id': city_id,
            'group': group,
            'name': name,
            'lat': lat,
            'lon': lon,
            'demand': demand,
            'priority': priority,
            'screen_pos': latlon_to_xy(lat, lon, min_lat, max_lat, min_lon, max_lon),
        }
        all_cities.append(city)
        groups[group].append(city)

    return all_cities, groups, depot


# ------------------------------
# Regras de negócio do VRP simplificado
# ------------------------------
def priority_position_penalty(route: List[Dict]) -> float:
    """Penaliza cidades críticas que ficaram muito tarde na rota.

    Quanto mais uma cidade crítica vai para o fim, maior a penalização.
    """
    penalty = 0.0
    reward = 0.0

    for index, city in enumerate(route):
        position_factor = index + 1
        if city['priority'] == 'critica':
            penalty += position_factor * 2.5
            reward += max(0, 20 - position_factor) * 1.2

    return penalty - reward


def route_demand(route: List[Dict]) -> int:
    """Soma a demanda total de uma rota."""
    return sum(city['demand'] for city in route)


def route_critical_count(route: List[Dict]) -> int:
    """Conta quantas cidades críticas existem na rota."""
    return sum(1 for city in route if city['priority'] == 'critica')


def evaluate_route_for_vehicle(route: List[Dict], vehicle: Dict, depot: Dict) -> Dict:
    """Avalia uma rota considerando um veículo específico."""
    distance_km = route_distance_km(route, depot)
    total_demand = route_demand(route)
    critical_count = route_critical_count(route)

    penalty = 0.0

    # Se a carga passar da capacidade, a penalidade cresce bastante.
    if total_demand > vehicle['capacity']:
        penalty += (total_demand - vehicle['capacity']) * 14.0

    # Se a distância passar da autonomia, também penaliza forte.
    if distance_km > vehicle['max_distance']:
        penalty += (distance_km - vehicle['max_distance']) * 10.0

    # Essa parte ajuda o AG a antecipar cidades críticas.
    priority_penalty = priority_position_penalty(route)

    # O custo operacional ajuda a diferenciar os veículos.
    operational_component = distance_km * vehicle['operational_cost'] + vehicle['fixed_cost']

    # Veículo maior ou mais preparado pode ter um ganho quando a rota tem mais criticidade.
    critical_discount = critical_count * vehicle['critical_bonus']

    # Fitness menor é melhor.
    fitness = distance_km + operational_component + priority_penalty + penalty - critical_discount

    return {
        'fitness': fitness,
        'distance_km': distance_km,
        'demand': total_demand,
        'critical_count': critical_count,
        'priority_penalty': priority_penalty,
        'penalty': penalty,
        'vehicle': vehicle,
    }


def choose_best_vehicle(route: List[Dict], available_vehicles: List[Dict], depot: Dict) -> Dict:
    """Escolhe o melhor veículo para uma rota olhando demanda, autonomia e criticidade."""
    evaluations = [evaluate_route_for_vehicle(route, vehicle, depot) for vehicle in available_vehicles]
    evaluations.sort(key=lambda item: item['fitness'])
    return evaluations[0]


def choose_best_vehicle_assignment(routes: List[List[Dict]], vehicles: List[Dict], depot: Dict) -> List[Dict]:
    """Testa todas as combinações possíveis entre 3 rotas e 3 veículos.

    Aqui eu garanto uma escolha mais inteligente sem repetir veículo.
    Como são só 3 rotas e 3 veículos, dá para fazer isso de forma simples.
    """
    best_assignment = None
    best_total = float('inf')

    for perm in itertools.permutations(vehicles, len(routes)):
        current_results = []
        total = 0.0

        for route, vehicle in zip(routes, perm):
            result = evaluate_route_for_vehicle(route, vehicle, depot)
            current_results.append(result)
            total += result['fitness']

        if total < best_total:
            best_total = total
            best_assignment = current_results

    return best_assignment


# ------------------------------
# Algoritmo genético por rota
# ------------------------------
def make_fitness_function(depot: Dict):
    """Cria uma função de fitness já presa ao depósito.

    Fiz assim para o two-opt conseguir reutilizar a mesma regra.
    """
    def evaluator(route: List[Dict]) -> float:
        return choose_best_vehicle(route, VEHICLES, depot)['fitness']

    return evaluator


def evolve_route_population(population: List[List[Dict]], depot: Dict):
    """Evolui uma população referente a uma única rota."""
    population_fitness = [choose_best_vehicle(individual, VEHICLES, depot)['fitness'] for individual in population]
    population, population_fitness = sort_population(population, population_fitness)

    best_route = population[0]
    best_result = choose_best_vehicle(best_route, VEHICLES, depot)

    new_population = [population[i] for i in range(min(ELITISM, len(population)))]
    route_fitness = make_fitness_function(depot)

    while len(new_population) < len(population):
        parent1 = tournament_selection(population, population_fitness, k=TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, population_fitness, k=TOURNAMENT_SIZE)

        # O crossover cria uma nova ordem de visita.
        child = order_crossover(parent1, parent2)

        # A mutação evita que a busca fique presa cedo demais.
        child = mutate(child, MUTATION_PROBABILITY)

        # Em parte dos filhos eu aplico 2-opt para dar uma refinada.
        if random.random() < 0.45:
            child = two_opt_improve(child, fitness_function=route_fitness)

        new_population.append(child)

    # Essa rota em teste é a que aparece em cinza na tela.
    testing_route = random.choice(population[:min(8, len(population))])
    testing_result = choose_best_vehicle(testing_route, VEHICLES, depot)

    return new_population, best_result, best_route, population_fitness, testing_route, testing_result


# ------------------------------
# Desenho da tela
# ------------------------------
def draw_map_background(screen: pygame.Surface, depot: Dict) -> None:
    """Desenha o fundo do mapa."""
    map_rect = pygame.Rect(MAP_LEFT, MAP_TOP, MAP_WIDTH, MAP_HEIGHT)
    pygame.draw.rect(screen, MAP_BACKGROUND, map_rect, border_radius=10)
    pygame.draw.rect(screen, (65, 82, 120), map_rect, 1, border_radius=10)

    for x in range(MAP_LEFT + 20, MAP_LEFT + MAP_WIDTH, 70):
        pygame.draw.line(screen, GRID_COLOR, (x, MAP_TOP + 10), (x, MAP_TOP + MAP_HEIGHT - 10), 1)
    for y in range(MAP_TOP + 20, MAP_TOP + MAP_HEIGHT, 70):
        pygame.draw.line(screen, GRID_COLOR, (MAP_LEFT + 10, y), (MAP_LEFT + MAP_WIDTH - 10, y), 1)

    draw_text(screen, 'Mapa das cidades da região de São Paulo', WHITE, (MAP_LEFT + 14, MAP_TOP + 10), font_size=20, bold=True)
    draw_text(screen, 'Cinza = rota em teste | Azul = melhor rota atual', SOFT_TEXT, (MAP_LEFT + 14, MAP_TOP + 36), font_size=14)

    pygame.draw.circle(screen, (255, 120, 120), depot['screen_pos'], 8)
    draw_text(screen, 'CD', (255, 255, 255), (depot['screen_pos'][0] + 8, depot['screen_pos'][1] - 8), font_size=12, bold=True)


def route_to_screen_points(route: List[Dict], depot: Dict) -> List[Tuple[int, int]]:
    """Converte uma rota para pontos da tela."""
    if not route:
        return []
    return [depot['screen_pos']] + [city['screen_pos'] for city in route] + [depot['screen_pos']]


# ------------------------------
# Programa principal
# ------------------------------
def main() -> None:
    all_cities, city_groups, depot = build_city_objects()

    # Só um check simples para garantir que a base bate com o enunciado ajustado.
    assert len(all_cities) == 90, 'A base precisa ter exatamente 90 cidades.'
    assert [len(city_groups[i]) for i in [1, 2, 3]] == ROUTE_SIZES, 'As rotas precisam respeitar 20, 25 e 45 cidades.'

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('VRP simplificado com Algoritmo Genético')
    clock = pygame.time.Clock()

    # Cada rota tem sua própria população.
    # Isso deixa a solução mais fácil de explicar na pós.
    populations = {
        1: generate_random_population(city_groups[1], POPULATION_SIZE),
        2: generate_random_population(city_groups[2], POPULATION_SIZE),
        3: generate_random_population(city_groups[3], POPULATION_SIZE),
    }

    best_global_fitness_history = []

    # Aqui eu guardo separadamente o histórico do fitness de cada veículo.
    # Isso permite mostrar no gráfico qual rota ficou com qual veículo em cada geração.
    vehicle_fitness_history = {
        1: [],
        2: [],
        3: [],
    }

    running = True
    paused = False
    generation = 0
    final_results = None

    print('Execução iniciada...')
    print('Controles: Q = sair | P = pausar/continuar')
    print('-' * 100)

    while running and generation < N_GENERATIONS:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused

        if paused:
            pygame.display.flip()
            clock.tick(FPS)
            continue

        generation += 1
        screen.fill(BACKGROUND)
        draw_map_background(screen, depot)

        best_routes = []
        testing_routes = []

        # Eu evoluo cada rota separadamente para deixar o trabalho mais didático.
        # Isso não é o VRP completo mais avançado, mas já mostra bem a ideia.
        for route_id in [1, 2, 3]:
            new_population, best_result, best_route, _, testing_route, _ = evolve_route_population(populations[route_id], depot)
            populations[route_id] = new_population
            best_routes.append(best_route)
            testing_routes.append(testing_route)

        # Depois que cada rota foi otimizada, eu faço a melhor combinação dos 3 veículos.
        assignment_results = choose_best_vehicle_assignment(best_routes, VEHICLES, depot)

        # Atualizo os resultados finais com as rotas e os veículos já casados.
        merged_results = []
        global_fitness = 0.0

        for route_index, assignment in enumerate(assignment_results):
            merged = {
                **assignment,
                'best_route': best_routes[route_index],
                'testing_route': testing_routes[route_index],
            }
            merged_results.append(merged)
            global_fitness += merged['fitness']

        best_global_fitness_history.append(global_fitness)
        final_results = merged_results

        # Aqui eu salvo qual foi o fitness da rota que ficou com cada veículo nessa geração.
        for result in merged_results:
            vehicle_id = result['vehicle']['id']
            vehicle_fitness_history[vehicle_id].append(result['fitness'])

        # Primeiro eu desenho as rotas em teste em cinza.
        for result in merged_results:
            test_points = route_to_screen_points(result['testing_route'], depot)
            draw_paths(screen, test_points, GRAY_ROUTE, width=2, close_path=False)

        # Depois eu desenho a melhor solução da geração.
        for result in merged_results:
            best_points = route_to_screen_points(result['best_route'], depot)
            draw_paths(screen, best_points, BLUE_ROUTE, width=4, close_path=False)

        draw_cities(screen, all_cities, YELLOW, NODE_RADIUS, show_labels=False)

        draw_text(screen, f'Geração: {generation}/{N_GENERATIONS}', WHITE, (RIGHT_PANEL_X, 20), font_size=22, bold=True)
        draw_text(screen, f'Fitness global: {global_fitness:.2f}', WHITE, (RIGHT_PANEL_X, 52), font_size=18)
        draw_text(screen, f'Mutação: {MUTATION_PROBABILITY:.2f}', SOFT_TEXT, (RIGHT_PANEL_X, 78), font_size=16)

        draw_vehicle_legend(screen, VEHICLES, (RIGHT_PANEL_X, 110))
        draw_route_summary(screen, merged_results, (RIGHT_PANEL_X, 275))

        plot_x = list(range(1, len(best_global_fitness_history) + 1))
        plot_series = [
            {
                'label': 'Fitness global',
                'values': best_global_fitness_history,
                'color': (255, 255, 255),
            }
        ]

        for vehicle in VEHICLES:
            plot_series.append(
                {
                    'label': f"{vehicle['label']} - {vehicle['name']}",
                    'values': vehicle_fitness_history[vehicle['id']],
                    'color': vehicle['color'],
                }
            )

        draw_plot(
            screen,
            plot_x,
            plot_series,
            x_label='Geração',
            y_label='Fitness',
            position=(RIGHT_PANEL_X, 470),
        )

        print(f'Geração {generation:03d} | Fitness global = {global_fitness:10.2f}')
        for idx, result in enumerate(merged_results, start=1):
            vehicle = result['vehicle']
            print(
                f"  Rota {idx} -> {vehicle['label']} ({vehicle['name']}) | "
                f"cidades={len(result['best_route'])} | demanda={result['demand']} | "
                f"distância={result['distance_km']:.1f} km | penalidade={result['penalty']:.1f} | "
                f"fitness={result['fitness']:.2f}"
            )
        print('-' * 100)

        pygame.display.flip()
        clock.tick(FPS)

    # No final eu salvo o gráfico para você poder usar na entrega.
    if best_global_fitness_history:
        final_series = [
            {
                'label': 'Fitness global',
                'x': list(range(1, len(best_global_fitness_history) + 1)),
                'values': best_global_fitness_history,
                'color': (255, 255, 255),
            }
        ]

        for vehicle in VEHICLES:
            final_series.append(
                {
                    'label': f"{vehicle['label']} - {vehicle['name']}",
                    'x': list(range(1, len(vehicle_fitness_history[vehicle['id']]) + 1)),
                    'values': vehicle_fitness_history[vehicle['id']],
                    'color': vehicle['color'],
                }
            )

        save_fitness_chart(
            final_series,
            output_path='fitness_evolution.png',
        )

    if final_results:
        print('\nResumo final da melhor solução encontrada nessa execução:')
        for idx, result in enumerate(final_results, start=1):
            vehicle = result['vehicle']
            city_names = ', '.join(city['name'] for city in result['best_route'][:6])
            print(
                f"Rota {idx}: {vehicle['label']} - {vehicle['name']} | "
                f"distância={result['distance_km']:.1f} km | demanda={result['demand']} | "
                f"primeiras cidades={city_names}..."
            )
        print('Gráfico final salvo em: fitness_evolution.png')

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
