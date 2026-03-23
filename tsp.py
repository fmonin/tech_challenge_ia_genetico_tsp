"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    SISTEMA DE OTIMIZAÇÃO DE ROTAS (TSP/VRP)                   ║
║               Algoritmo Genético com Integração LLM - Relatórios              ║
║                                                                               ║
║  PROPÓSITO: Resolver o Vehicle Routing Problem (VRP) para 20 cidades de       ║
║  São Paulo usando evolução genética, otimizando múltiplas métricas:           ║
║  - Minimizar distância total percorrida                                       ║
║  - Minimizar custo operacional                                                ║
║  - Respeitar capacidade dos veículos                                          ║
║  - Respeitar janelas de tempo e limites de distância                          ║
║  - Priorizar entrega de cidades críticas (high-priority)                      ║
║                                                                               ║
║  WORKFLOW:                                                                    ║
║  1. Inicializa 20 cidades reais (lat/lon) de São Paulo                        ║
║  2. Cria população de 90 permutações aleatórias                               ║
║  3. Para cada "geração" (180 vezes):                                          ║
║     a. Avalia cada permutação segundo função fitness complexa                 ║
║     b. Seleciona melhores (tournament selection, k=5)                         ║
║     c. Gera filhos via Order Crossover (OX)                                   ║
║     d. Aplica mutação (Swap ou Reverse, 22% probabilidade)                    ║
║     e. Melhora localmente com 2-Opt (~45% dos filhos)                         ║
║  4. Ao fim, retorna a melhor solução encontrada (menor fitness)               ║
║  5. Gera relatório visual em Pygame com histogramas de evolução               ║
║  6. Integra com OpenAI GPT para análise humanizada e Q&A                      ║
║                                                                               ║
║  FEATURE ÚNICO: Decodificação dinâmica de cromossoma                          ║
║  - AG trabalha com uma PERMUTAÇÃO GLOBAL (ordem das 20 cidades)               ║
║  - Decoder decide em TEMPO REAL qual veículo leva cada cidade                 ║
║  - Isso permite distribuição inteligente sem pré-divisão de clientes          ║
║                                                                               ║
║  AUTHOR: Fernando Monin                                                       ║
║  DATA: Março 2026                                                             ║
║  ACADÊMICO: FIAP - Pós-Graduação em Inteligência Artificial                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import math
import random
import sys
from typing import Dict, List, Tuple

import pygame

from cities_data import get_balanced_raw_cities
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
from llm_integration import (
    answer_route_question,
    append_history_entry,
    generate_daily_report,
    generate_driver_instructions,
    generate_process_improvements,
    generate_weekly_report,
)

# ────────────────────────────────────────────────────────────────────────────────
# CAMADA 1: CONFIGURAÇÃO GLOBAL
# ────────────────────────────────────────────────────────────────────────────────
# Define tamanho da interface, parâmetros do AG e características dos veículos

# CONFIGURAÇÃO DE INTERFACE (PyGame)
WIDTH, HEIGHT = 1600, 900
MAP_LEFT = 20
MAP_TOP = 20
MAP_WIDTH = 980
MAP_HEIGHT = 860
RIGHT_PANEL_X = 1030
FPS = 30

# PARÂMETROS DO ALGORITMO GENÉTICO
# ⚡ Estes valores impactam PROFUNDAMENTE na qualidade vs velocidade
# - POPULATION_SIZE = 90, N_GENERATIONS = 180 = ~5-10 min de execução (melhor qualidade)
# - POPULATION_SIZE = 30, N_GENERATIONS = 50  = ~30-40 seg (balanço)
# - POPULATION_SIZE = 8,  N_GENERATIONS = 10  = ~2-3 seg (para web/Streamlit - responsivo)
#
# Fórmula empiricamente validada: tempo_execução ≈ (pop_size × gen) × 0.003 segundos
POPULATION_SIZE = 90
N_GENERATIONS = 15
MUTATION_PROBABILITY = 0.22        # 22% de chance de mutar cada filho (exploração vs exploração)
ELITISM = 2                        # Preserva os 2 melhores indivíduos por geração (elitismo)
TOURNAMENT_SIZE = 6                # k=6 para tournament selection (balanceado entre seleção forte e diversidade)

# CORES DA INTERFACE (RGB Tuples)
NODE_RADIUS = 5
BACKGROUND = (14, 18, 28)          # Cinza escuro/azul para contraste com rotas
MAP_BACKGROUND = (24, 30, 45)      # Azul mais claro para diferençar mapa principal
GRID_COLOR = (50, 58, 82)          # Grid das ruas/linhas de referência
GRAY_ROUTE = (150, 150, 150)       # Rotas em TESTE (mais claras, para comparação)
WHITE = (240, 240, 240)            # Texto principal e elementos destacados
SOFT_TEXT = (205, 210, 220)        # Texto secundário (menos importante)
YELLOW = (240, 210, 90)            # Highlight das cidades no mapa

random.seed(42)  # SEED IMPORTANTE: garante reprodutibilidade dos resultados (essencial para pesquisa acadêmica)

# ────────────────────────────────────────────────────────────────────────────────
# CAMADA 2: DEFINIÇÃO DOS 3 VEÍCULOS (Fleet Configuration)
# ────────────────────────────────────────────────────────────────────────────────
# Define as características operacionais de cada veículo da frota
# O algoritmo genético atribui dinamicamente cidades para cada veículo
#
# ESTRUTURA DE CADA VEÍCULO:
# - capacity: kg máximo que pode transportar
# - max_distance: quilômetros máximos por dia (autonomia)
# - max_work_minutes: Tempo máximo de trabalho (em minutos por dia)
# - max_stops: Número máximo de paradas (cidades) que pode atender
# - service_minutes_*: Tempo de serviço em cada cidade (varia por prioridade)
# - operational_cost: R$ por km (combustível + manutenção)
# - fixed_cost: R$ fixo por dia (motorista, etc)
# - critical_bonus: Desconto em fitness por entregar cidades críticas (quanto maior, melhor)
#
# NOTA: O bonus é multiplicado pelo número de cidades críticas na rota
# Exemplo: Veículo 2 com 4 cidades críticas = 4 × 14.0 = 56 pontos de desconto no fitness

VEHICLES = [
    {
        'id': 1,
        'label': 'Veículo 1',
        'name': 'Pequeno',
        'capacity': 190,           # 190 kg de carga máxima
        'max_distance': 280.0,     # 280 km de autonomia por dia
        'max_work_minutes': 420,   # 7 horas = 420 minutos de trabalho
        'max_stops': 22,           # Máximo 22 cidades por dia
        'service_minutes_regular': 8,      # 8 min por cidade regular
        'service_minutes_critical': 14,    # 14 min por cidade crítica (mais importante)
        'operational_cost': 1.00,  # R$ 1.00 por km
        'fixed_cost': 15.0,       # R$ 15.00 fixo por dia (motorista, etc)
        'critical_bonus': 8.0,    # Desconto menor = menos preferência por críticas
        'color': (89, 163, 255),  # Azul (cor para renderização no mapa)
    },
    {
        'id': 2,
        'label': 'Veículo 2',
        'name': 'Médio',
        'capacity': 320,           # 320 kg - maior que Veículo 1
        'max_distance': 520.0,     # 520 km - maior autonomia
        'max_work_minutes': 540,   # 9 horas = 540 minutos
        'max_stops': 34,           # Pode fazer mais paradas
        'service_minutes_regular': 8,      # Tempo similar de serviço
        'service_minutes_critical': 15,    # Um pouco mais para críticas
        'operational_cost': 1.12,  # R$ 1.12 por km (um pouco mais caro)
        'fixed_cost': 25.0,       # R$ 25.00 fixo por dia (custos maiores)
        'critical_bonus': 14.0,   # MELHOR bonus para críticas (2x do Veículo 1)
        'color': (70, 210, 170),  # Verde (cor para renderização)
    },
    {
        'id': 3,
        'label': 'Veículo 3',
        'name': 'Grande',
        'capacity': 520,           # 520 kg - maior capacidade
        'max_distance': 900.0,     # 900 km - maior autonomia
        'max_work_minutes': 660,   # 11 horas = 660 minutos
        'max_stops': 50,           # Máximo 50 paradas (muito maior)
        'service_minutes_regular': 9,      # Um pouco mais de tempo por parada
        'service_minutes_critical': 16,    # Mais tempo para críticas
        'operational_cost': 1.28,  # R$ 1.28 por km (combustível maior)
        'fixed_cost': 35.0,       # R$ 35.00 fixo por dia (maior custo fixo)
        'critical_bonus': 18.0,   # MAIOR bonus (2.25x do Veículo 1) - ideal para críticas
        'color': (255, 165, 80),  # Laranja (cor para renderização)
    },
]

# ────────────────────────────────────────────────────────────────────────────────
# CAMADA 3: FUNÇÕES AUXLIARES DE MAPEAMENTO E DISTÂNCIA
# ────────────────────────────────────────────────────────────────────────────────

def latlon_to_xy(
    lat: float,
    lon: float,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> Tuple[int, int]:
    """
    Converte coordenadas geográficas (latitude, longitude) em pixels da tela (x, y).
    
    PROPÓSITO: Essa função é essencial para renderizar cidades no mapa PyGame.
    Nós recebemos coordenadas reais (lat/lon) de São Paulo e precisamos mapeá-las
    para a janela desenhável (1600×900).
    
    ALGORITMO:
    1. Calcula o "box" geográfico: (min_lat, max_lat, min_lon, max_lon)
    2. Normaliza cada coordenada para proporção [0, 1]
    3. Escala para pixels: [20, MAP_WIDTH-20] e [20, MAP_HEIGHT-20]
    
    EXEMPLO:
    - São Paulo center: lat=-23.55, lon=-46.63
    - Se min_lat=-24.0, max_lat=-23.0, min_lon=-47.5, max_lon=-46.0
    - ratio_x = (-46.63 - (-47.5)) / (-46.0 - (-47.5)) = 0.87 / 1.5 = 0.58
    - pixel_x = 20 + 20 + (0.58 × 940) ≈ 565 px
    
    NOTA: Invertemos a latitude (max_lat - lat) porque PyGame tem Y crescendo para baixo,
    enquanto latitude cresce para cima.
    """
    usable_width = MAP_WIDTH - 40   # 980 - 40 = 940 px úteis
    usable_height = MAP_HEIGHT - 40  # 860 - 40 = 820 px úteis
    x_ratio = (lon - min_lon) / (max_lon - min_lon)  # Normaliza longitude [0, 1]
    y_ratio = (max_lat - lat) / (max_lat - min_lat)  # Normaliza latitude invertida [0, 1]
    x = MAP_LEFT + 20 + int(x_ratio * usable_width)   # Escala para pixel
    y = MAP_TOP + 20 + int(y_ratio * usable_height)   # Escala para pixel
    return x, y


def haversine_km(city_a: Dict, city_b: Dict) -> float:
    """
    Calcula a distância real em quilômetros entre duas cidades usando a fórmula de HAVERSINE.
    
    PROPÓSITO: Em vez de usar distância euclidiana (linha reta), usamos a distância 
    sobre a ESFERA TERRESTRE. Isso é muito mais preciso para logística real.
    
    FÓRMULA DE HAVERSINE:
    d = 2 × R × arcsin(√(sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)))
    
    Onde:
    - R = 6371 km (raio médio da Terra)
    - Δlat = latitude_2 - latitude_1 (em radianos)
    - Δlon = longitude_2 - longitude_1 (em radianos)
    
    EXEMPLO:
    São Paulo (-23.55, -46.63) para Santos (-23.96, -46.33):
    - Distância em linha reta: ~64 km
    - Distância haversine (real): ~71 km ✓ Mais realista
    
    NOTA: Essa função é chamada centenas de vezes por geração do AG,
    portanto sua eficiência é crítica. Mantemos radianos pré-calculados.
    """
    radius = 6371.0  # Raio da Terra em km
    lat1 = math.radians(city_a['lat'])
    lon1 = math.radians(city_a['lon'])
    lat2 = math.radians(city_b['lat'])
    lon2 = math.radians(city_b['lon'])
    dlat = lat2 - lat1  # Diferença de latitude
    dlon = lon2 - lon1  # Diferença de longitude
    # Fórmula de haversine em sua forma canônica
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def route_distance_km(route: List[Dict], depot: Dict) -> float:
    """
    Calcula a distância TOTAL de uma rota completa (ida + volta ao depot).
    
    SEQUÊNCIA: depot → cidade_1 → cidade_2 → ... → cidade_n → depot
    
    EXEMPLO com 3 cidades:
    Depot (São Paulo) → Santos → Guarulhos → Osasco → Depot
    Distância = dist(SP, Santos) + dist(Santos, Guarulhos) + dist(Guarulhos, Osasco) + dist(Osasco, SP)
    
    NOTA: Rota vazia = 0 km
    """
    if not route:
        return 0.0
    total = haversine_km(depot, route[0])  # Depot até primeira cidade
    for i in range(len(route) - 1):
        total += haversine_km(route[i], route[i + 1])  # Cidade a cidade
    total += haversine_km(route[-1], depot)  # Última cidade volta ao depot
    return total


def estimate_route_minutes(route: List[Dict], vehicle: Dict, depot: Dict) -> float:
    """
    Estima o TEMPO TOTAL para completar uma rota, em minutos.
    
    COMPONENTES DO TEMPO:
    1. Tempo de deslocamento = distância_km / velocidade_média
    2. Tempo de serviço = soma dos tempos em cada parada
    
    VELOCIDADE MÉDIA: 45 km/h (velocidade média urbana para São Paulo)
    Fórmula: (distância / 45) × 60 = minutos de deslocamento
    \n    TEMPO POR PARADA:
    - Cidade regular: 8 min (unload rápido)
    - Cidade crítica: 14-16 min (mais importante, pode ter papelada extra)
    
    EXEMPLO com 5 cidades (2 críticas, 3 regulares):
    - Distância: 120 km
    - Tempo deslocamento: (120 / 45) × 60 = 160 min
    - Tempo serviço: 14 + 8 + 8 + 16 + 8 = 54 min
    - Total: 160 + 54 = 214 min (3,6 horas)
    """
    if not route:
        return 0.0
    distance_km = route_distance_km(route, depot)
    travel_minutes = (distance_km / 45.0) * 60.0  # 45 km/h é a velocidade média urbana
    service_minutes = 0.0
    for city in route:
        if city['priority'] == 'critica':
            service_minutes += vehicle['service_minutes_critical']
        else:
            service_minutes += vehicle['service_minutes_regular']
    return travel_minutes + service_minutes


def build_city_objects() -> Tuple[List[Dict], Dict]:
    """
    Carrega as 20 cidades reais de São Paulo e cria objetos estruturados para uso no AG.
    
    FLUXO:
    1. Importa 20 cidades balanceadas (70% regular, 30% crítica) de cities_data.py
    2. Calcula bounding box geográfico: (min/max lat/lon)
    3. Transforma coordenadas reais em pixels de tela (para renderização)
    4. Define depot em São Paulo (ponto de origem/destino de todas as rotas)
    5. Retorna lista de cidades estruturadas + depot
    
    ESTRUTURA DE CADA CIDADE:
    {
      'id': 1-20 (identificador único)
      'name': 'Santos', 'Guarulhos', etc (nome da cidade)
      'lat': -23.96 (latitude em graus decimais)
      'lon': -46.33 (longitude em graus decimais)
      'demand': 45 (kg a entregar)
      'priority': 'critica' ou 'regular' (10-15% recebem bonus/penalty maior)
      'group': ID do grupo interno
      'screen_pos': (x_pixel, y_pixel) - para renderização Pygame
    }
    
    IMPORTÂNCIA DO DEPOT:
    - Todas as rotas COMEÇAM e TERMINAM no depot
    - Coordenadas: Centro de Distribuição em São Paulo (-23.55, -46.63)
    - Demand = 0 (não é uma entrega)
    - Priority = 'depot' (tipo especial)
    """
    selected_raw_cities = get_balanced_raw_cities()

    min_lat = min(item[2] for item in selected_raw_cities)
    max_lat = max(item[2] for item in selected_raw_cities)
    min_lon = min(item[3] for item in selected_raw_cities)
    max_lon = max(item[3] for item in selected_raw_cities)

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
    for city_id, raw in enumerate(selected_raw_cities, start=1):
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

    return all_cities, depot


# ────────────────────────────────────────────────────────────────────────────────
# CAMADA 4: REGRAS DE NEGÓCIO DO VRP (Vehicle Routing Problem)
# ────────────────────────────────────────────────────────────────────────────────
# Estas funções definem as PENALIDADES e RECOMPENSAS que o AG otimiza
# 
# CONCEITO FUNDAMENTAL: Fitness não é apenas "menor distância"
# É uma COMPOSIÇÃO ponderada de múltiplas métricas de negócio:
# 
# Fitness = distância + custo_operacional + penalidades_violações - bonificações
# 
# Penalidades (incrementam fitness = piora):
#   - Superar capacidade: 14 R$/kg extra (MUITO CARO: não permitir!)
#   - Superar distância máx: 10 R$/km extra
#   - Superar tempo máx: 2.4 R$/min extra
#   - Entregar crítica fora de prazo: 2.5 R$ por posição (position * 2.5)
# 
# Bonificações (decrementam fitness = melhora):
#   - Entregar crítica cedo: -8 a -18 R$ dependendo do veículo
# IMPLEMENTAÇÃO: Aloca critical penalidades e rewards baseado em posição de entrega

def priority_position_penalty(route: List[Dict]) -> float:
    # Calcula penalidade/recompensa por ordem de entrega de cidades críticas
    # Penaliza entregas críticas em posições tardias, recompensa entregas rápidas
    penalty = 0.0
    reward = 0.0
    for index, city in enumerate(route):
        position_factor = index + 1
        if city['priority'] == 'critica':
            penalty += position_factor * 2.5
            reward += max(0, 20 - position_factor) * 1.2
    return penalty - reward


def route_demand(route: List[Dict]) -> int:
    """
    Soma a demanda (kg) de todas as cidades na rota.
    Simples, mas crucial: se ultrapassar, paga penalidade de 14 por kg extra!
    """
    return sum(city['demand'] for city in route)


def route_critical_count(route: List[Dict]) -> int:
    """
    Conta quantas cidades CRÍTICAS estão nesta rota.
    Cada uma gera desconto no fitness (recompensa do veículo).
    """
    return sum(1 for city in route if city['priority'] == 'critica')


def evaluate_route_for_vehicle(route: List[Dict], vehicle: Dict, depot: Dict) -> Dict:
    """
    FUNÇÃO CRÍTICA: Calcula a qualidade (fitness) de UMA rota para UM veículo.
    
    Esta é uma das funções mais importantes do projeto.
    Ela define literalmente o que significa "uma BOA rota".
    
    COMPONENTES DO FITNESS:
    1. Base: distância_km (minimizar km percorridos)
    2. Custo operacional: distância × cost_por_km + custo_fixo
    3. Penalidades de violação:
       - Capacidade: (demanda_extra) × 14 R$/kg (PESADO: capacidade é rígida!)
       - Distância: (km_extra) × 10 R$/km
       - Tempo: (min_extra) × 2.4 R$/min
       - Paradas: (paradas_extra) × 18 R$/parada (menos importante)
    4. Penalidade de prioridade: Penalizar entregar críticas tarde
    5. Desconto crítico: -critical_count × vehicle['critical_bonus']
       - Veículo 1: -8 por crítica
       - Veículo 2: -14 por crítica (preferido para críticas!)
       - Veículo 3: -18 por crítica (BEST para críticas)
    
    FÓRMULA FINAL:
    fitness = dist + custos + penalidades - descontos
    Objetivo: MINIMIZAR fitness (buscar solução com menor valor)
    
    EXEMPLO COM 3 CIDADES:
    Route = [Santos (50kg, regular), Guarulhos (30kg, crítica), Osasco (40kg, crítica)]
    Vehicle = Veículo 2 (capacity=320, max_distance=520, cost=1.12)
    
    Cálculos:
    - distance = 150 km
    - demand = 50 + 30 + 40 = 120 kg (OK, < 320)
    - critical_count = 2
    - time = (150/45)*60 + 8 + 15 + 15 = 200 + 38 = 238 min (OK, < 540)
    - penalties = 0 (tudo dentro dos limites)
    - priority_penalty = (1×2.5 + 2×2.5) - reward = -15 (net, com recompensa)
    - travel_cost = 150 × 1.12 = 168
    - operational = 168 + 25 = 193
    - critical_discount = 2 × 14 = 28
    - FITNESS = 150 + 193 + (-15) + 0 - 28 = 300
    """
    distance_km = route_distance_km(route, depot)
    total_demand = route_demand(route)
    critical_count = route_critical_count(route)
    stop_count = len(route)
    work_minutes = estimate_route_minutes(route, vehicle, depot)

    # Calcular penalidades de violação de constraints
    penalty = 0.0
    if total_demand > vehicle['capacity']:
        penalty += (total_demand - vehicle['capacity']) * 14.0  # Muito caro!
    if distance_km > vehicle['max_distance']:
        penalty += (distance_km - vehicle['max_distance']) * 10.0
    if work_minutes > vehicle['max_work_minutes']:
        penalty += (work_minutes - vehicle['max_work_minutes']) * 2.4
    if stop_count > vehicle['max_stops']:
        penalty += (stop_count - vehicle['max_stops']) * 18.0

    # Componentes do fitness
    priority_penalty = priority_position_penalty(route)
    travel_cost = distance_km * vehicle['operational_cost']
    fixed_cost = vehicle['fixed_cost']
    operational_component = travel_cost + fixed_cost
    critical_discount = critical_count * vehicle['critical_bonus']

    # FÓRMULA: fitness = distance + operational + priority_penalty + violations - discount
    fitness = distance_km + operational_component + priority_penalty + penalty - critical_discount

    return {
        'fitness': fitness,
        'distance_km': distance_km,
        'demand': total_demand,
        'critical_count': critical_count,
        'priority_penalty': priority_penalty,
        'travel_cost': travel_cost,
        'fixed_cost': fixed_cost,
        'total_cost': operational_component,
        'critical_discount': critical_discount,
        'penalty': penalty,
        'stop_count': stop_count,
        'work_minutes': work_minutes,
        'vehicle': vehicle,
    }


def rebalance_routes(routes_by_vehicle: Dict[int, List[Dict]], depot: Dict) -> Dict[int, List[Dict]]:
    """Faz pequenos ajustes movendo cidades do fim de rotas muito penalizadas.

    Isso ajuda a distribuição dinâmica entre veículos ficar mais aderente ao VRP.
    """
    changed = True
    max_passes = 5
    pass_count = 0

    while changed and pass_count < max_passes:
        changed = False
        pass_count += 1
        evaluations = {
            v['id']: evaluate_route_for_vehicle(routes_by_vehicle[v['id']], v, depot)
            for v in VEHICLES
        }
        overloaded = [v for v in VEHICLES if evaluations[v['id']]['penalty'] > 0 and routes_by_vehicle[v['id']]]
        underloaded = [v for v in VEHICLES if evaluations[v['id']]['penalty'] <= 0]

        for source_vehicle in overloaded:
            source_id = source_vehicle['id']
            route = routes_by_vehicle[source_id]
            if not route:
                continue

            candidate_indices = list(range(len(route) - 1, max(-1, len(route) - 5), -1))
            candidate_indices += [i for i, city in enumerate(route) if city['priority'] == 'regular']

            moved = False
            for idx in candidate_indices:
                city = route[idx]
                best_target_id = None
                best_improvement = 0.0
                source_before = evaluations[source_id]['fitness']

                for target_vehicle in underloaded:
                    target_id = target_vehicle['id']
                    if target_id == source_id:
                        continue

                    source_after_route = [c for pos, c in enumerate(route) if pos != idx]
                    target_after_route = routes_by_vehicle[target_id] + [city]
                    source_after = evaluate_route_for_vehicle(source_after_route, source_vehicle, depot)['fitness']
                    target_before = evaluations[target_id]['fitness']
                    target_after = evaluate_route_for_vehicle(target_after_route, target_vehicle, depot)['fitness']
                    improvement = (source_before + target_before) - (source_after + target_after)

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_target_id = target_id

                if best_target_id is not None:
                    moved_city = routes_by_vehicle[source_id].pop(idx)
                    routes_by_vehicle[best_target_id].append(moved_city)
                    changed = True
                    moved = True
                    break

            if moved:
                break

    return routes_by_vehicle


def decode_chromosome_to_routes(sequence: List[Dict], depot: Dict) -> List[Dict]:
    """
    🔑 FUNÇÃO MAIS IMPORTANTE! Decodifica um cromossoma do AG em uma solução VRP.
    
    CONCEITO FUNDAMENTAL - A INOVAÇÃO DESTE PROJETO:
    ═════════════════════════════════════════════════════════════════════════════
    
    O Algoritmo Genético trabalha com uma PERMUTAÇÃO GLOBAL das 20 cidades:
    [Santos, Guarulhos, Campinas, Osasco, Mogi das Cruzes, ...]
    
    MAS: Um cromossoma não define "qual veículo leva qual cidade".
    Por baixo dos panos, o DECODER implementa uma HEURÍSTICA GULOSA (greedy):
    
    Para cada cidade, na ordem do cromossoma:
      1. Avalia o IMPACTO de adicionar essa cidade a cada veículo
      2. Escolhe o veículo que causa o MENOR AUMENTO DE FITNESS
      3. Se a cidade é CRÍTICA e veículo tem alto bonus, favorece esse veículo
      4. Também considera BALANCE entre veículos (não sobrecarregar um só)
    
    VANTAGEM:
    - Reduz espaço de busca: AG busca em permutações (n!), não em (n choose k) divisões
    - Permite alocação DINÂMICA: próxima cidade "escolhe" melhor veículo
    - Capture soft constraints: críticas vão para veículos com maior critical_bonus
    - Muito mais eficiente computacionalmente!
    """
    routes_by_vehicle = {vehicle['id']: [] for vehicle in VEHICLES}

    # PASSO 1: Aloca cada cidade ao melhor veículo (GREEDY)
    for city in sequence:
        best_vehicle_id = None
        best_score = float('inf')

        for vehicle in VEHICLES:
            current_route = routes_by_vehicle[vehicle['id']]
            current_eval = evaluate_route_for_vehicle(current_route, vehicle, depot)
            candidate_route = current_route + [city]
            candidate_eval = evaluate_route_for_vehicle(candidate_route, vehicle, depot)

            # Impacto direto: quanto pior fica o fitness ao adicionar esta cidade?
            increment = candidate_eval['fitness'] - current_eval['fitness']
            
            # Penalidade de balanço: evitar sobrecarregar um veículo
            media_cidades_por_veiculo = len(sequence) / len(VEHICLES)
            balance_penalty = max(0, len(candidate_route) - media_cidades_por_veiculo) * 0.9
            
            # Bias para cidades críticas: preferir veículos com maior critical_bonus
            critical_bias = -6.0 if city['priority'] == 'critica' and vehicle['critical_bonus'] >= 14 else 0.0
            
            # Penalidade por proximidade de limite de capacidade
            capacity_margin = max(0, candidate_eval['demand'] - vehicle['capacity']) * 2.0
            
            # Score final: quanto menor, melhor é adicionar esta cidade neste veículo
            score = increment + balance_penalty + capacity_margin + critical_bias

            if score < best_score:
                best_score = score
                best_vehicle_id = vehicle['id']

        routes_by_vehicle[best_vehicle_id].append(city)

    # PASSO 2: Realbalanceamento (mover cidades de rotas penalizadas para boas)
    routes_by_vehicle = rebalance_routes(routes_by_vehicle, depot)

    # PASSO 3: Avalia solução final e retorna métricas
    results = []
    total_fitness = 0.0
    for vehicle in VEHICLES:
        route = routes_by_vehicle[vehicle['id']]
        result = evaluate_route_for_vehicle(route, vehicle, depot)
        result['best_route'] = route  # Armazena a ordem das cidades
        results.append(result)
        total_fitness += result['fitness']

    results.sort(key=lambda item: item['vehicle']['id'])
    return results


def evaluate_solution(sequence: List[Dict], depot: Dict) -> Tuple[float, List[Dict]]:
    results = decode_chromosome_to_routes(sequence, depot)
    total_fitness = sum(item['fitness'] for item in results)
    return total_fitness, results


def make_fitness_function(depot: Dict):
    def evaluator(sequence: List[Dict]) -> float:
        return evaluate_solution(sequence, depot)[0]
    return evaluator


def evolve_global_population(population: List[List[Dict]], depot: Dict):
    population_fitness = [evaluate_solution(individual, depot)[0] for individual in population]
    population, population_fitness = sort_population(population, population_fitness)

    best_sequence = population[0]
    best_total_fitness, best_results = evaluate_solution(best_sequence, depot)

    new_population = [population[i] for i in range(min(ELITISM, len(population)))]
    solution_fitness = make_fitness_function(depot)

    while len(new_population) < len(population):
        parent1 = tournament_selection(population, population_fitness, k=TOURNAMENT_SIZE)
        parent2 = tournament_selection(population, population_fitness, k=TOURNAMENT_SIZE)
        child = order_crossover(parent1, parent2)
        child = mutate(child, MUTATION_PROBABILITY)

        if random.random() < 0.45:
            child = two_opt_improve(child, fitness_function=solution_fitness)

        new_population.append(child)

    testing_sequence = random.choice(population[:min(8, len(population))])
    _, testing_results = evaluate_solution(testing_sequence, depot)

    return new_population, best_total_fitness, best_results, population_fitness, testing_results


def draw_map_background(screen: pygame.Surface, depot: Dict) -> None:
    map_rect = pygame.Rect(MAP_LEFT, MAP_TOP, MAP_WIDTH, MAP_HEIGHT)
    pygame.draw.rect(screen, MAP_BACKGROUND, map_rect, border_radius=10)
    pygame.draw.rect(screen, (65, 82, 120), map_rect, 1, border_radius=10)

    for x in range(MAP_LEFT + 20, MAP_LEFT + MAP_WIDTH, 70):
        pygame.draw.line(screen, GRID_COLOR, (x, MAP_TOP + 10), (x, MAP_TOP + MAP_HEIGHT - 10), 1)
    for y in range(MAP_TOP + 20, MAP_TOP + MAP_HEIGHT, 70):
        pygame.draw.line(screen, GRID_COLOR, (MAP_LEFT + 10, y), (MAP_LEFT + MAP_WIDTH - 10, y), 1)

    draw_text(screen, 'Mapa das cidades da região de São Paulo', WHITE, (MAP_LEFT + 14, MAP_TOP + 10), font_size=20, bold=True)
    draw_text(screen, 'Cinza = rota em teste | Melhor rota usa a cor do veículo', SOFT_TEXT, (MAP_LEFT + 14, MAP_TOP + 36), font_size=14)

    pygame.draw.circle(screen, (255, 120, 120), depot['screen_pos'], 8)
    draw_text(screen, 'CD', (255, 255, 255), (depot['screen_pos'][0] + 8, depot['screen_pos'][1] - 8), font_size=12, bold=True)


def route_to_screen_points(route: List[Dict], depot: Dict) -> List[Tuple[int, int]]:
    if not route:
        return []
    return [depot['screen_pos']] + [city['screen_pos'] for city in route] + [depot['screen_pos']]


def build_generation_totals(route_results: List[Dict]) -> Dict[str, float]:
    """Consolida as métricas da geração atual para facilitar logs e gráficos."""
    return {
        'distance': sum(item['distance_km'] for item in route_results),
        'demand': sum(item['demand'] for item in route_results),
        'time': sum(item['work_minutes'] for item in route_results),
        'cost': sum(item['total_cost'] for item in route_results),
        'priority': sum(item['priority_penalty'] for item in route_results),
        'penalty': sum(item['penalty'] for item in route_results),
        'fitness': sum(item['fitness'] for item in route_results),
    }


def build_dashboard_data(
    best_global_fitness_history: List[float],
    vehicle_metric_history: Dict[int, Dict[str, List[float]]],
    total_metric_history: Dict[str, List[float]],
) -> Dict:
    x_values = list(range(1, len(best_global_fitness_history) + 1))
    fitness_series = [
        {
            'label': 'Fitness global',
            'x': x_values,
            'values': best_global_fitness_history,
            'color': (255, 255, 255),
        }
    ]

    vehicle_distance_series = []
    for vehicle in VEHICLES:
        metric_series = vehicle_metric_history[vehicle['id']]
        fitness_series.append(
            {
                'label': f"{vehicle['label']} - {vehicle['name']}",
                'x': list(range(1, len(metric_series['fitness']) + 1)),
                'values': metric_series['fitness'],
                'color': vehicle['color'],
            }
        )
        vehicle_distance_series.append(
            {
                'label': f"{vehicle['label']} - {vehicle['name']}",
                'x': list(range(1, len(metric_series['distance']) + 1)),
                'values': metric_series['distance'],
                'color': vehicle['color'],
            }
        )

    totals = {
        'x': x_values,
        'distance': total_metric_history['distance'],
        'demand': total_metric_history['demand'],
        'time': total_metric_history['time'],
        'cost': total_metric_history['cost'],
        'priority': total_metric_history['priority'],
        'penalty': total_metric_history['penalty'],
    }

    return {
        'fitness_series': fitness_series,
        'totals': totals,
        'vehicle_distance_series': vehicle_distance_series,
    }


def print_generation_report(generation: int, global_fitness: float, route_results: List[Dict], generation_totals: Dict[str, float]) -> None:
    """Mostra um relatório mais limpo e agradável no terminal."""
    header = (
        f"\nGeração {generation:03d} | Fitness global: {global_fitness:8.2f} | "
        f"Distância total: {generation_totals['distance']:7.1f} km | "
        f"Tempo total: {generation_totals['time']:7.1f} min"
    )
    print(header)
    print('=' * len(header))
    print(
        f"{'Veículo':<22} {'Cidades':>6} {'Demanda':>8} {'Dist(km)':>10} {'Tempo':>9} "
        f"{'Custo':>9} {'Priorid.':>10} {'Penalid.':>10} {'Fitness':>10}"
    )
    print('-' * 104)

    for result in route_results:
        vehicle = result['vehicle']
        vehicle_name = f"{vehicle['label']} - {vehicle['name']}"
        print(
            f"{vehicle_name:<22} {len(result['best_route']):>6} {result['demand']:>8} "
            f"{result['distance_km']:>10.1f} {result['work_minutes']:>9.1f} "
            f"{result['total_cost']:>9.1f} {result['priority_penalty']:>10.1f} "
            f"{result['penalty']:>10.1f} {result['fitness']:>10.2f}"
        )

    print('-' * 104)
    print(
        f"{'TOTAL':<22} {sum(len(item['best_route']) for item in route_results):>6} "
        f"{generation_totals['demand']:>8.0f} {generation_totals['distance']:>10.1f} "
        f"{generation_totals['time']:>9.1f} {generation_totals['cost']:>9.1f} "
        f"{generation_totals['priority']:>10.1f} {generation_totals['penalty']:>10.1f} "
        f"{generation_totals['fitness']:>10.2f}"
    )
    print()


def print_text_block(title: str, content: str) -> None:
    """Mostra blocos de texto de um jeito simples e legível no terminal."""
    line = '=' * max(20, len(title) + 8)
    print(f"\n{line}")
    print(title)
    print(line)
    print(content)


# ────────────────────────────────────────────────────────────────────────────────
# CAMADA 5: PONTO DE ENTRADA - MAIN()
# ────────────────────────────────────────────────────────────────────────────────
# Orquestra a execução completa: AG + PyGame + LLM

def main() -> None:
    all_cities, depot = build_city_objects()
    assert len(all_cities) == 20, 'A base ativa precisa ter exatamente 20 cidades.'

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('VRP com Algoritmo Genético - 20 cidades balanceadas')
    clock = pygame.time.Clock()

    population = generate_random_population(all_cities, POPULATION_SIZE)

    best_global_fitness_history = []
    vehicle_metric_history = {
        vehicle['id']: {
            'fitness': [],
            'distance': [],
            'demand': [],
            'time': [],
            'cost': [],
            'priority': [],
            'penalty': [],
        }
        for vehicle in VEHICLES
    }
    total_metric_history = {
        'distance': [],
        'demand': [],
        'time': [],
        'cost': [],
        'priority': [],
        'penalty': [],
        'fitness': [],
    }

    running = True
    paused = False
    generation = 0
    final_results = None

    print('Execução iniciada...')
    print('Controles: Q = sair | P = pausar/continuar')
    print('-' * 110)

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

        population, global_fitness, merged_results, _, testing_results = evolve_global_population(population, depot)
        final_results = merged_results
        best_global_fitness_history.append(global_fitness)

        generation_totals = build_generation_totals(merged_results)
        total_metric_history['distance'].append(generation_totals['distance'])
        total_metric_history['demand'].append(generation_totals['demand'])
        total_metric_history['time'].append(generation_totals['time'])
        total_metric_history['cost'].append(generation_totals['cost'])
        total_metric_history['priority'].append(generation_totals['priority'])
        total_metric_history['penalty'].append(generation_totals['penalty'])
        total_metric_history['fitness'].append(generation_totals['fitness'])

        for result in merged_results:
            vehicle_id = result['vehicle']['id']
            vehicle_metric_history[vehicle_id]['fitness'].append(result['fitness'])
            vehicle_metric_history[vehicle_id]['distance'].append(result['distance_km'])
            vehicle_metric_history[vehicle_id]['demand'].append(result['demand'])
            vehicle_metric_history[vehicle_id]['time'].append(result['work_minutes'])
            vehicle_metric_history[vehicle_id]['cost'].append(result['total_cost'])
            vehicle_metric_history[vehicle_id]['priority'].append(result['priority_penalty'])
            vehicle_metric_history[vehicle_id]['penalty'].append(result['penalty'])

        for result in testing_results:
            test_points = route_to_screen_points(result['best_route'], depot)
            draw_paths(screen, test_points, GRAY_ROUTE, width=2, close_path=False)

        for result in merged_results:
            best_points = route_to_screen_points(result['best_route'], depot)
            draw_paths(screen, best_points, result['vehicle']['color'], width=4, close_path=False)

        draw_cities(screen, all_cities, YELLOW, NODE_RADIUS, show_labels=True)

        draw_text(screen, f'Geração: {generation}/{N_GENERATIONS}', WHITE, (RIGHT_PANEL_X, 20), font_size=22, bold=True)
        draw_text(screen, f'Fitness global: {global_fitness:.2f}', WHITE, (RIGHT_PANEL_X, 52), font_size=18)
        draw_text(screen, f'Mutação: {MUTATION_PROBABILITY:.2f}', SOFT_TEXT, (RIGHT_PANEL_X, 78), font_size=16)

        draw_vehicle_legend(screen, VEHICLES, (RIGHT_PANEL_X, 110))
        draw_route_summary(screen, merged_results, (RIGHT_PANEL_X, 275))

        dashboard_data = build_dashboard_data(
            best_global_fitness_history,
            vehicle_metric_history,
            total_metric_history,
        )

        draw_plot(
            screen,
            dashboard_data,
            position=(RIGHT_PANEL_X, 470),
        )

        print_generation_report(generation, global_fitness, merged_results, generation_totals)

        pygame.display.flip()
        clock.tick(FPS)

    # Salva exatamente o que está visível na janela ao encerrar a execução.
    screen_snapshot_path = 'execution_screen.png'
    try:
        pygame.image.save(screen, screen_snapshot_path)
        print(f'Tela final salva em: {screen_snapshot_path}')
    except pygame.error as exc:
        print(f'Não foi possível salvar a imagem da tela final: {exc}')

    if best_global_fitness_history:
        dashboard_data = build_dashboard_data(
            best_global_fitness_history,
            vehicle_metric_history,
            total_metric_history,
        )
        save_fitness_chart(dashboard_data, output_path='fitness_evolution.png')

    if final_results:
        print('\nResumo final da melhor solução encontrada nessa execução:')
        for result in final_results:
            vehicle = result['vehicle']
            city_names = ', '.join(city['name'] for city in result['best_route'][:6])
            print(
                f"{vehicle['label']} - {vehicle['name']} | cidades={len(result['best_route'])} | "
                f"distância={result['distance_km']:.1f} km | demanda={result['demand']} | "
                f"tempo={result['work_minutes']:.1f} min | custo={result['total_cost']:.1f} | "
                f"prioridade={result['priority_penalty']:.1f} | penalidade={result['penalty']:.1f} | "
                f"primeiras cidades={city_names}..."
            )
        print('Gráfico final salvo em: fitness_evolution.png')

        # Aqui eu salvo um histórico simples para usar no relatório semanal
        append_history_entry(final_results)

        # Aqui eu gero os textos da parte de LLM
        try:
            daily_report = generate_daily_report(final_results)
            driver_instructions = generate_driver_instructions(final_results)
            improvement_report = generate_process_improvements(final_results)
            weekly_report = generate_weekly_report()

            print_text_block('RELATÓRIO DIÁRIO', daily_report)
            print_text_block('INSTRUÇÕES POR VEÍCULO', driver_instructions)
            print_text_block('SUGESTÕES DE MELHORIA', improvement_report)
            print_text_block('RELATÓRIO SEMANAL', weekly_report)
        except RuntimeError as e:
            if "insufficient_quota" in str(e) or "quota" in str(e).lower():
                print("\n⚠️  QUOTA DA API OPENAI ESGOTADA")
                print("💡 O algoritmo genético funcionou perfeitamente!")
                print("💡 Para relatórios LLM, adicione créditos em: https://platform.openai.com/usage")
                print("💡 Ou use um modelo local gratuito como Ollama")
                print("\n📊 RESULTADOS FINAIS:")
                for result in final_results:
                    vehicle = result['vehicle']
                    print(f"  {vehicle['label']} - {vehicle['name']}: {len(result['best_route'])} cidades, "
                          f"{result['distance_km']:.1f}km, R${result['total_cost']:.1f}")
            else:
                print(f"\n❌ Erro na geração de relatórios: {e}")
                print("📊 Resultados básicos:")
                for result in final_results:
                    vehicle = result['vehicle']
                    print(f"  {vehicle['label']}: {len(result['best_route'])} cidades")

    pygame.quit()

    if final_results:
        try:
            while True:
                question = input('\nDigite uma pergunta sobre as rotas ou "E" para sair: ').strip()
                if question.lower() == 'e':
                    break
                if not question:
                    print('Digite uma pergunta válida ou "E" para sair.')
                    continue
                answer = answer_route_question(final_results, question)
                print_text_block('RESPOSTA DA LLM', answer)
        except (EOFError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and ("insufficient_quota" in str(e) or "quota" in str(e).lower()):
                print("\n⚠️  Sistema de perguntas indisponível - quota da API esgotada")
            else:
                print(f"\n❌ Erro no sistema de perguntas: {e}")

    sys.exit()


if __name__ == '__main__':
    main()
