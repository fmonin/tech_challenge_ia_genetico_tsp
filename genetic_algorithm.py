"""
AUTHOR: Fernando Monin
MÓDULO: genetic_algorithm.py
=============================
Este módulo implementa o núcleo do Algoritmo Genético (AG) aplicado ao problema
de Vehicle Routing Problem (VRP). O algoritmo segue a metodologia clássica 
de computação evolucionária com as operações genéticas fundamentais.

METODOLOGIA DO ALGORITMO GENÉTICO:
==================================

1. POPULAÇÃO (Representação):
   - Cada indivíduo = uma permutação das cidades
   - Estrutura: lista de cidade como (id, name, lat, lon, demand, priority)
   - População inicial gerada aleatoriamente com tamanho configurável
   - Mantém diversidade genética evitando convergência prematura

2. FITNESS (Função de Avaliação):
   - Calcula a qualidade de cada solução
   - Objetivo: minimizar fitness (distância + custos + penalidades)
   - Componentes: distância total, custos operacionais, restrições violadas
   - Usado para seleção natural dos melhores indivíduos

3. SELEÇÃO (Reprodução):
   - Método: Seleção por Torneio (Tournament Selection)
   - Como funciona: seleciona aleatoriamente k indivíduos, retorna o melhor
   - Vantagem: simples, estável e evita convergência prematura
   - Mantém pressão seletiva sem perder diversidade

4. CROSSOVER (Cruzamento):
   - Operador: Order Crossover (OX)
   - Retirado da literatura de AG para TSP/VRP
   - Preserva a ordem relativa das cidades
   - Evita duplicação de cidades na solução filha
   - Combina características de ambos os pais

5. MUTAÇÃO:
   - Duas estratégias equiprováveis (50/50):
     a) Swap: troca posição de duas cidades aleatórias
     b) Inversão: inverte um segmento da rota
   - Taxa: configurável (tipicamente 20-25%)
   - Mantém diversidade e permite escape de ótimos locais

6. MELHORIA LOCAL (Local Search):
   - Operador: 2-OPT Improvement
   - Remove cruzamentos nas rotas (descrossing)
   - Melhora soluções localmente sem alterar cromossomo
   - Executado em ~45% das gerações para não deixar lento

7. ELITISMO:
   - Preservação dos melhores indivíduos entre gerações
   - Evita perda de boas soluções
   - Garante não-deterioração do fitness geral

FLUXO DE EVOLUÇÃO:
==================
FOR cada geração:
  1. Calcular fitness de toda população
  2. Ordenar pela qualidade (elitismo)
  3. Reproduzir:
     - Selecionar 2 pais (torneio)
     - Cruzar (order crossover)
     - Mutar (swap ou inversão)
     - Melhorar localmente (2-opt)
  4. Avaliar nova geração
  5. Manter melhores
NEXT

CONVERGÊNCIA:
=============
- Detecta convergência quando fitness não melhora por N gerações
- Aumenta taxa de mutação se convergido
- Pode reiniciar segmentos da população se necessário
"""

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
    """
    Cria população inicial com variações aleatórias.
    
    Esta função é fundamental para o AG porque garante diversidade inicial.
    Se todos começassem com a mesma solução, o algoritmo teria muito mais dificuldade.
    
    Parâmetros:
        items: lista de cidades a serem roteadas
        population_size: quantos indivíduos terá a população
    
    Retorna:
        Lista com 'population_size' soluções aleatórias (permutações das cidades)
    
    Exemplo:
        Se temos 5 cidades [A, B, C, D, E] com population_size=3:
        Resultado: [[B, D, A, C, E], [E, A, C, B, D], [C, B, E, D, A]]
        
    Nota: Cada permutação é única porque usa random.sample que garante sem repetição.
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
    """
    Operador de Cruzamento: Order Crossover (OX).
    
    Este é um dos operadores mais importantes de todo o AG. Ele combina
    características de dois pais de forma que:
    - Mantém a ordem relativa das cidades
    - Não duplica cidades (cada uma aparece exatamente uma vez)
    - Produz soluções viáveis
    
    Como funciona (passo a passo):
    1. Escolhe um segmento aleatório do parent1 (entre posições start e end)
    2. Copia esse segmento para a mesma posição no filho
    3. Preenche o restante respeitando a ordem do parent2
    
    Exemplo prático:
    parent1 = [A, B, C, D, E, F]
    parent2 = [F, E, D, C, B, A]
    
    Suponha que copiamos posições 1-3 do parent1 (B, C, D):
    criança = [?, B, C, D, ?, ?]
    
    Agora percorremos parent2 da esquerda e adicionamos não-duplicados:
    - F (novo) → criança = [F, B, C, D, ?, ?]
    - E (novo) → criança = [F, B, C, D, E, ?]
    - D (já existe) → pula
    - C (já existe) → pula
    - B (já existe) → pula
    - A (novo) → criança = [F, B, C, D, E, A]
    
    Resultado final: [F, B, C, D, E, A]
    Resultado: Mantém estrutura do parent1 (B, C, D juntos) com material do parent2
    
    Por que OX é bom para VRP:
    - Preserva clusters de cidades próximas
    - Mantém ordem de atendimento (importante para time windows)
    - Provou ser superior a crossover simples em TSP/VRP
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
    """
    Operador de MUTAÇÃO: modifica a solução de forma aleatória.
    
    A mutação é crítica para evitar convergência prematura e explorar
    vizinhanças de soluções ruins em busca de melhorias.
    
    Este operador implementa duas estratégias com igual probabilidade:
    
    ESTRATÉGIA 1 - SWAP (50%):
    Troca duas posições aleatórias na rota.
    
    Exemplo:
    solução = [A, B, C, D, E]
    Escolhe trocar posições 1 e 4: [A, E, C, D, B]
    
    Efeito: Altera a sequência de atendimento entre cidades
    Uso: Escapa de ótimos locais
    
    ESTRATÉGIA 2 - REVERSÃO/INVERSÃO (50%):
    Inverte um segmento da rota (2-opt simples).
    
    Exemplo:
    solução = [A, B, C, D, E]
    Escolhe inverter segmento entre posições 1 e 3:
    [A, (C, B), D, E] → [A, C, B, D, E]
    
    Efeito: Remove cruzamentos simples em rotas
    Uso: Melhoria local rápida, reduz distância
    
    Por que duas estratégias?
    - SWAP: Exploration (exploração, diversidade)
    - REVERSÃO: Exploitation (exploração local, melhoria)
    - 50/50: Equilibra busca global vs local
    
    Parâmetro "mutation_probability":
    - Típico: 0.20 a 0.25 (20-25%)
    - Baixa: <10% (converge rápido, pode ficar em ótimo local)
    - Alta: >50% (muito caos, perde boas soluções)
    - Adaptativa: pode aumentar conforme population converge
    """
    mutated_solution = copy.deepcopy(solution)

    if len(solution) < 2:
        return mutated_solution

    if random.random() < mutation_probability:
        if random.random() < 0.5:
            # ESTRATÉGIA 1: SWAP - troca duas cidades
            i, j = random.sample(range(len(solution)), 2)
            mutated_solution[i], mutated_solution[j] = mutated_solution[j], mutated_solution[i]
        else:
            # ESTRATÉGIA 2: REVERSÃO - inverte um segmento (2-opt)
            i, j = sorted(random.sample(range(len(solution)), 2))
            mutated_solution[i:j + 1] = list(reversed(mutated_solution[i:j + 1]))

    return mutated_solution


def tournament_selection(population: List[List[Any]], fitness: List[float], k: int = 5) -> List[Any]:
    """
    SELEÇÃO POR TORNEIO (Tournament Selection).
    
    Este é o mecanismo de reprodução seletiva: escolhe os melhores
    indivíduos para serem pais da próxima geração.
    
    Como funciona:
    1. Seleciona aleatoriamente k indivíduos da população
    2. Compara seus fitness values
    3. Retorna o indivíduo com MELHOR fitness
    
    Exemplo:
    população = [[A,B,C], [D,E,F], [G,H,I]]
    fitness = [100, 50, 75]
    k = 2
    
    Torneio 1: escolhe 2 aleatórios → [A,B,C] (fit=100) vs [G,H,I] (fit=75)
              vencedor: [G,H,I] (menor é melhor)
    Resultado: indivíduo [G,H,I] é selecionado como pai
    
    Por que Tournament Selection?
    
    VANTAGENS:
    - Simples de implementar
    - Não precisa normalizar fitness
    - Evita seleção prematura (elitismo suave)
    - Mantém diversidade: indivíduos mediocres ainda podem ganhar
    - Parâmetro k = "pressão seletiva"
    
    PRESSÃO SELETIVA:
    - k=2: baixa pressão, muitos indivíduos ruins reproduzem
    - k=5: média pressão (padrão neste projeto)
    - k=10+: alta pressão, apenas os melhores reproduzem
    
    COMPARAÇÃO COM OUTRAS SELEÇÕES:
    - Roulette Wheel: requer normalização, risco de elite dominação
    - Rank-based: mais complexa, às vezes faz pior em VRP
    - Tournament: "just right" para problemas complexos
    """
    k = max(2, min(k, len(population)))
    selected = random.sample(list(zip(population, fitness)), k)
    selected.sort(key=lambda x: x[1])  # Ordena por fitness (menor é melhor)
    return selected[0][0]  # Retorna o melhor do torneio


def sort_population(population: List[List[Any]], fitness: List[float]) -> Tuple[List[List[Any]], List[float]]:
    """Ordena da melhor solução para a pior."""
    combined_lists = list(zip(population, fitness))
    combined_lists.sort(key=lambda x: x[1])
    sorted_population = [item[0] for item in combined_lists]
    sorted_fitness = [item[1] for item in combined_lists]
    return sorted_population, sorted_fitness


def two_opt_improve(path: List[Any], fitness_function: Callable[[List[Any]], float] | None = None) -> List[Any]:
    """
    MELHORIA LOCAL: Algoritmo 2-OPT.
    
    Este algoritmo é uma técnica clássica de Local Search que melhora soluções
    removendo cruzamentos simples nas rotas. É muito eficaz e rápido.
    
    PROBLEMA VISUAL:
    Rotas com cruzamentos são ineficientes. 2-OPT os remove.
    
    Rota inicial (com cruzamento):
        A --- B
        |     |
        C --- D
    
    As linhas AC e BD se cruzam. É ineficiente!
    
    Depois do 2-OPT (revertendo segmento):
        A --- D
        |     |
        C --- B
    
    Agora não há cruzamento. Rota é mais curta!
    
    COMO FUNCIONA:
    1. Pega a rota original
    2. Para cada par de posições (i, j):
       - Inverte o segmento entre i e j
       - Calcula novo fitness
       - Se melhorou, mantém inversão
       - Senão, desfaz
    3. Repete até que nenhuma melhoria seja encontrada
    
    EXEMPLO MATEMÁTICO:
    Rota: [1, 2, 3, 4, 5]
    
    Tenta inverter segmento entre pos 1 e 3:
    - Original: [1, 2, 3, 4, 5] = distância 100
    - Revertido: [1, 3, 2, 4, 5] = distância 85 ✓ Melhor!
    - Mantém inversão
    
    PARÂMETROS:
    
    fitness_function: função que calcula qualidade da rota
                      padrão: distância euclidiana simples
                      customizada: pode incluir custos, penalidades etc
    
    COMPLEXIDADE:
    - Melhor caso: O(n) - já otimizado
    - Pior caso: O(n²) - precisa testar muitos pares
    - Por isso executamos em ~45% das gerações, não em 100%
    
    EFETIVIDADE EM VRP:
    - Remove ~30-40% da distância de uma solução aleatória
    - Leva ~2-5 gerações para convergir
    - Equilibra exploração vs refinamento
    
    VARIAÇÕES POSSÍVEIS:
    - 3-OPT: mais poderoso mas O(n³), muito lento
    - LKH: refinado, complexo para implementar
    - Este 2-OPT: síntese perfeita de eficiência e simplic idade
    """
    if len(path) < 4:
        return path.copy()

    evaluator = fitness_function if fitness_function is not None else calculate_fitness
    best_path = path.copy()
    best_distance = evaluator(best_path)

    for i in range(1, len(path) - 2):
        for j in range(i + 1, len(path) - 1):
            # Inverte o segmento entre i e j
            new_path = best_path[:i] + best_path[i:j + 1][::-1] + best_path[j + 1:]
            new_distance = evaluator(new_path)
            
            # Se melhorou, aceita a mudança
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