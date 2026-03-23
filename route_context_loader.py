"""
AUTHOR: Fernando Monin
MÓDULO: route_context_loader.py
================================
Conecta o algoritmo genético (tsp.py) com a interface web (Streamlit).

Responsabilidade Única: Adaptar dados do GA para LLM/UI sem modificar tsp.py

PROBLEMAS RESOLVIDO:
- tsp.py importa pygame (requer display gráfico, não funciona em web)
- tsp.py tem main() que abre janela Pygame
- Precisamos rodar GA SEM interface gráfica, para Streamlit

SOLUÇÃO:
- Import tardio do tsp (lazy import)
- Reutiliza funções públicas do tsp
- Executa AG sem pygame.display
- Retorna dados no mesmo formato que tsp usa
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def _import_tsp_module():
    """
    Importa tsp.py de forma TARDIA (lazy import).
    
    Por quê TARDIA? 
    - Se importássemos no topo, pygame.init() seria chamado na inicialização
    - pygame.init() falha em ambiente sem display (servidores, CI/CD)
    - Lazy import permite:
      * Imports bem-sucedidos até chamar essa função
      * Erros claros quando tentar usar GA em ambiente sem display
    
    Benefício adicional:
    - Streamlit pode importar este módulo sem inicializar pygame
    - pytest/testes funcionam mesmo sem X11
    """
    import tsp  # noqa: WPS433 - import tardio proposital

    return tsp


def build_route_results(
    population_size: int | None = None,
    generations: int | None = None,
    seed: int = 42,
) -> List[Dict]:
    """
    FUNÇÃO CORE: Executa Algoritmo Genético SEM interface gráfica.
    
    Esta é a ponte entre Streamlit e o GA implementado em tsp.py.
    
    FLUXO TÉCNICO:
    ===============
    1. Importa tsp de forma tardia
    2. Define seed para reprodutibilidade
    3. Carrega 20 cidades balanceadas do dataset
    4. Cria população inicial aleatória
    5. Evolui por N gerações
    6. Retorna melhor solução encontrada
    
    POPULAÇÃO E GERAÇÕES:
    =====================
    - Se population_size=None: usa padrão do tsp.py (90)
    - Se generations=None: usa padrão do tsp.py (180)
    - Para web: usar population_size=8, generations=10 para resposta rápida
    
    RETORNO - Lista de dicts com formato:
    =====================================
    [
      {
        'vehicle': {...dados do veículo...},
        'best_route': [cidades em ordem],
        'distance_km': 250.5,
        'demand': 130,
        'work_minutes': 480.0,
        'fitness': 800.0,
        'priority_penalty': -50.0,
        'penalty': 0.0,
        'total_cost': 250.0,
      },
      ... (um dict por veículo)
    ]
    
    Este formato é exatamente compatível com:
    - llm_integration.answer_route_question()
    - draw_functions (para visualização)
    - Relatórios em geral
    
    CUSTO COMPUTACIONAL:
    ====================
    - 8 população × 10 gerações = ~2-3 segundos
    - 30 população × 50 gerações = ~30-40 segundos
    - 90 população × 180 gerações = ~5-10 minutos (padrão do tsp.py)
    
    DETERMINISMO:
    ==============
    Com mesmo seed, sempre gera mesma solução.
    Importante para testes e reprodutibilidade acadêmica.
    """
    tsp = _import_tsp_module()

    # Mantém a execução reproduzível.
    tsp.random.seed(seed)

    all_cities, depot = tsp.build_city_objects()

    # Se o usuário não informar nada, usamos os mesmos padrões do projeto.
    effective_population = population_size or tsp.POPULATION_SIZE
    effective_generations = generations or tsp.N_GENERATIONS

    population = tsp.generate_random_population(all_cities, effective_population)

    final_results: List[Dict] | None = None

    for _ in range(effective_generations):
        population, _, merged_results, _, _ = tsp.evolve_global_population(population, depot)
        final_results = merged_results

    if not final_results:
        raise RuntimeError("Não foi possível montar os resultados das rotas.")

    return final_results


def summarize_route_results(route_results: List[Dict]) -> Dict:
    """Gera um pequeno resumo para exibir na sidebar do Streamlit."""
    if not route_results:
        return {
            "vehicles_used": 0,
            "total_stops": 0,
            "total_distance_km": 0.0,
            "total_demand": 0,
            "total_time_min": 0.0,
        }

    return {
        "vehicles_used": len(route_results),
        "total_stops": sum(len(item["best_route"]) for item in route_results),
        "total_distance_km": sum(item["distance_km"] for item in route_results),
        "total_demand": sum(item["demand"] for item in route_results),
        "total_time_min": sum(item["work_minutes"] for item in route_results),
    }


def list_example_questions() -> List[str]:
    """Centraliza exemplos de perguntas para a interface."""
    return [
        "Qual é a melhor rota do veículo 1?",
        "Qual veículo ficou com o menor percurso?",
        "Quais cidades críticas estão na operação?",
        "Gere um resumo das rotas do dia.",
        "Quais melhorias a LLM sugere?",
    ]
