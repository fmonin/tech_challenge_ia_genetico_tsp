from __future__ import annotations

from typing import Dict, List, Tuple


# Esse módulo tem uma única missão: montar o route_results
# reaproveitando a lógica já existente do projeto, sem mexer no tsp.py.
# A ideia é usar as funções públicas do próprio módulo tsp.


def _import_tsp_module():
    """Importa o módulo tsp de forma tardia.

    Eu deixei a importação dentro da função para evitar que a tela Streamlit
    tente importar tudo logo na inicialização sem necessidade.
    Isso também facilita mostrar mensagens de erro mais amigáveis.
    """
    import tsp  # noqa: WPS433 - import tardio proposital

    return tsp


def build_route_results(
    population_size: int | None = None,
    generations: int | None = None,
    seed: int = 42,
) -> List[Dict]:
    """Executa a lógica de otimização sem abrir a interface pygame.

    A estratégia aqui é simples:
    1. reaproveitar build_city_objects do tsp.py;
    2. reaproveitar generate_random_population do projeto já existente;
    3. reaproveitar evolve_global_population por algumas gerações;
    4. devolver o mesmo formato de dados que a LLM já espera.

    O retorno dessa função é uma lista de dicionários no mesmo formato do
    final_results do tsp.py, compatível com answer_route_question(...).
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
