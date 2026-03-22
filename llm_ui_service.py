from __future__ import annotations

import os
from typing import Dict, List

from llm_integration import (
    answer_route_question,
    generate_daily_report,
    generate_driver_instructions,
    generate_process_improvements,
    generate_weekly_report,
)

from route_context_loader import build_route_results, summarize_route_results


"""
MÓDULO: llm_ui_service.py
==========================
Camada de SERVIÇO para a interface web Streamlit.

Padrão de Design: SERVICE / ADAPTER LAYER
==========================================
Este módulo implementa o padrão Service Layer, que é uma arquitetura
intermediária entre a interface (UI) e a lógica complexa.

BENEFÍCIOS:
- Desacopla Streamlit da lógica de GA e LLM
- Esconde detalhes técnicos da UI
- Centraliza regra de negócio
- Facilita testes
- Permite trocar UI sem quebrar lógica

FLUXO:
app_streamlit.py
       ↓
RouteLLMService (este arquivo)
       ↓
build_route_results() ← GA de tsp.py
answer_route_question() ← LLM de llm_integration.py
"""


def _read_env_int(name: str, fallback: int) -> int:
    """
    Lê variável de ambiente como inteiro de forma segura.
    
    Uso:
    - STREAMLIT_GA_POPULATION=15
    - STREAMLIT_GA_GENERATIONS=20
    
    Benefício: permite tuning sem editar código
    """
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        value = int(raw)
        return value if value > 0 else fallback
    except ValueError:
        return fallback


def _read_env_optional_int(name: str) -> int | None:
    """
    Lê variável de ambiente como inteiro positivo opcional.

    Retorna None quando não existe/é inválida, permitindo fallback
    para os padrões oficiais definidos no módulo tsp.py.
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return None


class RouteLLMService:
    """
    SERVIÇO CENTRAL: Orquestra toda a pipeline de GA + LLM.
    
    Responsabilidades:
    1. Carregar e otimizar rotas (via GA)
    2. Cachear resultados em memória
    3. Fazer chamadas para LLM
    4. Expor API simples para Streamlit
    
    Instância é criada UMA VEZ e reutilizada durante a sessão
    (via @st.cache_resource), economizando tempo de cálculo.
    """

    def __init__(
        self,
        population_size: int | None = None,
        generations: int | None = None,
        seed: int = 42,
    ) -> None:
        """
        Inicializa o serviço com parâmetros do AG.
        
        Parameters:
        -----------
        population_size: int | None
            Tamanho da população do AG. Se None, usa padrão.
            Padrão leve para web: 8 indivíduos
            Pode ser sobrescrito por STREAMLIT_GA_POPULATION
        
        generations: int | None
            Quantas gerações evoluir. Se None, usa padrão.
            Padrão leve para web: 10 gerações
            Pode ser sobrescrito por STREAMLIT_GA_GENERATIONS
        
        seed: int
            Seed do RNG para reprodutibilidade.
            Com mesmo seed, AG gera mesma solução sempre.
        
        DEFAULTS OTIMIZADOS PARA UI WEB:
        =================================
        - 8 indivíduos: população pequena, convergência rápida
        - 10 gerações: com 8 indivíduos, ~2-3 segundos de execução
        - Valores podem ser aumentados se máquina for potente:
          $ setenv STREAMLIT_GA_POPULATION 30
          $ setenv STREAMLIT_GA_GENERATIONS 50
        """
        # Sem override explícito, usa os padrões do tsp.py para manter
        # consistência entre o fluxo Streamlit e o fluxo principal (TSP + LLM).
        env_population = _read_env_optional_int("STREAMLIT_GA_POPULATION")
        env_generations = _read_env_optional_int("STREAMLIT_GA_GENERATIONS")

        self.population_size = (
            population_size if population_size is not None else env_population
        )
        self.generations = generations if generations is not None else env_generations
        self.seed = seed
        self._route_results: List[Dict] | None = None

    def load_route_results(self, force_reload: bool = False) -> List[Dict]:
        """
        Carrega/executa o AG para otimizar rotas.
        
        CACHE INTELIGENTE:
        - Primeira chamada: executa AG completo
        - Chamadas subsequentes: devolver resultado em cache
        - force_reload=True: descarta cache, recalcula
        
        Performance:
        - Primeira execução: 3-5 segundos (depende de population_size/generations)
        - Chamadas posteriores: <1ms (cache)
        
        O cache é fundamental para experiência fluida no Streamlit,
        pois a página é re-executada a cada interação.
        """
        if self._route_results is None or force_reload:
            self._route_results = build_route_results(
                population_size=self.population_size,
                generations=self.generations,
                seed=self.seed,
            )
        return self._route_results

    def get_summary(self) -> Dict:
        """
        Retorna resumo agregado das rotas para exibir na interface.
        
        Útil para dashboard rápido sem mostrar todos os detalhes.
        
        Retorna:
        {
            'vehicles_used': 3,
            'total_stops': 20,
            'total_distance_km': 948.7,
            'total_demand': 246,
            'total_time_min': 1510.9
        }
        """
        route_results = self.load_route_results()
        return summarize_route_results(route_results)

    def ask_question(self, question: str) -> str:
        """
        Responde uma pergunta do usuário sobre as rotas com LLM.
        
        Fluxo:
        1. Valida pergunta é não-nula
        2. Carrega rotas (do cache ou executa AG)
        3. Envia para LLM com contexto das rotas
        4. Retorna resposta em texto humanizado
        
        Exemplo:
        >>> svc.ask_question("Qual veículo teve a maior distância?")
        "O Veículo 2 percorreu 520.3 km..."
        """
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            raise ValueError("Digite uma pergunta válida antes de enviar.")

        route_results = self.load_route_results()
        return answer_route_question(route_results, cleaned_question)

    def get_daily_report(self) -> str:
        """Gera relatório executivo do dia."""
        route_results = self.load_route_results()
        return generate_daily_report(route_results)

    def get_driver_instructions(self) -> str:
        """Gera instruções por veículo para motoristas."""
        route_results = self.load_route_results()
        return generate_driver_instructions(route_results)

    def get_process_improvements(self) -> str:
        """Gera recomendações de otimização."""
        route_results = self.load_route_results()
        return generate_process_improvements(route_results)

    def get_weekly_report(self) -> str:
        """
        Gera relatório semanal baseado no histórico.
        
        Nota: esta não precisa das rotas atuais, lê arquivo JSON.
        """
        return generate_weekly_report()
