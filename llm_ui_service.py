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


# Esse módulo é a camada de serviço da aplicação.
# Ele evita que a interface Streamlit conheça detalhes da integração com a LLM
# e também evita espalhar a regra de negócio pela UI.


class RouteLLMService:
    """Orquestra a carga das rotas e as chamadas já existentes da LLM."""

    def __init__(
        self,
        population_size: int | None = None,
        generations: int | None = None,
        seed: int = 42,
    ) -> None:
        # Defaults mais leves para UI web; podem ser sobrescritos por variável de ambiente.
        default_population = int(os.getenv("STREAMLIT_GA_POPULATION", "30"))
        default_generations = int(os.getenv("STREAMLIT_GA_GENERATIONS", "45"))

        self.population_size = population_size or default_population
        self.generations = generations or default_generations
        self.seed = seed
        self._route_results: List[Dict] | None = None

    def load_route_results(self, force_reload: bool = False) -> List[Dict]:
        """Carrega o contexto das rotas apenas quando necessário.

        O cache em memória deixa a experiência mais rápida durante a sessão.
        """
        if self._route_results is None or force_reload:
            self._route_results = build_route_results(
                population_size=self.population_size,
                generations=self.generations,
                seed=self.seed,
            )
        return self._route_results

    def get_summary(self) -> Dict:
        route_results = self.load_route_results()
        return summarize_route_results(route_results)

    def ask_question(self, question: str) -> str:
        """Envia uma pergunta do usuário para a função já existente do projeto."""
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            raise ValueError("Digite uma pergunta válida antes de enviar.")

        route_results = self.load_route_results()
        return answer_route_question(route_results, cleaned_question)

    def get_daily_report(self) -> str:
        route_results = self.load_route_results()
        return generate_daily_report(route_results)

    def get_driver_instructions(self) -> str:
        route_results = self.load_route_results()
        return generate_driver_instructions(route_results)

    def get_process_improvements(self) -> str:
        route_results = self.load_route_results()
        return generate_process_improvements(route_results)

    def get_weekly_report(self) -> str:
        # Essa função já existe no projeto e consulta o histórico salvo.
        return generate_weekly_report()
