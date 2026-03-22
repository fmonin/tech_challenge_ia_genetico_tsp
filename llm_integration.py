import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib import error, request

# ==========================================================
# CONFIGURACAO DA LLM
# ==========================================================
# NÃO COMMITAR CHAVES em código. Use variável de ambiente:
# export OPENAI_API_KEY="sua_chave"  (Linux/macOS) ou
# setx OPENAI_API_KEY "sua_chave" (Windows PowerShell)


def _load_local_env_file() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Não sobrescreve variáveis já definidas no ambiente do sistema.
            os.environ.setdefault(key, value)


_load_local_env_file()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Modelo que sera usado.
OPENAI_MODEL = "gpt-4o-mini"

# URL da API.
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Arquivo simples para guardar historico e ajudar no relatorio semanal.
HISTORY_FILE = "llm_reports_history.json"


# Aqui eu deixo o texto da prioridade mais amigavel.
def _priority_text(priority: str) -> str:
    if priority == "critica":
        return "Crítica"
    if priority == "regular":
        return "Regular"
    if priority == "depot":
        return "Depósito"
    return str(priority)


# Aqui eu somo os dados principais da operacao.
def _build_totals(route_results: List[Dict]) -> Dict:
    return {
        "vehicles_used": len(route_results),
        "distance": sum(item["distance_km"] for item in route_results),
        "demand": sum(item["demand"] for item in route_results),
        "time": sum(item["work_minutes"] for item in route_results),
        "cost": sum(item["total_cost"] for item in route_results),
        "priority": sum(item["priority_penalty"] for item in route_results),
        "penalty": sum(item["penalty"] for item in route_results),
        "fitness": sum(item["fitness"] for item in route_results),
        "critical_cities": [
            city["name"]
            for result in route_results
            for city in result["best_route"]
            if city["priority"] == "critica"
        ],
    }


# Aqui eu transformo a melhor solucao em texto para mandar para a LLM.
def build_route_context(route_results: List[Dict]) -> str:
    totals = _build_totals(route_results)
    lines = []

    lines.append("DADOS GERAIS DA OPERAÇÃO")
    lines.append(
        f"Veículos usados: {totals['vehicles_used']} | Distância total: {totals['distance']:.1f} km | "
        f"Demanda total: {totals['demand']} | Tempo total: {totals['time']:.1f} min | "
        f"Custo total: {totals['cost']:.1f} | Penalidade total: {totals['penalty']:.1f} | "
        f"Fitness total: {totals['fitness']:.2f}"
    )
    lines.append(
        "Cidades críticas atendidas: "
        + (", ".join(totals["critical_cities"]) if totals["critical_cities"] else "Nenhuma")
    )
    lines.append("")

    for result in route_results:
        vehicle = result["vehicle"]
        city_sequence = []
        critical_cities = []

        for city in result["best_route"]:
            city_text = (
                f"{city['name']} ({_priority_text(city['priority'])}, demanda {city['demand']})"
            )
            city_sequence.append(city_text)
            if city["priority"] == "critica":
                critical_cities.append(city["name"])

        lines.append(f"ROTA DO {vehicle['label'].upper()} - {vehicle['name'].upper()}")
        lines.append(
            "Sequência das cidades: "
            + (" -> ".join(city_sequence) if city_sequence else "Sem cidades")
        )
        lines.append(
            "Cidades críticas da rota: "
            + (", ".join(critical_cities) if critical_cities else "Nenhuma")
        )
        lines.append(
            f"Paradas: {len(result['best_route'])} | Distância: {result['distance_km']:.1f} km | "
            f"Demanda: {result['demand']} | Tempo: {result['work_minutes']:.1f} min | "
            f"Custo: {result['total_cost']:.1f} | Prioridade: {result['priority_penalty']:.1f} | "
            f"Penalidade: {result['penalty']:.1f} | Fitness: {result['fitness']:.2f}"
        )
        lines.append("")

    return "\n".join(lines)


# Aqui eu valido a configuracao da API antes da chamada.
def _validate_config() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Preencha a variável de ambiente OPENAI_API_KEY antes de rodar o projeto."
        )


# Aqui eu faço a chamada real para a LLM da OpenAI.
def call_llm(prompt: str) -> str:
    _validate_config()

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um assistente especialista em logística, rotas, entregas, "
                    "eficiência operacional e análise de dados de transporte."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OPENAI_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            return response_data["choices"][0]["message"]["content"]
    except error.HTTPError as exc:
        detail = json.loads(exc.read().decode("utf-8")) if exc.info().get_content_type() == "application/json" else str(exc)
        raise RuntimeError(f"Erro HTTP ao chamar a LLM: {detail}") from exc


# Aqui eu guardo o resultado da execucao no historico.
def append_history_entry(route_results: List[Dict]) -> None:
    totals = _build_totals(route_results)
    entry = {
        "created_at": datetime.now().isoformat(),
        "distance": round(totals["distance"], 2),
        "demand": totals["demand"],
        "time": round(totals["time"], 2),
        "cost": round(totals["cost"], 2),
        "priority": round(totals["priority"], 2),
        "penalty": round(totals["penalty"], 2),
        "fitness": round(totals["fitness"], 2),
        "critical_cities": totals["critical_cities"],
    }

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(entry)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# Aqui eu gero o relatorio diario.
def generate_daily_report(route_results: List[Dict]) -> str:
    context = build_route_context(route_results)
    prompt = f"""
Com base nos dados da operação de hoje, gere um relatório executivo conciso sobre o desempenho da frota.

{context}

INSTRUÇÕES:
- Foque nos pontos positivos e oportunidades de melhoria
- Seja objetivo e use dados para fundamentar suas observações
- Mantenha o relatório conciso (máximo 300 palavras)
- Destaque qualquer problema crítico identificado
- Sugira ações práticas para otimização

FORMATO:
- Use markdown para formatação
- Seções: Resumo Executivo, Desempenho por Veículo, Recomendações
"""
    return call_llm(prompt)


# Aqui eu gero as instrucoes para cada motorista.
def generate_driver_instructions(route_results: List[Dict]) -> str:
    context = build_route_context(route_results)
    prompt = f"""
Com base nas rotas definidas para hoje, gere instruções claras e práticas para cada motorista.

{context}

INSTRUÇÕES:
- Para cada veículo, forneça instruções específicas
- Inclua sequência de cidades, prioridades e cuidados especiais
- Considere tempo estimado e janelas de atendimento
- Destaque cidades críticas e restrições de cada rota
- Mantenha linguagem clara e direta

FORMATO:
- Use markdown com headers para cada veículo
- Liste cidades em ordem com horários estimados
- Inclua alertas para cidades críticas
"""
    return call_llm(prompt)


# Aqui eu identifico possiveis melhorias no processo.
def generate_process_improvements(route_results: List[Dict]) -> str:
    context = build_route_context(route_results)
    prompt = f"""
Analise os dados da operação atual e identifique oportunidades de melhoria no processo de roteirização.

{context}

INSTRUÇÕES:
- Identifique padrões ineficientes nas rotas
- Sugira redistribuição de carga entre veículos
- Considere balanceamento de tempo e distância
- Avalie impacto de cidades críticas no desempenho
- Foque em melhorias práticas e mensuráveis

FORMATO:
- Use markdown com seções claras
- Priorize melhorias de maior impacto
- Inclua estimativa de benefícios quando possível
- Liste ações específicas e responsáveis
"""
    return call_llm(prompt)


# Aqui eu compilo o relatorio semanal baseado no historico.
def generate_weekly_report() -> str:
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Sem dados históricos suficientes para gerar relatório semanal."

    if len(history) < 3:
        return "São necessários pelo menos 3 dias de dados para gerar relatório semanal."

    # Aqui eu pego so os ultimos 7 dias
    recent_entries = history[-7:] if len(history) >= 7 else history

    context = "\n".join([
        f"Dia {i+1}: {entry['distance']:.1f}km, {entry['demand']} demanda, "
        f"R${entry['cost']:.1f}, {len(entry['critical_cities'])} cidades críticas"
        for i, entry in enumerate(recent_entries)
    ])

    prompt = f"""
Com base nos dados dos últimos dias, gere um relatório semanal de performance.

DADOS SEMANAIS:
{context}

INSTRUÇÕES:
- Analise tendências de performance
- Identifique padrões de melhoria ou deterioração
- Compare eficiência entre dias
- Sugira ajustes estratégicos para a semana seguinte
- Foque em métricas chave: distância, custo, demanda crítica

FORMATO:
- Use markdown com seções: Tendências, Análise, Recomendações
- Inclua gráficos de tendência quando relevante
- Seja específico sobre causas e efeitos
"""
    return call_llm(prompt)


# Aqui eu respondo perguntas especificas sobre as rotas.
def answer_route_question(route_results: List[Dict], question: str) -> str:
    context = build_route_context(route_results)
    prompt = f"""
Com base nas rotas atuais, responda à seguinte pergunta do usuário de forma clara e precisa.

PERGUNTA: {question}

CONTEXTO DAS ROTAS:
{context}

INSTRUÇÕES:
- Seja específico e use dados das rotas para fundamentar a resposta
- Se a pergunta for sobre uma cidade específica, forneça detalhes completos
- Se for sobre otimização, sugira alternativas baseadas nos dados
- Mantenha resposta concisa mas informativa
- Use linguagem natural e amigável
"""
    return call_llm(prompt)
