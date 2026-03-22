"""
MÓDULO: llm_integration.py
===========================
Integração com GPT da OpenAI para análise inteligente de rotas VRP.

VISÃO GERAL:
Este módulo conecta os resultados do Algoritmo Genético (cálculos puros)
com uma Large Language Model (LLM) para gerar análises humanizadas,
relatórios operacionais e recomendações estratégicas.

FLUXO DE DADOS:
===============
1. GA gera melhor solução de rotas (vários km, custos, etc)
2. build_route_context() formata dados em texto estruturado
3. Texto é enviado como "contexto" para a LLM
4. LLM processa com prompt específico
5. call_llm() faz request HTTP para OpenAI
6. Resposta retorna como texto humanizado

CONCEITOS FUNDAMENTAIS:
======================

A) TOKENS:
   - Unidade básica de processamento em LLMs
   - Não é caractere! Um token ≈ 4 caracteres em inglês
   - Exemplos de tokenização:
     
     "Qual é a melhor rota?" → tokeniza como:
     ["Qual", "é", "a", "melhor", "rota", "?"]
     = 6 tokens
     
     "190 km" → pode ser 1 ou 2 tokens (modelo decide)
     
     Português tem menor ratio: 1 token ≈ 3-3.5 caracteres
   
   - CUSTO: API OpenAI cobra por token processado
   - Input tokens (dados enviados): custo X
   - Output tokens (resposta): custo Y (tipicamente 2X a 3X mais caro)
   - Modelo gpt-4o-mini: ~0,15 USD por 1M de input tokens
   
   E.g., análise de 50 cidades = ~1500 tokens = $0,0002 por requisição

B) PROMPT ENGINEERING:
   - Instruções claras determinam qualidade da resposta
   - 2 partes: SYSTEM (papel) + USER (tarefa + dados)
   - Bom prompt > modelo poderoso (para tarefas específicas)
   - Exemplos ajudam: "por favor seja conciso" gera diferença enorme

C) CONTEXT WINDOW:
   - Tamanho máximo de texto que a LLM processa por vez
   - gpt-4o-mini: 128K tokens (≈ 512K caracteres)
   - Nossos prompts: <<1% desse limite
   - Importa: não há limite prático para nossas análises

D) CHAMADAS HTTP:
   - API OpenAI usa REST (HTTP POST)
   - Autenticação: Bearer token (API key)
   - Request format: JSON com model, messages, parâmetros
   - Response: JSON com choices[0].message.content

SEGURANÇA E BOAS PRÁTICAS:
==========================
- NUNCA commitar API keys em código (use variáveis de ambiente)
- Rate limits: OpenAI limita a ~3,500 requests/minuto
- Timeouts: configurado em 60 segundos
- Error handling: trata quota esgotada e erros HTTP graciosamente
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib import error, request

# ==================================================================
# SEÇÃO 1: CONFIGURAÇÃO E CARREGAMENTO DE API KEY
# ==================================================================

def _load_local_env_file() -> None:
    """
    Carrega variáveis de ambiente de arquivo .env local.
    
    Prática comum em desenvolvimento: manter .env no .gitignore
    para não expor secrets no repositório público.
    
    Formato do arquivo .env:
    ===========================
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
    OPENAI_MODEL=gpt-4o-mini
    
    Segurança:
    - Não usa eval() (perigoso!)
    - Ignora comentários (#) e linhas vazias
    - Não sobrescreve variáveis já definidas (permite override via sistema)
    """
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

# Configuração do modelo
OPENAI_MODEL = "gpt-4o-mini"  # Mini é mais barato, ótimo para tarefas estruturadas

# URL base da API (versão v1)
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Arquivo de histórico para relatórios semanais
HISTORY_FILE = "llm_reports_history.json"


# ==================================================================
# SEÇÃO 2: FORMATAÇÃO E CONSTRUÇÃO DE CONTEXTO
# ==================================================================

def _priority_text(priority: str) -> str:
    """
    Mapeia código interno de prioridade para texto legível.
    Util para exibir em relatórios humanizados.
    """
    if priority == "critica":
        return "Crítica"
    if priority == "regular":
        return "Regular"
    if priority == "depot":
        return "Depósito"
    return str(priority)


def _build_totals(route_results: List[Dict]) -> Dict:
    """
    Agrega métricas de todas as rotas em totais operacionais.
    
    Entrada: Lista com dados de cada veículo (melhor rota encontrada)
    Saída: Dicionário com somas totais
    
    Métricas agregadas:
    - Veículos usados: contagem de rota
    - Distância: soma de todos os km
    - Demanda: soma de todas as unidades entregues
    - Tempo: soma de minutos de trabalho
    - Custo: soma de custos operacionais
    - Cidades críticas: lista única de cidades com prioridade
    """
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


def build_route_context(route_results: List[Dict]) -> str:
    """
    FUNÇÃO CRÍTICA: Transforma dados estruturados em contexto texto.
    
    Esta função converte os resultados do Algoritmo Genético (dicts Python)
    em PLASMA DE CONTEXTO: texto estruturado que a LLM consegue processar.
    
    ESTRUTURA DO CONTEXTO:
    =======================
    1. Seção de DADOS GERAIS
       - Métricas totais de toda operação
       - Sumário executivo em 1 linha
    
    2. Seção por VEÍCULO
       - Qual é o veículo (label, capacidade, etc)
       - Qual é a sequência de cidades (com ordem crítica)
       - Métricas específicas desse veículo
    
    IMPORTÂNCIA DE BOM CONTEXTO:
    ============================
    - Garbage in, garbage out: contexto ruim = resposta ruim
    - LLM não "entende" estrutura de dados Python
    - Precisa de TEXTO ESTRUTURADO mas LEGÍVEL
    - Muito detalhado: caro (tokens)
    - Muito genérico: perde informação
    - EQUILIBRIO: estrutura clara + dados essenciais
    
    USO:
    Contexto é injetado em prompts como:
    
    "Baseado no contexto abaixo, responda à pergunta:
     
     {contexto_de_rotas_aqui}
     
     PERGUNTA: Qual é a melhor rota?"
    
    ESTIMATIVA DE TOKENS:
    ≈ 20 cidades + 3 veículos = 1000-1500 tokens de contexto
    + 200 tokens de prompt = ≈ 1700 tokens total por requisição
    """
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

    # Para cada veículo, monta uma seção de descrição detalhada
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


# ==================================================================
# SEÇÃO 3: CHAMADA PARA API DA OPENAI
# ==================================================================

def _validate_config() -> None:
    """
    Valida se a API key está presente antes de fazer chamadas.
    
    Erro com mensagem clara é melhor que erro criptografado depois.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Preencha a variável de ambiente OPENAI_API_KEY antes de rodar o projeto."
        )


def call_llm(prompt: str) -> str:
    """
    FUNÇÃO CORE: Chamada HTTP para API da OpenAI.
    
    FLUXO TÉCNICO:
    ================
    
    1. VALIDAÇÃO
       - Checa se API key está configurada
       - Erro precoce, fails fast
    
    2. MONTAGEM DO PAYLOAD (JSON)
       Estrutura:
       {
         "model": "gpt-4o-mini",
         "messages": [
           {"role": "system", "content": "Você é um especialista..."},
           {"role": "user", "content": "{prompt}"}
         ],
         "temperature": 0.7 (padrão OpenAI)
       }
       
       Notas sobre parâmetros:
       - "model": versão do GPT a usar
       - "messages": conversa (sistema fala primeiro, depois usuário)
       - "role": quem está falando (system, user, assistant)
       - "temperature": criatividade (0=determinístico, 1=criativo)
    
    3. CODIFICAÇÃO
       - Converte dict Python para JSON string
       - Codifica como utf-8 bytes para HTTP
    
    4. CONSTRUÇÃO DO REQUEST HTTP
       - Método: POST (enviamos dados)
       - URL: endpoint da OpenAI
       - Headers:
         * Content-Type: application/json (tipo de conteúdo)
         * Authorization: Bearer {API_KEY} (autenticação)
       
       Headers explicados:
       ====================================================================
       Content-Type: application/json
       └─ Diz ao servidor que ele vai receber JSON no corpo da requisição
       
       Authorization: Bearer sk-proj-xxxxx
       └─ Token de autenticação. Formato "Bearer {token}" é padrão OAuth2.
       └─ API Key começa com "sk-proj-" (Org keys começam com "org-")
       └─ JAMAIS deve ser incluída em logs ou compartilhada.
    
    5. REQUISIÇÃO HTTP
       - timeout=60: aguarda até 60 segundos pela resposta
       - Conexão HTTPs (segura, criptografada)
       - request.urlopen() é do built-in urllib (sem dependências extras)
    
    RESPOSTA:
    ===========
    Response é JSON:
    {
      "id": "chatcmpl-...",
      "object": "chat.completion",
      "model": "gpt-4o-mini-...",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "Texto da resposta aqui..."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 1234,
        "completion_tokens": 567,
        "total_tokens": 1801
      }
    }
    
    Extrai o conteúdo: response["choices"][0]["message"]["content"]
    
    TRATAMENTO DE ERROS:
    ====================
    - HTTPError 401: API key inválida ou expirada
    - HTTPError 429: Rate limit atingido (muitas requisições)
    - HTTPError 401 com "insufficient_quota": Conta sem créditos
    - Timeout: Sem resposta em 60 segundos (rede lenta? Service down?)
    
    SEGURANÇA IMPLEMENTADA:
    =======================
    - API key não é logada ou exibida
    - HTTPS (transmission criptografada)
    - Timeout previne hanging infinito
    - Error messages não expõem detalhes internos
    """
    _validate_config()

    # MONTAGEM DO PAYLOAD
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

    # CODIFICAÇÃO PARA HTTP
    data = json.dumps(payload).encode("utf-8")
    
    # MONTAGEM DO REQUEST
    req = request.Request(
        OPENAI_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )

    # EXECUÇÃO DA REQUISIÇÃO
    try:
        with request.urlopen(req, timeout=60) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            return response_data["choices"][0]["message"]["content"]
    except error.HTTPError as exc:
        # Tenta extrair detalhes de erro do JSON da resposta
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except:
            detail = str(exc)
        raise RuntimeError(f"Erro HTTP ao chamar a LLM: {detail}") from exc


# ==================================================================
# SEÇÃO 4: FUNÇÕES DE RELATÓRIO (usam call_llm internamente)
# ==================================================================

def append_history_entry(route_results: List[Dict]) -> None:
    """
    Mantém histórico de execuções para análise semanal.
    
    Estrutura do histórico:
    [
      {
        "created_at": "2026-03-22T15:30:00",
        "distance": 898.5,
        "demand": 246,
        "time": 1500.0,
        "cost": 1050.0,
        "critical_cities": ["São Paulo", "Santos"]
      },
      ...
    ]
    
    Uso: Arquivo JSON simples, fácil de consultar e analisar.
    """
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


def generate_daily_report(route_results: List[Dict]) -> str:
    """
    Gera relatório executivo diário com insights sobre desempenho.
    
    Prompt engineering aqui:
    - System: especialista em logística
    - User: contexto de rotas + instrução específica
    - Instruções adicionais no próprio prompt (keeps LLM focused)
    """
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


def generate_driver_instructions(route_results: List[Dict]) -> str:
    """
    Gera instruções operacionais por veículo para os motoristas.
    
    Exemplo de saída:
    # Veículo 1 - Pequeno
    1. Sair do CD em direção a São Paulo
    2. Entregar em Santos (CRÍTICA - 14:30)
    ...
    """
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


def generate_process_improvements(route_results: List[Dict]) -> str:
    """
    Identifica oportunidades de otimização com análise inteligente.
    """
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


def generate_weekly_report() -> str:
    """
    Compila análise de tendências dos últimos 7 dias.
    
    Lê arquivo de histórico e faz comparative analysis:
    -Qual foi o melhor dia?
    - Qual métrica piorou?
    - Qual estratégia funcionou?
    """
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Sem dados históricos suficientes para gerar relatório semanal."

    if len(history) < 3:
        return "São necessários pelo menos 3 dias de dados para gerar relatório semanal."

    # Pega os últimos 7 dias
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


def answer_route_question(route_results: List[Dict], question: str) -> str:
    """
    Função interativa: responde perguntas do usuário sobre as rotas.
    
    Exemplos de perguntas:
    - "Qual é a melhor rota do veículo 1?"
    - "Qual veículo ficou com o menor percurso?"
    - "Há alguma cidade crítica não atendida?"
    
    A LLM analisa os dados e dá respostas específicas.
    """
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
