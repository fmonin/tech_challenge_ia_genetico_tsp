# 📚 DOCUMENTAÇÃO COMPLETA - Metodologia e Comentários

## 🎯 Visão Geral do Projeto

Este projeto implementa um **Sistema Inteligente de Otimização de Rotas** combinando:
- **Algoritmo Genético (AG)** para otimização combinatória
- **Vehicle Routing Problem (VRP)** com múltiplas restrições
- **Large Language Model (LLM)** para análise humanizada de resultados

---

## 🧬 1. ALGORITMO GENÉTICO - Metodologia

### 1.1 Representação (Cromossoma)
```
Indivíduo = Lista de cidades em sequência
Exemplo: [São Paulo, Santos, Campinas, Sorocaba, Mogi das Cruzes]
```

**Por que esta representação?**
- Simples: cada permutação é uma solução válida
- Eficiente: fácil de calcular fitness
- Flexível: compatível com operadores genéticos padrão

### 1.2 População Inicial
- **Tamanho padrão (tsp.py):** 90 indivíduos
- **Tamanho web (Streamlit):** 8 indivíduos (resposta rápida)
- **Método:** random.sample() garante variedade

**Fórmula:**
```
população = [random.sample(cidades, len(cidades)) for i in range(90)]
```

### 1.3 Função de Avaliação (Fitness)
Minimiza uma composição de métricas:

```
fitness = distância_km 
          + custo_operacional 
          + penalidade_por_restrições
          - desconto_por_cidades_críticas
```

**Componentes:**
| Componente | Significado | Peso |
|-----------|-----------|------|
| `distance_km` | Distância percorrida | 1x |
| `operational_cost` | Custos de combustível + fixo | 1.0-1.28 (por veículo) |
| `priority_penalty` | Atraso em entregas críticas | 2.5x posição |
| `capacity_violation` | Carga acima da capacidade | 14x unidades extras |
| `distance_violation` | Distância acima do limite | 10x km extras |
| `time_violation` | Tempo acima do limite | 2.4x min extras |
| `critical_bonus` | Desconto por entregar críticas cedo | -8 a -18 (por veículo) |

**Pseudocódigo:**
```python
def fitness(rota, veiculo):
  distancia = haversine(rota)
  custo = distancia * op_cost + fixed_cost
  penalidades = (violação_capacidade * 14 + 
                 violação_distancia * 10 +
                 violação_tempo * 2.4)
  bonus = num_criticas_atendidas * critical_bonus
  return distancia + custo + penalidades - bonus
```

### 1.4 Operadores Genéticos

#### A) **SELEÇÃO: Tournament Selection (k=5 ou k=6)**

```
Para cada pai:
  1. Seleciona aleatoriamente k indivíduos
  2. Compara fitness de todos
  3. Retorna o melhor (menor fitness)
```

**Vantagem:** Evita elitismo excessivo (pior solução ainda tem 1/k de chance)

#### B) **CROSSOVER: Order Crossover (OX)**

```
Parent 1: [A, B, C, D, E, F]
Parent 2: [F, E, D, C, B, A]

Segmento selecionado: posições 1-3 (B, C, D)
Filho = [?, B, C, D, ?, ?]

Preenchimento a partir do Parent 2:
- F → filho = [F, B, C, D, ?, ?]
- E → filho = [F, B, C, D, E, ?]
- D (já existe) → pula
- C (já existe) → pula
- B (já existe) → pula
- A → filho = [F, B, C, D, E, A]

Resultado: [F, B, C, D, E, A]
```

**Por que OX é melhor?**
- Mantém "clusters" de cidades próximas juntas
- Preserva ordem relativa (importante para time windows)
- Literamente aprovado por 30+ anos de literatura TSP

#### C) **MUTAÇÃO: Duas Estratégias (50/50)**

**Estratégia 1 - SWAP (exploração):**
```python
Troca 2 posições aleatórias
[A, B, C, D] → [D, B, C, A]  # trocou posições 0 e 3
```

**Estratégia 2 - REVERSÃO (exploração local):**
```python
Inverte um segmento (2-opt)
[A, B, C, D, E] → [A, C, B, D, E]  # inverteu segmento B-C
```

**Taxa de mutação:** 22% (MUTATION_PROBABILITY = 0.22)

#### D) **MELHORIA LOCAL: 2-OPT**

Remove cruzamentos em rotas:
```
Rotas com cruzamento X: [A→B] e [C→D] se cruzam
Inverte segmento: [A→D] e [C→B] não se cruzam
Resultado: rota mais curta (~30% de melhoria)
```

Executado em ~45% das gerações para não ficar lento.

### 1.5 FLUXO EVOLUTIVO (Pseudo-código)

```
FOR geração = 1 TO N_GENERATIONS:
  # Avaliar fitness de toda população
  fitness_scores = [calcular_fitness(ind) for ind in população]
  
  # Manter melhores (ELITISMO)
  população_ordenada = sort(população, fitness_scores)
  melhores = população_ordenada[:2]  # ELITISM = 2
  
  # Gerar nova população
  nova_população = [melhores]  # Começa com elite
  
  WHILE len(nova_população) < POPULATION_SIZE:
    # Selecionar pais
    pai1 = tournament_selection(população, fitness_scores, k=6)
    pai2 = tournament_selection(população, fitness_scores, k=6)
    
    # Reproduozir
    filho = order_crossover(pai1, pai2)
    filho = mutate(filho, probability=0.22)
    
    # Melhorar localmente
    if random() < 0.45:
      filho = two_opt_improve(filho)
    
    nova_população.append(filho)
  
  população = nova_população
  
RETORNAR população[0]  # Melhor indivíduo
```

### 1.6 Parâmetros Configuráveis

| Parâmetro | Descrição | Padrão | Recomendado Web |
|-----------|-----------|--------|-----------------|
| `POPULATION_SIZE` | Indivíduos por geração | 90 | 8-30 |
| `N_GENERATIONS` | Gerações evoluir | 180 | 10-50 |
| `MUTATION_PROBABILITY` | Taxa mutação | 0.22 | 0.20-0.25 |
| `TOURNAMENT_SIZE` | k para seleção | 6 | 5-8 |
| `ELITISM` | Melhores a preservar | 2 | 2 |

**Dica:** Para web responsivo, usar population_size=8, generations=10 (~2-3 segundos)

---

## 🤖 2. INTEGRAÇÃO COM LLM (OpenAI GPT)

### 2.1 Tokenização - Conceitos Fundamentais

#### O que são Tokens?
- **Definição:** Unidade mínima de processamento em LLMs
- **NÃO é caractere!** Um token ≈ 4 caracteres em inglês
- **Português é menor:** 1 token ≈ 3-3.5 caracteres

#### Exemplos de Tokenização
```
"Qual é a melhor rota?" 
→ ["Qual", "é", "a", "melhor", "rota", "?"]
→ 6 tokens

"190 km de distância"
→ ["190", "km", "de", "distância"]
→ 4 tokens

Números podem ser 1 ou 2 tokens dependendo do modelo
```

#### Custo em Tokens
**Modelo: gpt-4o-mini**
- Input: ~$0.00015 por 1K tokens ($0.00000015 por token)
- Output: ~$0.0006 por 1K tokens ($0.0000006 por token, ~3-4x mais caro)

**Exemplo de análise de rotas com 20 cidades:**
- Contexto formatado: ~1200-1500 tokens
- Prompt (instruções): ~200 tokens
- Custo total input: ≈ $0.0003
- Resposta estimada: ~300 tokens
- Custo total output: ≈ $0.0002
- **Custo por análise: ~$0.0005 (0.05 centavos)**

### 2.2 Fluxo de Dados para API OpenAI

#### PASSO 1: Construir Contexto (build_route_context)

```python
def build_route_context(route_results):
  # Formata dados estruturados em TEXTO LEGÍVEL para LLM
  
  contexto = """
  DADOS GERAIS DA OPERAÇÃO
  Veículos usados: 3 | Distância total: 950 km | Custo total: R$1200
  Cidades críticas atendidas: São Paulo, Santos, Campinas
  
  ROTA DO VEÍCULO 1 - PEQUENO
  Sequência das cidades: São Paulo -> Osasco -> Guarulhos -> ...
  Paradas: 7 | Distância: 280 km | Tempo: 450 min | Custo: R$350
  ...
  """
  
  return contexto
```

**Por que formatar em texto?**
- LLM não entende estruturas de dados Python
- Texto estruturado é legível + processável
- Melhor qualidade de resposta

#### PASSO 2: Montar Prompt

```python
prompt = f"""
Com base nas rotas atuais, responda à pergunta:

{contexto}

PERGUNTA: {pergunta_do_usuario}

INSTRUÇÕES:
- Seja específico e use dados
- Mantenha resposta concisa
- Use linguagem natural
"""
```

#### PASSO 3: Construir Requisição HTTP POST

```python
payload = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": "Você é especialista em logística..."
    },
    {
      "role": "user",
      "content": prompt
    }
  ]
}

request_json = json.dumps(payload).encode("utf-8")

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer sk-proj-{API_KEY}"
}

# Enviar via HTTPS (criptografado)
response = urllib.request.urlopen(
  request=Request(API_URL, data=request_json, headers=headers),
  timeout=60
)
```

#### PASSO 4: Processar Resposta

```python
response_data = json.loads(response.read())

resposta = response_data["choices"][0]["message"]["content"]
# Exemplo: "O Veículo 1 percorreu 280 km com 7 paradas..."

tokens_usados = response_data["usage"]
# {
#   "prompt_tokens": 1450,
#   "completion_tokens": 280,
#   "total_tokens": 1730
# }
```

### 2.3 Componentes da Solução LLM

#### A) System Message (Contexto do Assistente)
```
"Você é um assistente especialista em logística, rotas, entregas, 
eficiência operacional e análise de dados de transporte."
```

**Por que importante?**
- Define "papel" da LLM
- Torna respostas mais relevantes
- Reduz alucinações (respostas inventadas)

#### B) User Message (Pergunta + Dados)
```
Com base nas rotas atuais, responda:

{CONTEXTO_DETALHADO_AQUI}

PERGUNTA: {pergunta}

INSTRUÇÕES:
- Seja objetivo
- Use dados para fundamentar
- Mantenha respostas conc isas
```

#### C) Prompt Engineering - Boas Práticas

| Padrão | Efeito | Exemplo |
|--------|--------|---------|
| **Seja específico** | Reduz alucinações | "Liste 3 otimizações" em vez de "Sugira melhorias" |
| **Dê exemplos** | Melhora formato | "Formatar como: Item 1: descrição" |
| **Limite escopo** | Controla custo | "máximo 200 palavras" |
| **Peça estrutura** | Organiza resposta | "Use seções com headers" |
| **Cite contexto** | Aumenta precisão | "Dados de rotas: {contexto}" |

### 2.4 Tratamento de Erros da API

```python
try:
  response = urllib.request.urlopen(request, timeout=60)
except HTTPError as exc:
  if 401 in str(exc):
    # API key inválida ou expirada
    raise RuntimeError("API key inválida")
  
  if 429 in str(exc):
    # Rate limit (muitas requisições rápido)
    raise RuntimeError("Muitas requisições, aguarde")
  
  if "insufficient_quota" in str(exc):
    # Sem créditos na conta
    raise RuntimeError("Quota esgotada, adicione créditos")
  
  raise RuntimeError(f"Erro HTTP: {exc}")
```

### 2.5 Funções de Relatório

#### 1. **generate_daily_report()** - Relatório Especificado
```
Entrada: route_results (saída do GA)
Saída: Texto markdown com:
  - Resumo Executivo (KPIs do dia)
  - Desempenho por Veículo
  - Recomendações de Melhoria
```

#### 2. **generate_driver_instructions()** - Operacional
```
Entrada: route_results
Saída: Instruções por motorista:
  - Sequência de cidades
  - Horários estimados
  - Alertas para cidades críticas
```

#### 3. **answer_route_question()** - Interativa
```
Entrada: route_results + pergunta do usuário
Exemplo: "Qual veículo tinha a maior carga?"
Saída: Resposta humanizada baseada em dados
```

#### 4. **generate_weekly_report()** - Histórico
```
Entrada: 7 últimos dias (arquivo JSON)
Saída: Análise de tendências:
  - Qual dia foi melhor/pior
  - Tendências de performance
  - Recomendações estratégicas
```

---

## 🏗️ 3. ARQUITETURA DO PROJETO

```
┌─────────────────────────────────────┐
│  Interface: app_streamlit.py        │
│  - UI/UX em navegador               │
│  - Cache de sessão                  │
│  - Histórico de perguntas           │
└──────────┬──────────────────────────┘
           │
┌──────────▼──────────────────────────┐
│  Serviço: llm_ui_service.py         │
│  - Orquestra GA + LLM               │
│  - Cache em memória                 │
│  - API simples para Streamlit       │
└──────────┬──────────────────────────┘
           │
      ┌────┴────┐
      │          │
┌─────▼────┐ ┌──▼──────────────────┐
│ GA Core  │ │ LLM Integration      │
│ tsp.py   │ │ llm_integration.py   │
│          │ │ - Tokenização        │
│ - Genét. │ │ - API OpenAI         │
│ - Fitness│ │ - Prompt Engineering │
│ - Mut.   │ │ - Relatórios         │
│ - Vtação │ │                      │
└──────────┘ └──────────────────────┘
```

---

## 📊 4. CONFIGURAÇÕES RECOMENDADAS

### Para Trabalho Acadêmico (Qualidade)
```
POPULATION_SIZE = 90
N_GENERATIONS = 180
MUTATION_PROBABILITY = 0.22
TOURNAMENT_SIZE = 6
Tempo estimado: 5-10 minutos
```

### Para Demonstração em Streamlit (Responsividade)
```
POPULATION_SIZE = 8-15
N_GENERATIONS = 10-30
Tempo estimado: 2-5 segundos
Define via: STREAMLIT_GA_POPULATION=15, STREAMLIT_GA_GENERATIONS=30
```

### Para Testes Rápidos (Sanidade)
```
POPULATION_SIZE = 5
N_GENERATIONS = 3
Tempo estimado: <1 segundo
```

---

## 📈 5. MÉTRICAS E KPIs

| Métrica | Significado | Objetivo |
|---------|-----------|---------|
| **Fitness** | Qualidade geral da solução | Minimizar |
| **Distância Total** | km somados de todas rotas | Minimizar |
| **Custo Total** | R$ operacional | Minimizar |
| **Tempo Total** | Minutos de trabalho | Minimizar |
| **Demanda entregue** | Unidades totais | Maximizar |
| **Cidades críticas atendidas** | Count com prioridade alta | Maximizar |
| **Violações de capacidade** | Vezes que ultrapassou capacity | Minimizar = 0 |
| **Violações de distância** | Vezes que ultrapassou autonomia | Minimizar = 0 |

---

## 🚀 6. COMO EXECUTAR

### Terminal (Completo)
```bash
python tsp.py  # AG com interface Pygame + LLM
```

### Web (Streamlit)
```bash
streamlit run app_streamlit.py  # Interface web
# Acessa em http://localhost:8501
```

### Com Tuning de Performance
```bash
set STREAMLIT_GA_POPULATION=20
set STREAMLIT_GA_GENERATIONS=40
streamlit run app_streamlit.py
```

---

## 📚 Referências Acadêmicas

1. **Algoritmo Genético:** Holland (1975) - "Adaptation in Natural and Artificial Systems"
2. **Order Crossover (OX):** Davis (1985) - Padrão para TSP/VRP
3. **2-OPT:** Lin & Kernighan (1973) - Local search clássico
4. **Tournament Selection:** Goldberg & Deb (1991) - Seleção eficiente
5. **VRP:** Dantzig & Ramser (1959) - Problema original de roteamento
6. **LLM Integration:** OpenAI (2023) - GPT-4 API documentation

---

**Documentação compilada em Março 2026**  
**Projeto: IA Genético para TSP com LLM**  
**FIAP - Pós-Graduação em IA**
