# VRP simplificado com Algoritmo Genético

Este projeto adapta uma base de TSP para uma versão simplificada de **VRP (Vehicle Routing Problem)**, com foco em uma entrega acadêmica de pós-graduação.

A ideia do sistema é otimizar rotas de entrega de medicamentos e insumos usando **Algoritmo Genético**, considerando:
- 90 cidades reais da região de São Paulo
- prioridade de entrega
- capacidade dos veículos
- autonomia dos veículos
- até 3 veículos disponíveis
- visualização das rotas e da evolução do fitness

## Objetivo da atividade

O trabalho parte do problema do caixeiro viajante e evolui para um cenário mais realista de roteamento.

O algoritmo tenta encontrar boas sequências de atendimento das cidades e, depois, divide essa sequência em viagens menores para verificar quais veículos conseguem executar cada trecho.

## Estrutura dos arquivos

### `genetic_algorithm.py`
Arquivo com a lógica principal do algoritmo genético.

Funções principais:
- geração da população inicial
- crossover do tipo OX
- mutação por troca ou inversão
- seleção por torneio
- ordenação da população
- melhoria local com 2-opt
- criação da próxima geração

### `tsp.py`
Arquivo principal do projeto.

Ele contém:
- cadastro das 90 cidades
- definição dos 3 veículos
- cálculo de distância geográfica
- lógica para dividir a rota em até 3 viagens
- seleção automática dos veículos
- função fitness
- laço principal com animação e exibição no terminal

### `draw_functions.py`
Arquivo de desenho da interface em Pygame.

Ele mostra:
- mapa com as cidades
- rota em teste em cinza
- melhor rota em azul
- gráfico da evolução do fitness
- painel com distância, penalidades e veículos escolhidos

### `benchmark_att48.py`
Permanece como benchmark auxiliar da estrutura original.

### `demo_crossover.py`
Demonstra o crossover usado no projeto.

### `demo_mutation.py`
Demonstra a mutação usada no projeto.

## Como o algoritmo funciona

1. O sistema cria várias rotas aleatórias com as cidades clientes.
2. Cada rota recebe uma avaliação.
3. A rota é dividida em viagens menores, respeitando a viabilidade dos veículos.
4. O sistema testa qual combinação de veículos atende melhor essas viagens.
5. A fitness leva em conta distância, custo, prioridade e penalidades.
6. As melhores rotas são mantidas.
7. Novas rotas são criadas com crossover e mutação.
8. O processo se repete por várias gerações.

## Veículos usados

O projeto trabalha com 3 tipos de veículos:

### Veículo Pequeno
- capacidade: 45
- autonomia: 240 km
- custo operacional menor

### Veículo Médio
- capacidade: 90
- autonomia: 420 km
- custo intermediário

### Veículo Grande
- capacidade: 160
- autonomia: 700 km
- custo maior, mas suporta cargas maiores

## Função fitness

A fitness foi pensada de forma simples, mas coerente para a apresentação.

Ela considera:
- distância total percorrida
- custo de uso dos veículos
- penalidade quando a solução exige mais de 3 viagens
- penalidade para entregas críticas muito tarde
- recompensa pequena quando cidades críticas aparecem mais cedo

**Quanto menor a fitness, melhor a solução.**

## Dependências

Instale as bibliotecas abaixo:

```bash
pip install pygame
```

## Como executar

No terminal, dentro da pasta do projeto:

```bash
python tsp.py
```

## Interface Streamlit (chat com LLM)

Se quiser usar a interface web com perguntas sobre as rotas:

```bash
pip install -r requirements_ui.txt
streamlit run app_streamlit.py
```

No PowerShell, se precisar chamar o Python da venv diretamente:

```powershell
& ".venv/Scripts/python.exe" -m streamlit run app_streamlit.py
```

Observação: o projeto já inclui o arquivo `.streamlit/config.toml` com `gatherUsageStats = false`, evitando o prompt inicial de e-mail do Streamlit.

## O que aparece durante a execução

No terminal:
- geração atual
- fitness da melhor solução
- distância total da melhor solução
- veículos escolhidos

Na tela:
- cidades do mapa
- rota em teste em cinza
- melhor rota em azul
- gráfico da evolução do fitness
- painel com métricas da melhor solução

## Observação importante

As cidades são reais e da região de São Paulo, mas as coordenadas foram colocadas de forma **aproximada** para fins didáticos e de visualização.

## Melhorias futuras

Algumas ideias para evoluir o projeto:
- usar mapa real com biblioteca geográfica
- separar depósito, hospitais e farmácias por tipo
- colocar janela de tempo
- colocar mais de um veículo por tipo
- exportar resultado para CSV ou PDF
- comparar com outras heurísticas
