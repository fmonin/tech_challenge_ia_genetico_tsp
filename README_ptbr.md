# Projeto TSP com Algoritmo Genético

Este projeto resolve o problema do caixeiro viajante (TSP) usando Algoritmo Genético e mostra a evolução das rotas com visualização em Pygame.

A ideia principal é simples: criar várias rotas aleatórias, medir qual rota é melhor, misturar as melhores rotas, aplicar pequenas alterações e repetir isso por várias gerações até encontrar uma solução cada vez melhor.

## Objetivo do projeto

O objetivo é encontrar uma rota curta para visitar todas as cidades e voltar ao ponto inicial.

Neste repositório, o foco está em:
- gerar uma população inicial de rotas
- calcular a fitness de cada rota
- selecionar bons pais
- fazer crossover entre rotas
- aplicar mutação
- melhorar rotas com 2-opt
- mostrar a evolução da melhor solução na tela

## Estrutura dos arquivos

### `genetic_algorithm.py`
Arquivo com a lógica principal do algoritmo genético.

Funções principais:
- `generate_random_population`: cria várias rotas aleatórias
- `calculate_distance`: calcula a distância entre dois pontos
- `calculate_fitness`: mede o tamanho total da rota
- `order_crossover`: mistura duas rotas sem repetir cidades
- `mutate`: faz pequenas mudanças na rota
- `two_opt_improve`: tenta melhorar localmente a rota
- `tournament_selection`: escolhe pais melhores
- `sort_population`: ordena a população da melhor para a pior

### `tsp.py`
Arquivo principal da aplicação.

Ele:
- carrega as cidades
- cria a população inicial
- roda o algoritmo geração por geração
- mostra a melhor rota em azul
- mostra uma rota concorrente em cinza
- exibe o valor da fitness
- desenha o gráfico da evolução do algoritmo

### `draw_functions.py`
Arquivo com as funções de desenho da interface.

Ele desenha:
- o gráfico da fitness
- as cidades
- as rotas
- os textos informativos na tela

### `benchmark_att48.py`
Contém um conjunto clássico de cidades para testar o algoritmo.

### Arquivos comentados
Também foram geradas versões comentadas com explicações mais simples:
- `genetic_algorithm_comentado.py`
- `tsp_comentado.py`
- `draw_functions_comentado.py`

## Como executar

### 1. Criar ambiente virtual
No Windows:

```bash
python -m venv venv
```

### 2. Ativar o ambiente virtual
No PowerShell:

```bash
venv\Scripts\Activate.ps1
```

No CMD:

```bash
venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install pygame matplotlib numpy
```

### 4. Rodar o projeto

```bash
python tsp.py
```

Se quiser estudar a versão com explicações mais didáticas, você também pode abrir os arquivos comentados.

## Controles durante a execução

- `Q`: fecha o programa
- `P`: pausa ou continua
- `Espaço`: avança uma geração quando estiver pausado
- `+` ou `=`: aumenta a mutação
- `-` ou `_`: diminui a mutação

## Como o algoritmo funciona

1. O sistema cria várias rotas aleatórias.
2. Cada rota recebe uma fitness.
3. As melhores rotas ficam nas primeiras posições.
4. O melhor indivíduo é preservado.
5. Novos filhos são criados com crossover.
6. Esses filhos podem sofrer mutação.
7. Depois disso, o 2-opt tenta melhorar a rota.
8. O processo se repete até encontrar resultados melhores.

## O que significa fitness neste projeto

Aqui a fitness é a soma total da distância da rota.

Então:
- fitness menor = rota melhor
- fitness maior = rota pior

## Melhorias futuras

Este projeto já funciona bem como base de estudo, mas pode ser melhorado com:
- cidades reais com latitude e longitude
- múltiplos veículos
- restrição de capacidade
- restrição de autonomia
- prioridade de entregas
- uso de mapa real
- animação mais detalhada da evolução da população
- exportação de resultados

## Observação importante

O código atual está estruturado como um TSP clássico com visualização. Para transformar em um VRP com 3 veículos, capacidade, prioridade e cidades reais próximas de São Paulo, a estrutura já serve como base, mas será necessário adaptar a representação genética e a função de fitness.

## Autor

Projeto base usado para estudo e evolução prática em Algoritmos Genéticos aplicados ao problema do caixeiro viajante.
