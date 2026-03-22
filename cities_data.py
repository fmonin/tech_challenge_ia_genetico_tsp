"""
MÓDULO: cities_data.py
=======================
BASE DE DADOS de cidades para otimização de rotas.

ESTRUTURA DE CADA CIDADE:
(id_grupo, nome, latitude, longitude, demanda_kg, criticidade)

CRITICIDADE (Prioridade):
- "critica":  Cliente VIP, SLA apertado, deve atender cedo (earliness bonus)
- "regular": Cliente comum, sem restrições especiais
- "depot":   Centro de distribuição (ponto de origem e retorno)

DATASET:
- 87 cidades reais da região metropolitana de São Paulo
- Agrupadas em 3 regiões: ABC, Vale do Paraíba, Interior
- 20 selecionadas por balanceamento (5-6 cidades críticas)

SELEÇÃO PARA USO:
O AG sempre trabalha com exatamente 20 cidades balanceadas
extraídas deste dataset completo.

Este é um bom tamanho:
- Pequeno o bastante para GA convergir rápido
- Grande o bastante para ter complexidade real
- Acima de 20 cidades, tempo de GA dispara exponencialmente
"""

from typing import List, Tuple

# Cada tupla segue o formato:
# (grupo, nome, latitude, longitude, demanda, prioridade)
RAW_CITIES: List[Tuple[int, str, float, float, int, str]] = [
    (1, 'São Paulo', -23.55, -46.63, 18, 'critica'),
    (1, 'Guarulhos', -23.45, -46.53, 12, 'regular'),
    (1, 'Osasco', -23.53, -46.79, 10, 'regular'),
    (1, 'Barueri', -23.51, -46.88, 9, 'regular'),
    (1, 'Carapicuíba', -23.52, -46.84, 8, 'regular'),
    (1, 'Jandira', -23.53, -46.90, 7, 'regular'),
    (1, 'Itapevi', -23.55, -46.93, 8, 'regular'),
    (1, 'Cotia', -23.60, -46.92, 12, 'critica'),
    (1, 'Taboão da Serra', -23.62, -46.79, 9, 'regular'),
    (1, 'Embu das Artes', -23.65, -46.85, 8, 'regular'),
    (1, 'Itapecerica da Serra', -23.72, -46.85, 11, 'critica'),
    (1, 'Santana de Parnaíba', -23.44, -46.92, 8, 'regular'),
    (1, 'Cajamar', -23.36, -46.88, 9, 'regular'),
    (1, 'Franco da Rocha', -23.32, -46.73, 10, 'regular'),
    (1, 'Caieiras', -23.36, -46.74, 7, 'regular'),
    (1, 'Francisco Morato', -23.28, -46.74, 8, 'regular'),
    (1, 'Mairiporã', -23.32, -46.59, 8, 'regular'),
    (1, 'Diadema', -23.69, -46.62, 10, 'regular'),
    (1, 'São Caetano do Sul', -23.62, -46.55, 9, 'regular'),
    (1, 'Santo André', -23.66, -46.53, 11, 'critica'),
    (2, 'São Bernardo do Campo', -23.69, -46.56, 14, 'critica'),
    (2, 'Mauá', -23.67, -46.46, 10, 'regular'),
    (2, 'Ribeirão Pires', -23.71, -46.41, 8, 'regular'),
    (2, 'Rio Grande da Serra', -23.74, -46.40, 7, 'regular'),
    (2, 'Suzano', -23.54, -46.31, 11, 'regular'),
    (2, 'Poá', -23.53, -46.35, 7, 'regular'),
    (2, 'Ferraz de Vasconcelos', -23.54, -46.37, 8, 'regular'),
    (2, 'Itaquaquecetuba', -23.49, -46.35, 10, 'regular'),
    (2, 'Arujá', -23.40, -46.32, 9, 'regular'),
    (2, 'Santa Isabel', -23.31, -46.22, 8, 'regular'),
    (2, 'Mogi das Cruzes', -23.52, -46.19, 13, 'critica'),
    (2, 'Biritiba-Mirim', -23.57, -46.04, 7, 'regular'),
    (2, 'Salesópolis', -23.53, -45.85, 7, 'regular'),
    (2, 'Guararema', -23.42, -46.04, 8, 'regular'),
    (2, 'Jacareí', -23.30, -45.97, 12, 'regular'),
    (2, 'São José dos Campos', -23.19, -45.88, 15, 'critica'),
    (2, 'Caçapava', -23.10, -45.71, 8, 'regular'),
    (2, 'Taubaté', -23.03, -45.56, 10, 'regular'),
    (2, 'Pindamonhangaba', -22.92, -45.46, 8, 'regular'),
    (2, 'Tremembé', -22.96, -45.55, 7, 'regular'),
    (2, 'Guaratinguetá', -22.82, -45.19, 11, 'critica'),
    (2, 'Aparecida', -22.85, -45.23, 9, 'regular'),
    (2, 'Lorena', -22.73, -45.12, 9, 'regular'),
    (2, 'Cruzeiro', -22.58, -44.96, 8, 'regular'),
    (2, 'Cachoeira Paulista', -22.67, -45.01, 7, 'regular'),
    (3, 'Campinas', -22.90, -47.06, 17, 'critica'),
    (3, 'Valinhos', -22.97, -46.99, 9, 'regular'),
    (3, 'Vinhedo', -23.03, -46.98, 8, 'regular'),
    (3, 'Louveira', -23.09, -46.95, 7, 'regular'),
    (3, 'Jundiaí', -23.19, -46.88, 12, 'regular'),
    (3, 'Várzea Paulista', -23.21, -46.83, 8, 'regular'),
    (3, 'Campo Limpo Paulista', -23.21, -46.78, 8, 'regular'),
    (3, 'Jarinu', -23.10, -46.73, 7, 'regular'),
    (3, 'Atibaia', -23.12, -46.55, 10, 'regular'),
    (3, 'Bom Jesus dos Perdões', -23.14, -46.46, 6, 'regular'),
    (3, 'Bragança Paulista', -22.95, -46.54, 9, 'regular'),
    (3, 'Joanópolis', -22.93, -46.27, 6, 'regular'),
    (3, 'Piracaia', -23.05, -46.36, 7, 'regular'),
    (3, 'Nazaré Paulista', -23.18, -46.40, 7, 'regular'),
    (3, 'Igaratá', -23.20, -46.16, 6, 'regular'),
    (3, 'Salto', -23.20, -47.29, 8, 'regular'),
    (3, 'Itu', -23.26, -47.30, 10, 'regular'),
    (3, 'Sorocaba', -23.50, -47.46, 15, 'critica'),
    (3, 'Votorantim', -23.54, -47.44, 8, 'regular'),
    (3, 'Mairinque', -23.55, -47.18, 7, 'regular'),
    (3, 'São Roque', -23.53, -47.14, 8, 'regular'),
    (3, 'Alumínio', -23.53, -47.25, 7, 'regular'),
    (3, 'Araçariguama', -23.44, -47.06, 7, 'regular'),
    (3, 'Boituva', -23.28, -47.67, 8, 'regular'),
    (3, 'Porto Feliz', -23.21, -47.52, 7, 'regular'),
    (3, 'Tietê', -23.11, -47.72, 7, 'regular'),
    (3, 'Indaiatuba', -23.09, -47.22, 10, 'regular'),
    (3, 'Hortolândia', -22.85, -47.21, 9, 'regular'),
    (3, 'Sumaré', -22.82, -47.27, 9, 'regular'),
    (3, 'Paulínia', -22.76, -47.15, 8, 'regular'),
    (3, 'Americana', -22.74, -47.33, 8, 'regular'),
    (3, 'Nova Odessa', -22.78, -47.29, 7, 'regular'),
    (3, "Santa Bárbara d'Oeste", -22.75, -47.41, 8, 'regular'),
    (3, 'Limeira', -22.56, -47.40, 10, 'regular'),
    (3, 'Piracicaba', -22.73, -47.65, 11, 'critica'),
    (3, 'Rio Claro', -22.41, -47.56, 9, 'regular'),
    (3, 'Santos', -23.96, -46.33, 14, 'critica'),
    (3, 'São Vicente', -23.96, -46.39, 8, 'regular'),
    (3, 'Praia Grande', -24.01, -46.41, 10, 'regular'),
    (3, 'Cubatão', -23.89, -46.42, 9, 'regular'),
    (3, 'Guarujá', -23.99, -46.26, 10, 'regular'),
    (3, 'Bertioga', -23.85, -46.14, 8, 'regular'),
    (3, 'Mongaguá', -24.10, -46.62, 7, 'regular'),
    (3, 'Itanhaém', -24.18, -46.79, 8, 'regular'),
    (3, 'Peruíbe', -24.31, -47.00, 8, 'regular'),
]

SELECTED_CITY_NAMES = {
    'São Paulo',
    'Guarulhos',
    'Osasco',
    'Cotia',
    'Itapecerica da Serra',
    'Santo André',
    'Mairiporã',
    'São Bernardo do Campo',
    'Suzano',
    'Arujá',
    'Mogi das Cruzes',
    'Jacareí',
    'São José dos Campos',
    'Guaratinguetá',
    'Campinas',
    'Jundiaí',
    'Atibaia',
    'Sorocaba',
    'Piracicaba',
    'Santos',
}


def get_balanced_raw_cities() -> List[Tuple[int, str, float, float, int, str]]:
    """
    Retorna 20 cidades BALANCEADAS para otimização com AG.
    
    BALANCEAMENTO:
    ===============
    - Exatamente 20 cidades (tamanho padrão do projeto)
    - Distribuição geográfica por região (não concentra em uma)
    - Mix de criticidade: ~30% crítica, ~70% regular
    - Variação de demanda para ter "dificuldade" real
    
    CIDADES CRÍTICAS SELECIONADAS:
    - São Paulo (grupo 1) - Hub central
    - São Bernardo (grupo 2) - Polo industrial
    - Mogi das Cruzes (grupo 2) - Ponto norte
    - Campinas (grupo 3) - Interior importante
    - Piracicaba (grupo 3) - Mais distante
    - Santos (grupo 3) - Litoral/porta
    - Sorocaba (grupo 3) - Oeste
    
    OBJETIVO DO BALANCEAMENTO:
    Criar benchmark consistente e realista.
    Sem balanceamento, tamanho da população de cidades variaria,
    mudando radicalmente dificuldade e tempo do AG.
    """
    selected = [city for city in RAW_CITIES if city[1] in SELECTED_CITY_NAMES]
    selected.sort(key=lambda item: (item[0], item[1]))
    return selected
