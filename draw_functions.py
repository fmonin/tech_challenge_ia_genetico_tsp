# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:03:11 2023

@author: SérgioPolimante
"""
import pylab
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
import pygame
from typing import List, Tuple

matplotlib.use("Agg")


def draw_plot(screen: pygame.Surface, x: list, y: list, x_label: str = 'Generation', y_label: str = 'Fitness') -> None:
    """Desenha um gráfico de linha no surface do Pygame usando Matplotlib.

    Parâmetros:
    - screen: superfície para desenhar.
    - x: valores do eixo x.
    - y: valores do eixo y.
    - x_label: rótulo do eixo x.
    - y_label: rótulo do eixo y.
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(x, y)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    raw_data = canvas.buffer_rgba()

    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    screen.blit(surf, (0, 0))
    plt.close(fig)
    
def draw_cities(screen: pygame.Surface, cities_locations: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], node_radius: int) -> None:
    """Desenha círculos para cada cidade em tela do Pygame.

    Parâmetros:
    - screen: superfície Pygame.
    - cities_locations: lista de posições (x, y).
    - rgb_color: cor do círculo.
    - node_radius: raio do nó.
    """
    for city_location in cities_locations:
        pygame.draw.circle(screen, rgb_color, city_location, node_radius)



def draw_paths(screen: pygame.Surface, path: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], width: int = 1):
    """Desenha um caminho poligonal conectando as cidades em ordem.

    Parâmetros:
    - screen: superfície Pygame.
    - path: lista de coordenadas do caminho.
    - rgb_color: cor da linha.
    - width: espessura da linha.
    """
    pygame.draw.lines(screen, rgb_color, True, path, width=width)


def draw_text(screen: pygame.Surface, text: str, color: pygame.Color) -> None:
    """Desenha texto na tela do Pygame.

    Parâmetros:
    - screen: superfície Pygame.
    - text: texto para renderizar.
    - color: cor do texto.
    """
    pygame.font.init()  # Inicializa o sistema de fontes do Pygame

    font_size = 15
    my_font = pygame.font.SysFont('Arial', font_size)
    text_surface = my_font.render(text, False, color)

    # OBS: esta função usa uma variável `cities_locations` vazia local.
    # Pode ser substituído por uma posição fixa ou recebida como argumento.
    cities_locations = []
    text_position = (0, 0)
    screen.blit(text_surface, text_position)

