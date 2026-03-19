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
    """
    Desenha um gráfico na tela do Pygame usando Matplotlib.

    Parâmetros:
    - screen (pygame.Surface): A superfície do Pygame para desenhar o gráfico.
    - x (list): Valores do eixo x.
    - y (list): Valores do eixo y.
    - x_label (str): Rótulo do eixo x (padrão: 'Generation').
    - y_label (str): Rótulo do eixo y (padrão: 'Fitness').
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(x, y)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    raw_data = canvas.tostring_argb()
    size = canvas.get_width_height()
    surf = pygame.image.fromstring(raw_data, size, "ARGB")
    screen.blit(surf, (0, 0))
    
def draw_cities(screen: pygame.Surface, cities_locations: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], node_radius: int) -> None:
    """
    Desenha círculos representando as cidades na tela do Pygame.

    Parâmetros:
    - screen (pygame.Surface): A superfície do Pygame em que desenhar as cidades.
    - cities_locations (List[Tuple[int, int]]): Lista de coordenadas (x, y) das cidades.
    - rgb_color (Tuple[int, int, int]): Tupla RGB da cor dos círculos.
    - node_radius (int): Raio dos círculos.

    Retorna:
    None
    """
    for city_location in cities_locations:
        pygame.draw.circle(screen, rgb_color, city_location, node_radius)



def draw_paths(screen: pygame.Surface, path: List[Tuple[int, int]], rgb_color: Tuple[int, int, int], width: int = 1):
    """
    Desenha um caminho na tela do Pygame.

    Parâmetros:
    - screen (pygame.Surface): A superfície do Pygame para desenhar o caminho.
    - path (List[Tuple[int, int]]): Lista de coordenadas do caminho.
    - rgb_color (Tuple[int, int, int]): Valores RGB da cor do caminho.
    - width (int): Largura das linhas (padrão 1).
    """
    pygame.draw.lines(screen, rgb_color, True, path, width=width)


def draw_text(screen: pygame.Surface, text: str, color: pygame.Color) -> None:
    """
    Desenha texto na tela do Pygame.

    Parâmetros:
    - screen (pygame.Surface): A superfície do Pygame para desenhar o texto.
    - text (str): O texto a ser exibido.
    - color (pygame.Color): A cor do texto.
    """
    pygame.font.init()  # Deve ser chamado no início

    font_size = 15
    my_font = pygame.font.SysFont('Arial', font_size)
    text_surface = my_font.render(text, False, color)
    
    cities_locations = []  # Assuming you have this list defined somewhere
    text_position = (np.average(np.array(cities_locations)[:, 0]), HEIGHT - 1.5 * font_size)
    
    screen.blit(text_surface, text_position)

