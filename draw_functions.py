import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, List, Tuple


def draw_plot(
    screen: pygame.Surface,
    x: list,
    series: list,
    x_label: str = 'Geração',
    y_label: str = 'Fitness',
    position: Tuple[int, int] = (0, 0)
) -> None:
    """Desenha um gráfico dentro da tela.

    Agora ele aceita várias linhas.
    Cada item de series precisa ter:
    - label
    - values
    - color
    """
    fig, ax = plt.subplots(figsize=(5.1, 3.2), dpi=100)

    for item in series:
        # O matplotlib usa cor de 0 a 1, então aqui eu converto do padrão RGB.
        rgb = item['color']
        color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

        ax.plot(
            x,
            item['values'],
            label=item['label'],
            linewidth=2.0,
            color=color
        )

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    raw_data = canvas.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, 'RGBA')
    screen.blit(surf, position)
    plt.close(fig)


def draw_text(
    screen: pygame.Surface,
    text: str,
    color: Tuple[int, int, int],
    position: Tuple[int, int],
    font_size: int = 16,
    bold: bool = False
) -> None:
    """Escreve um texto simples na tela."""
    pygame.font.init()
    font = pygame.font.SysFont('Arial', font_size, bold=bold)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


def draw_cities(
    screen: pygame.Surface,
    cities: List[Dict],
    rgb_color: Tuple[int, int, int],
    node_radius: int = 5,
    show_labels: bool = False
) -> None:
    """Desenha as cidades no mapa.

    Quando show_labels=True, mostra também a criticidade de cada ponto.
    """
    for city in cities:
        position = city['screen_pos']
        pygame.draw.circle(screen, rgb_color, position, node_radius)
        pygame.draw.circle(screen, (255, 255, 255), position, max(1, node_radius - 2))

        if city['priority'] == 'critica':
            pygame.draw.circle(screen, (220, 60, 60), position, node_radius + 2, 1)

        if show_labels:
            priority_text = 'Crítica' if city['priority'] == 'critica' else 'Regular'
            label = f"{city['name']} - {priority_text}"
            label_x = position[0] + 8
            label_y = position[1] - 10

            # sombra simples para melhorar a leitura no mapa
            draw_text(screen, label, (20, 20, 20), (label_x + 1, label_y + 1), font_size=11, bold=False)
            label_color = (255, 210, 210) if city['priority'] == 'critica' else (230, 230, 230)
            draw_text(screen, label, label_color, (label_x, label_y), font_size=11, bold=False)


def draw_paths(
    screen: pygame.Surface,
    path: List[Tuple[int, int]],
    rgb_color: Tuple[int, int, int],
    width: int = 1,
    close_path: bool = True
) -> None:
    """Desenha a rota na tela."""
    if len(path) < 2:
        return

    if close_path:
        pygame.draw.lines(screen, rgb_color, True, path, width)
    else:
        pygame.draw.lines(screen, rgb_color, False, path, width)


def draw_vehicle_legend(screen: pygame.Surface, vehicles: List[Dict], position: Tuple[int, int]) -> None:
    """Desenha a legenda com os veículos disponíveis."""
    x, y = position
    panel_rect = pygame.Rect(x, y, 360, 145)
    pygame.draw.rect(screen, (20, 28, 40), panel_rect, border_radius=8)
    pygame.draw.rect(screen, (80, 120, 180), panel_rect, 1, border_radius=8)

    draw_text(screen, 'Legenda dos veículos', (255, 255, 255), (x + 10, y + 8), font_size=18, bold=True)

    for index, vehicle in enumerate(vehicles):
        top = y + 36 + index * 32
        pygame.draw.circle(screen, vehicle['color'], (x + 14, top + 8), 6)
        text = (
            f"{vehicle['label']} | {vehicle['name']} | cap: {vehicle['capacity']} | "
            f"aut: {vehicle['max_distance']} km | custo: {vehicle['operational_cost']}"
        )
        draw_text(screen, text, (225, 225, 225), (x + 28, top), font_size=14)


def draw_route_summary(screen: pygame.Surface, route_results: List[Dict], position: Tuple[int, int]) -> None:
    """Mostra um resumo das 3 rotas."""
    x, y = position
    panel_rect = pygame.Rect(x, y, 430, 170)
    pygame.draw.rect(screen, (20, 28, 40), panel_rect, border_radius=8)
    pygame.draw.rect(screen, (80, 120, 180), panel_rect, 1, border_radius=8)

    draw_text(screen, 'Resumo das rotas', (255, 255, 255), (x + 10, y + 8), font_size=18, bold=True)

    for i, result in enumerate(route_results):
        line_y = y + 40 + i * 40
        vehicle = result['vehicle']
        info = (
            f"Rota {i + 1} | {vehicle['label']} - {vehicle['name']} | "
            f"cidades: {len(result['best_route'])} | demanda: {result['demand']} | "
            f"dist: {result['distance_km']:.1f} km | fit: {result['fitness']:.1f}"
        )
        draw_text(screen, info, vehicle['color'], (x + 10, line_y), font_size=14)
        extra = f"críticas: {result['critical_count']} | penalidade: {result['penalty']:.1f}"
        draw_text(screen, extra, (215, 215, 215), (x + 28, line_y + 18), font_size=13)


def save_fitness_chart(series: list, output_path: str = 'fitness_evolution.png') -> None:
    """Salva o gráfico final com várias linhas."""
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)

    for item in series:
        rgb = item['color']
        color = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

        ax.plot(
            item['x'],
            item['values'],
            label=item['label'],
            linewidth=2.2,
            color=color
        )

    ax.set_title('Evolução do fitness por veículo e fitness global')
    ax.set_xlabel('Geração')
    ax.set_ylabel('Fitness')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)