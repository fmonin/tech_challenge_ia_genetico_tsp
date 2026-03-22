"""
MÓDULO: draw_functions.py
===========================
Renderização gráfica via Pygame + Matplotlib para visualizations em tempo real.

RESPONSABILIDADES:
1. Desenho do mapa de cidades (Pygame)
2. Desenho de rotas (linhas coloridas por veículo)
3. Desenho de painel de informações (legenda, resumo)
4. Geração de gráficos de evolução (Matplotlib)

USO:
- draw_cities(), draw_paths(), draw_text() → Pygame em tempo real
- draw_plot() → Matplotlib embutido em canvas Pygame
- save_fitness_chart() → Salva gráfico como PNG

PADRÃO DE CORES:
- RGB tuples (0-255 range)
- Cada veículo tem cor única (hardcoded em tsp.py VEHICLES)
- Rotas em teste: cinza, Melhor rota: cor do veículo
- Cidades críticas: anel vermelho ao redor
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, List, Tuple


def _rgb_to_mpl(rgb: Tuple[int, int, int]):
    """Converte RGB 0-255 para matplotlib 0.0-1.0 (float)."""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _render_figure_to_surface(fig: plt.Figure) -> pygame.Surface:
    """Renderiza matplotlib figure como Pygame surfacepara exibição. """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    raw_data = canvas.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    return surf


def draw_plot(
    screen: pygame.Surface,
    dashboard_data: Dict,
    position: Tuple[int, int] = (0, 0)
) -> None:
    """
    Desenha painel de gráficos em tempo real durante evolução.
    
    4 subgráficos:
    1. Fitness (global vs por veículo)
    2. Distância, demanda, tempo
    3. Custos e penalidades
    4. Distância por veículo
    
    Renderiza em meia resolução para não travar o FPS.
    """


def _render_figure_to_surface(fig: plt.Figure) -> pygame.Surface:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    raw_data = canvas.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    return surf


def draw_plot(
    screen: pygame.Surface,
    dashboard_data: Dict,
    position: Tuple[int, int] = (0, 0)
) -> None:
    """Desenha um painel com vários gráficos resumindo a evolução da solução."""
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.0), dpi=100)
    fig.patch.set_facecolor('#ffffff')

    ax1, ax2, ax3, ax4 = axes.flat

    fitness_series = dashboard_data['fitness_series']
    for item in fitness_series:
        ax1.plot(item['x'], item['values'], label=item['label'], linewidth=2.0, color=_rgb_to_mpl(item['color']))
    ax1.set_title('Fitness')
    ax1.set_xlabel('Geração')
    ax1.set_ylabel('Valor')
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=6, loc='best')

    totals = dashboard_data['totals']
    ax2.plot(totals['x'], totals['distance'], label='Distância total (km)', linewidth=1.8)
    ax2.plot(totals['x'], totals['demand'], label='Demanda total', linewidth=1.8)
    ax2.plot(totals['x'], totals['time'], label='Tempo total (min)', linewidth=1.8)
    ax2.set_title('Carga operacional')
    ax2.set_xlabel('Geração')
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=6, loc='best')

    ax3.plot(totals['x'], totals['cost'], label='Custo total', linewidth=1.8)
    ax3.plot(totals['x'], totals['priority'], label='Prioridade', linewidth=1.8)
    ax3.plot(totals['x'], totals['penalty'], label='Penalidade', linewidth=1.8)
    ax3.set_title('Custos e restrições')
    ax3.set_xlabel('Geração')
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=6, loc='best')

    for vehicle_data in dashboard_data['vehicle_distance_series']:
        ax4.plot(vehicle_data['x'], vehicle_data['values'], label=vehicle_data['label'], linewidth=1.8, color=_rgb_to_mpl(vehicle_data['color']))
    ax4.set_title('Distância por veículo')
    ax4.set_xlabel('Geração')
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=6, loc='best')

    for ax in axes.flat:
        ax.set_facecolor('#f7f8fa')
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

    plt.tight_layout(pad=1.0)
    surf = _render_figure_to_surface(fig)
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
    """Desenha as cidades no mapa."""
    for city in cities:
        position = city['screen_pos']
        pygame.draw.circle(screen, rgb_color, position, node_radius)
        pygame.draw.circle(screen, (255, 255, 255), position, max(1, node_radius - 2))

        if city['priority'] == 'critica':
            pygame.draw.circle(screen, (220, 60, 60), position, node_radius + 2, 1)

        if show_labels:
            # Criar rótulo com nome e indicador de criticidade
            label = city['name']
            if city['priority'] == 'critica':
                label += ' [C]'
            
            # Definir cor baseada na criticidade
            label_color = (255, 120, 120) if city['priority'] == 'critica' else (230, 230, 230)
            draw_text(screen, label, label_color, (position[0] + 6, position[1] - 8), font_size=10)


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
    """Mostra um resumo das rotas."""
    x, y = position
    panel_rect = pygame.Rect(x, y, 530, 170)
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
        extra = (
            f"tempo: {result['work_minutes']:.1f} min | custo: {result['total_cost']:.1f} | "
            f"prioridade: {result['priority_penalty']:.1f} | penalidade: {result['penalty']:.1f}"
        )
        draw_text(screen, extra, (215, 215, 215), (x + 28, line_y + 18), font_size=13)


def save_fitness_chart(dashboard_data: Dict, output_path: str = 'fitness_evolution.png') -> None:
    """Salva um dashboard com fitness e métricas operacionais."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=120)
    fig.patch.set_facecolor('#ffffff')
    ax1, ax2, ax3, ax4 = axes.flat

    for item in dashboard_data['fitness_series']:
        ax1.plot(item['x'], item['values'], label=item['label'], linewidth=2.2, color=_rgb_to_mpl(item['color']))
    ax1.set_title('Evolução do fitness global e por veículo')
    ax1.set_xlabel('Geração')
    ax1.set_ylabel('Fitness')
    ax1.grid(alpha=0.3)
    ax1.legend()

    totals = dashboard_data['totals']
    ax2.plot(totals['x'], totals['distance'], label='Distância total (km)', linewidth=2.0)
    ax2.plot(totals['x'], totals['demand'], label='Demanda total', linewidth=2.0)
    ax2.plot(totals['x'], totals['time'], label='Tempo total (min)', linewidth=2.0)
    ax2.set_title('Distância, demanda e tempo')
    ax2.set_xlabel('Geração')
    ax2.grid(alpha=0.3)
    ax2.legend()

    ax3.plot(totals['x'], totals['cost'], label='Custo total', linewidth=2.0)
    ax3.plot(totals['x'], totals['priority'], label='Prioridade', linewidth=2.0)
    ax3.plot(totals['x'], totals['penalty'], label='Penalidade', linewidth=2.0)
    ax3.set_title('Custo, prioridade e penalidade')
    ax3.set_xlabel('Geração')
    ax3.grid(alpha=0.3)
    ax3.legend()

    for vehicle_data in dashboard_data['vehicle_distance_series']:
        ax4.plot(vehicle_data['x'], vehicle_data['values'], label=vehicle_data['label'], linewidth=2.0, color=_rgb_to_mpl(vehicle_data['color']))
    ax4.set_title('Distância por veículo')
    ax4.set_xlabel('Geração')
    ax4.set_ylabel('km')
    ax4.grid(alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
