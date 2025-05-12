import pygame
import time

CELL_SIZE = 2
VISIBLE_COLS = 400
VISIBLE_ROWS = 200
WINDOW_WIDTH = VISIBLE_COLS * CELL_SIZE
WINDOW_HEIGHT = VISIBLE_ROWS * CELL_SIZE
MAP_COLS = 153
MAP_ROWS = 400

# Cores
colors = {
    'M': (139, 69, 19),      # Montanha - marrom
    'A': (0, 0, 255),        # Água - azul escuro
    'N': (176, 224, 230),    # Neve - azul muito claro
    'F': (0, 255, 0),        # Floresta - verde claro
    'R': (255, 255, 255),    # Rua - branco
    '.': (190, 190, 190),    # Livre - cinza claro
    'P': (190, 190, 190),
    '0': (190, 190, 190)
}

AGENT_COLOR = (255, 0, 0)         # Vermelho - agente
PATH_COLOR = (0, 191, 255)        # Azul claro - caminho
EVENT_COLOR = (50, 205, 50)       # Verde limão - eventos
START_COLOR = (0, 255, 255)       # Ciano - início
END_COLOR = (255, 0, 255)         # Magenta - fim
EXPLORED_COLOR = (105, 105, 105)  # Cinza escuro

def read_mapa(filename):
    mapa = []
    with open(filename, 'r') as f:
        for linha in f:
            mapa.append(list(linha.strip()))
    return mapa

def read_search_vis(filename):
    frontier = []
    visited  = []
    with open(filename) as f:
        for line in f:
            tag, rest = line.split(' ', 1)
            step, coords = rest.split(':')
            i, j = map(int, coords.split(','))
            if tag == 'VIS':
                visited.append(((i, j), int(step)))
            else:
                frontier.append(((i, j), int(step)))

    visited.sort(key=lambda x: x[1])
    frontier.sort(key=lambda x: x[1])
    return visited, frontier

def read_path(filename):
    percurso = []
    with open(filename, 'r') as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            if linha.startswith('start'):
                coords = linha.split(':')[1].strip()
            elif linha.startswith('go to'):
                linha = linha.split(')')[0] + ')'
                coords = linha[5:].strip()
            else:
                continue
            coords = coords.replace('(', '').replace(')', '')
            i, j = map(int, coords.split(','))
            percurso.append((i, j))
    return percurso

def find_special_points(mapa):
    especiais = {}
    validos = set('0123456789BCDEGHIJKOP')
    for i, linha in enumerate(mapa):
        for j, c in enumerate(linha):
            if c in validos:
                especiais[(i, j)] = c
    return especiais

def desenhar_mapa(screen, mapa, caminho_percorrido, eventos, explorados, agente_pos, offset_x, offset_y, start_pos, end_pos):
    for i in range(VISIBLE_ROWS):
        for j in range(VISIBLE_COLS):
            map_i = offset_y + i
            map_j = offset_x + j
            if 0 <= map_i < len(mapa) and 0 <= map_j < len(mapa[0]):
                terreno = mapa[map_i][map_j]
                cor = colors.get(terreno, (0, 0, 0))
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, cor, rect)

                pos = (map_i, map_j)
                if pos == start_pos:
                    pygame.draw.rect(screen, START_COLOR, rect)
                elif pos == end_pos:
                    pygame.draw.rect(screen, END_COLOR, rect)
                elif pos in caminho_percorrido:
                    pygame.draw.rect(screen, PATH_COLOR, rect)
                elif pos in eventos:
                    pygame.draw.rect(screen, EVENT_COLOR, rect)
                elif pos in explorados:
                    pygame.draw.rect(screen, EXPLORED_COLOR, rect)

    agente_i, agente_j = agente_pos
    agent_screen_x = (agente_j - offset_x) * CELL_SIZE
    agent_screen_y = (agente_i - offset_y) * CELL_SIZE
    if 0 <= agent_screen_x < WINDOW_WIDTH and 0 <= agent_screen_y < WINDOW_HEIGHT:
        pygame.draw.rect(screen, AGENT_COLOR, (agent_screen_x, agent_screen_y, CELL_SIZE, CELL_SIZE))

def calcular_offset(agente_pos):
    i, j = agente_pos
    offset_y = i - VISIBLE_ROWS // 2
    offset_x = j - VISIBLE_COLS // 2
    offset_y = max(0, min(offset_y, MAP_ROWS - VISIBLE_ROWS))
    offset_x = max(0, min(offset_x, MAP_COLS - VISIBLE_COLS))
    return offset_x, offset_y

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Visualização da Busca Heurística")
    clock = pygame.time.Clock()

    mapa = read_mapa('mapa.txt')
    percurso = read_path('path.txt')
    eventos = find_special_points(mapa)

    start_pos = percurso[0]
    end_pos = percurso[-1]
    caminho_percorrido = set()
    explorados = set()

    sim_nos_explorados = min(300, len(percurso))
    for idx in range(sim_nos_explorados):
        pos = percurso[idx]
        explorados.add(pos)
        offset_x, offset_y = calcular_offset(pos)
        screen.fill((0, 0, 0))
        desenhar_mapa(screen, mapa, caminho_percorrido, eventos, explorados, pos, offset_x, offset_y, start_pos, end_pos)
        pygame.display.flip()
        time.sleep(0.005)

    idx = 0
    total_steps = len(percurso)
    while True:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        if idx < total_steps:
            pos = percurso[idx]
            caminho_percorrido.add(pos)
            offset_x, offset_y = calcular_offset(pos)

            screen.fill((0, 0, 0))
            desenhar_mapa(screen, mapa, caminho_percorrido, eventos, explorados, pos, offset_x, offset_y, start_pos, end_pos)

            idx += 1
            pygame.display.flip()

            if pos in eventos:
                print(f"Evento encontrado em {pos}")
                time.sleep(1)
            else:
                time.sleep(0.01)
        else:
            pygame.display.flip()

if __name__ == "__main__":
    main()