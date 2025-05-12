import math
import random
import time
import heapq
from copy import deepcopy
import pygame
import sys
import threading

#VAR - Interface
CELL_SIZE     = 2
MARKER_FACTOR = 2
VISIBLE_COLS  = 400
VISIBLE_ROWS  = 153
WINDOW_WIDTH  = VISIBLE_COLS * CELL_SIZE
WINDOW_HEIGHT = VISIBLE_ROWS * CELL_SIZE

colors = {
    'M': (128, 128, 128),  # montanha - cinza médio
    'A': (0, 0, 255),      # água - azul
    'N': (173, 216, 230),  # neve - azul claro
    'F': (120, 250, 0),   # floresta - verde mais claro
    'R': (255, 255, 255),  # rochoso - branco
    '.': (211, 211, 211),  # livre - cinza claro
    'P': (211, 211, 211),  # ponto especial - cinza claro
    '0': (211, 211, 211)   # ponto extra - cinza claro
}

PATH_COLOR         = (255, 0, 255)
EVENT_COLOR        = (50, 205, 50)
FINAL_EVENT_COLOR  = (255, 140, 0)
START_COLOR        = (0, 255, 255)
END_COLOR          = (255, 0, 255)
AGENT_COLOR        = (255, 0, 0)

AGENT_RADIUS = CELL_SIZE * MARKER_FACTOR
EVENT_RADIUS = CELL_SIZE * MARKER_FACTOR
MARKER_SIZE  = CELL_SIZE * MARKER_FACTOR



#VAR - Combinatória
num_characters = 6
max_energy     = 5
num_events     = 19
power          = [1.8, 1.6, 1.4, 1.3, 1.2, 1.0]
difficulty     = [55, 60, 65, 70, 75, 80, 85, 90, 95,
                  120,125,130,135,150,155,160,170,180,100]
total_uses     = num_characters * max_energy - 1  

character_names = [
    "Dragonborn",
    "Ralof/Hadvar",
    "Lydia",
    "Farengar",
    "Balgruuf",
    "Delphine"
]

#Combinatória
def format_steps(full_path, specials):
    coord2label = {coord: lbl for lbl, coord in specials.items()}
    lines = [f"start: {specials['0']}"]
    for coord in full_path:
        msg = f"go to {coord}"
        lbl = coord2label.get(coord)
        if lbl and lbl != '0':
            tag = "arrived at destination" if lbl == 'P' else "found event"
            msg += f" -> {tag} {lbl}"
        lines.append(msg)
    return lines

import math
import random
import time
from copy import deepcopy

num_characters = 6
max_energy = 5
num_events = 19
power = [1.8, 1.6, 1.4, 1.3, 1.2, 1.0]
difficulty = [
    55, 60, 65, 70, 75, 80, 85, 90, 95,
    120, 125, 130, 135, 150, 155, 160, 170, 180, 100
]
total_uses = num_characters * max_energy - 1

def greedy_initialize_guaranteed():
    state = [[] for _ in range(num_events)]
    remaining = [max_energy] * num_characters
    used = 0

    chars_sorted = sorted(
        range(num_characters),
        key=lambda i: power[i],
        reverse=True
    )
    events_sorted = sorted(
        range(num_events),
        key=lambda j: difficulty[j],
        reverse=True
    )

    idxs = events_sorted[:]
    while used < total_uses:
        for pos, j in enumerate(idxs):
            if used >= total_uses:
                break

            candidates = [
                i for i in chars_sorted
                if remaining[i] > 0 and i not in state[j]
            ]
            if not candidates:
                continue

            factor = pos / (len(idxs) - 1) if len(idxs) > 1 else 0
            idx_cand = int(factor * (len(candidates) - 1))
            i = candidates[idx_cand]

            state[j].append(i)
            remaining[i] -= 1
            used += 1

    return state

def total_time(state):
    t = 0.0
    for j, team in enumerate(state):
        if not team:
            t += 1e6
        else:
            t += difficulty[j] / sum(power[i] for i in team)
    return t

def random_neighbor(state):
    new = deepcopy(state)
    j_from = random.randrange(num_events)
    if not new[j_from]:
        return new
    i = random.choice(new[j_from])
    new[j_from].remove(i)
    dests = [
        j for j in range(num_events)
        if i not in new[j] and len(new[j]) < num_characters
    ]
    if dests:
        new[random.choice(dests)].append(i)
    return new

def heuristic_neighbor(state):
    new = deepcopy(state)
    ratios = [
        (difficulty[j] / sum(power[i] for i in team))
        if team else float('inf')
        for j, team in enumerate(new)
    ]
    j_high = max(range(num_events), key=lambda j: ratios[j])
    j_low = min(range(num_events), key=lambda j: ratios[j])
    if new[j_low]:
        i_low = min(new[j_low], key=lambda i: power[i])
        if i_low not in new[j_high] and len(new[j_high]) < num_characters:
            new[j_low].remove(i_low)
            new[j_high].append(i_low)
    return new

def swap_neighbor(state):
    new = deepcopy(state)
    j1, j2 = random.sample(range(num_events), 2)
    if not new[j1] or not new[j2]:
        return new
    i1 = random.choice(new[j1])
    i2 = random.choice(new[j2])
    if i2 not in new[j1] and i1 not in new[j2]:
        new[j1].remove(i1)
        new[j2].remove(i2)
        new[j1].append(i2)
        new[j2].append(i1)
    return new

def best_of_k_neighbors(state, k=5):
    best_state = None
    best_eval = float('inf')
    for _ in range(k):
        r = random.random()
        if r < 1/3:
            cand = random_neighbor(state)
        elif r < 2/3:
            cand = heuristic_neighbor(state)
        else:
            cand = swap_neighbor(state)
        val = total_time(cand)
        if val < best_eval:
            best_eval = val
            best_state = cand
    return best_state

def simulated_annealing_multirun(
    temp0=100.0, alpha=0.999,
    p_h=0.3, p_s=0.2, p_b=0.3,
    runs=5, replace_interval=10000, threshold=25000
):
    succ = {'random': 1, 'heuristic': 1, 'swap': 1, 'best_of_k': 1}
    k0 = 5

    def init_run():
        state = greedy_initialize_guaranteed()
        ev = total_time(state)
        return {
            'current': state,
            'current_eval': ev,
            'best': deepcopy(state),
            'best_eval': ev,
            'T': temp0,
            'step': 0
        }

    threads = [init_run() for _ in range(runs)]
    total_iters = 0
    check_interval = 100
    equal_count = 0
    start_time = time.time()

    print(f"Iniciando SA com {runs} threads, replace a cada {replace_interval} iterações...")

    while True:
        for idx, t in enumerate(threads):
            r = random.random()

            if r < p_b:
                k_current = max(1, int(k0 * (t['T'] / temp0)))
                cand = best_of_k_neighbors(t['current'], k=k_current)
                op = 'best_of_k'
            elif r < p_b + p_h:
                cand = heuristic_neighbor(t['current'])
                op = 'heuristic'
            elif r < p_b + p_h + p_s:
                cand = swap_neighbor(t['current'])
                op = 'swap'
            else:
                cand = random_neighbor(t['current'])
                op = 'random'

            cand_eval = total_time(cand)

            if cand_eval < t['current_eval']:
                succ[op] += 1

            delta = cand_eval - t['current_eval']
            if delta < 0 or random.random() < math.exp(-delta / t['T']):
                t['current'], t['current_eval'] = cand, cand_eval
                if cand_eval < t['best_eval']:
                    t['best'], t['best_eval'] = deepcopy(cand), cand_eval

            t['T'] *= alpha
            t['step'] += 1

            if t['step'] % replace_interval == 0:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.2f}s] thread {idx+1} – step={t['step']}, best={t['best_eval']:.5f}")

        total_iters += 1

        if total_iters % replace_interval == 0:
            worst = max(threads, key=lambda x: x['best_eval'])
            print(f"Substituindo thread pior (best={worst['best_eval']:.5f}) por nova")
            threads.remove(worst)
            threads.append(init_run())

        if total_iters % check_interval == 0:
            best_vals = [round(t['best_eval'], 4) for t in threads]
            freq = max(best_vals.count(v) for v in best_vals)
            equal_count = equal_count + check_interval if freq >= runs - 1 else 0

        total_s = sum(succ.values())
        p_b = succ['best_of_k'] / total_s
        p_h = succ['heuristic'] / total_s
        p_s = succ['swap'] / total_s

        if equal_count >= threshold:
            print(f"Threshold de {threshold} atingido. Encerrando.")
            break

    best_run = min(threads, key=lambda x: x['best_eval'])
    print(f"Solução final: {best_run['best_eval']:.6f}")
    return best_run['best'], best_run['best_eval']

#A*
costs = {'M':50,'A':20,'N':15,'F':10,'R':5, '.':1}

def dijkstra(grid, start):
    n, m = len(grid), len(grid[0])
    dist = [[float('inf')]*m for _ in range(n)]
    prev = [[None]*m for _ in range(n)]
    hq = [(0, start[0], start[1])]
    dist[start[0]][start[1]] = 0

    while hq:
        d, i, j = heapq.heappop(hq)
        if d > dist[i][j]:
            continue
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < n and 0 <= nj < m and grid[ni][nj] != '#':
                c = costs.get(grid[ni][nj], 1)
                nd = d + c
                if nd < dist[ni][nj]:
                    dist[ni][nj] = nd
                    prev[ni][nj] = (i, j)
                    heapq.heappush(hq, (nd, ni, nj))
    return dist, prev

def compute_all_pairs(specials, grid):
    nodes = list(specials.keys())
    idx   = {n:i for i,n in enumerate(nodes)}
    N     = len(nodes)
    distM = [[float('inf')]*N for _ in range(N)]
    pathM = [[None]*N for _ in range(N)]

    for u in nodes:
        ui = idx[u]
        dist, prev = dijkstra(grid, specials[u])
        for v in nodes:
            vi = idx[v]
            pi, pj = specials[v]
            distM[ui][vi] = dist[pi][pj]
            p = []
            if dist[pi][pj] < float('inf'):
                cur = (pi,pj)
                while cur != specials[u]:
                    p.append(cur)
                    cur = prev[cur[0]][cur[1]]
                p.append(specials[u])
                p.reverse()
            pathM[ui][vi] = p

    return nodes, idx, distM, pathM

def mst_cost(unvisited, distM):
    if not unvisited:
        return 0
    visited = {unvisited[0]}
    edges = []
    cost = 0
    for v in unvisited[1:]:
        heapq.heappush(edges, (distM[unvisited[0]][v], v))
    while len(visited) < len(unvisited):
        d, u = heapq.heappop(edges)
        if u in visited:
            continue
        visited.add(u)
        cost += d
        for v in unvisited:
            if v not in visited:
                heapq.heappush(edges, (distM[u][v], v))
    return cost

def heuristic(u, mask, k, end, distM):
    full = (1<<k) - 1
    if mask == full:
        return distM[u][end]
    unvis = [i+1 for i in range(k) if not(mask & (1<<i))]
    h = min(distM[u][v] for v in unvis)
    h += mst_cost(unvis, distM)
    h += min(distM[v][end] for v in unvis)
    return h

def solve_tsp_astar(distM, start, end, k):
    full = (1<<k)-1
    pq = []
    h0 = heuristic(start, 0, k, end, distM)
    heapq.heappush(pq, (h0, 0, start, 0, [start]))
    seen = {}

    while pq:
        f, g, u, mask, path = heapq.heappop(pq)
        if seen.get((u,mask), float('inf')) <= g:
            continue
        seen[(u,mask)] = g

        if u == end and mask == full:
            return g, path

        for i in range(1, k+1):
            if not(mask & (1<<(i-1))):
                ng   = g + distM[u][i]
                nmask= mask | (1<<(i-1))
                h    = heuristic(i, nmask, k, end, distM)
                heapq.heappush(pq, (ng+h, ng, i, nmask, path+[i]))

        if mask == full:
            ng = g + distM[u][end]
            heapq.heappush(pq, (ng, ng, end, mask, path+[end]))

    return float('inf'), []

def spinner(stop_event):
    count = 0
    
    print("Iniciando a busca pelo caminho ótimo...")
    while not stop_event.is_set():
        count += 1
        print(f"\rProcurando por {count} segundos...", end="", flush=True)
        time.sleep(1)

    print()

#Interface
def read_path(fname):
    """
    Lê path.txt linha a linha, processa 'start:' e 'go to', descarta texto após '->'.
    Retorna lista de tuplas (i, j).
    """
    path = []
    with open(fname) as f:
        for l in f:
            l = l.strip()
            if l.startswith('start:'):
                part = l.split(':', 1)[1]
            elif l.startswith('go to'):
                part = l[len('go to'):]
            else:
                continue
            part = part.split('->')[0].strip()
            if part.startswith('(') and part.endswith(')'):
                part = part[1:-1]
            i_str, j_str = part.split(',', 1)
            try:
                path.append((int(i_str), int(j_str)))
            except ValueError:
                print(f"Skipping invalid line in path file: {l}")
    return path

def read_final_eventos(fname):
    ev = {}
    with open(fname) as f:
        for l in f:
            if l.startswith('Evento'):
                partes = l.split('em', 1)
                label = partes[0].split()[1]
                coord = partes[1].split(':', 1)[0].strip()
                if coord.startswith('(') and coord.endswith(')'):
                    coord = coord[1:-1]
                i, j = map(int, coord.split(','))
                ev[(i, j)] = label
    return ev

def draw_map(screen, mapa, path_so_far, specials, final_events,
             agent_pos, start_pos, end_pos, offset_x, offset_y):
    if specials and isinstance(next(iter(specials)), str):
        specials = { coord: lbl for lbl, coord in specials.items() }

    screen.fill((0, 0, 0))
    for di in range(VISIBLE_ROWS):
        for dj in range(VISIBLE_COLS):
            mi = offset_y + di
            mj = offset_x + dj
            if 0 <= mi < len(mapa) and 0 <= mj < len(mapa[0]):
                color = colors.get(mapa[mi][mj], (0, 0, 0))
                pygame.draw.rect(screen, color,
                                 (dj * CELL_SIZE, di * CELL_SIZE,
                                  CELL_SIZE, CELL_SIZE))

    for (i, j) in path_so_far:
        xi = (j - offset_x) * CELL_SIZE
        yi = (i - offset_y) * CELL_SIZE
        pygame.draw.rect(screen, PATH_COLOR,
                         (xi, yi, MARKER_SIZE, MARKER_SIZE))

    for (i, j), lbl in specials.items():
        xi = (j - offset_x) * CELL_SIZE
        yi = (i - offset_y) * CELL_SIZE
        color = FINAL_EVENT_COLOR if (i, j) in final_events else EVENT_COLOR
        pygame.draw.circle(screen, color,
                           (xi + CELL_SIZE // 2, yi + CELL_SIZE // 2),
                           EVENT_RADIUS)

    for pos, col in [(start_pos, START_COLOR), (end_pos, END_COLOR)]:
        i, j = pos
        xi = (j - offset_x) * CELL_SIZE
        yi = (i - offset_y) * CELL_SIZE
        pygame.draw.rect(screen, col,
                         (xi, yi, MARKER_SIZE, MARKER_SIZE))

    ai, aj = agent_pos
    xi = (aj - offset_x) * CELL_SIZE
    yi = (ai - offset_y) * CELL_SIZE
    pygame.draw.circle(screen, AGENT_COLOR,
                       (xi + CELL_SIZE // 2, yi + CELL_SIZE // 2),
                       AGENT_RADIUS)

#Geral
def read_map(filename):
    with open(filename, 'r') as f:
        return [list(line.rstrip('\n')) for line in f]

def find_special_points(grid):
    specials = {}
    valid = set('0123456789BCDEGHIJKOP')
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell in valid:
                specials[cell] = (i, j)
    return specials


#Main 
def main():
    start_time = time.time()
    assignment, combinatory_value = simulated_annealing_multirun(
        temp0=100.0,
        alpha=0.995,
        p_h=0.3,
        p_s=0.2,
        p_b=0.3,
        runs=4,
        replace_interval=10000,
        threshold=30000
    )
    elapsed = time.time() - start_time

    print("\n=== RESULTADOS DA COMBINATÓRIA ===")
    print(f"Custo ótimo encontrado: {combinatory_value:.6f}")
    print(f"Tempo total de execução: {elapsed:.2f}s")
    print("Atribuição por evento (lista de índices de personagens):")
    for event_idx, team in enumerate(assignment, start=1):
        print(f"  Evento {event_idx:02d}: {team}")

    grid     = read_map('mapa.txt')
    specials = find_special_points(grid)
    nodes, idx, distM, pathM = compute_all_pairs(specials, grid)

    start = '0'
    end = 'P'
    events = [n for n in nodes if n not in (start, end)]
    ordered = [start] + events + [end]
    new_idx = {n:i for i,n in enumerate(ordered)}
    k = len(events)
    N2 = len(ordered)
    D  = [[0]*N2 for _ in range(N2)]
    Pm=[[None]*N2 for _ in range(N2)]
    for u in ordered:
        for v in ordered:
            ui,vi=new_idx[u],new_idx[v]
            D[ui][vi]=distM[idx[u]][idx[v]]
            Pm[ui][vi]=pathM[idx[u]][idx[v]]

    stop_event = threading.Event()

    spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
    spinner_thread.start()
    path_cost, path_idx = solve_tsp_astar(D, 0, N2-1, k)
    stop_event.set()
    spinner_thread.join()


    full_path = []
    cur = 0
    for nxt in path_idx[1:]:
        seg = Pm[cur][nxt]
        full_path += seg[1:]
        cur = nxt

    sequence = [ordered[i] for i in path_idx]

    steps = format_steps(full_path, specials)

    with open('optimal_path.txt', 'w', encoding='utf-8') as f:
        for line in steps:
            f.write(line + '\n')

    for line in steps:
        print(line)


    print("Caminho encontrado e salvo em optimal_path.txt!")

    adjusted_path_cost = path_cost - num_events

    print("\n=== Atribuição de personagens por evento ===")
    label_to_sa_index = {
        **{str(i+1): i   for i in range(9)},
        'B': 9, 'C':10, 'D':11, 'E':12, 'G':13,
        'H':14, 'I':15, 'J':16, 'K':17, 'O':18
    }
    for lbl in sequence:
        if lbl in label_to_sa_index:
            ev_idx = label_to_sa_index[lbl]
            chars = [character_names[i] for i in assignment[ev_idx]]
            coords = specials[lbl]
            print(f"Evento {lbl} em {coords}: {', '.join(chars)}")

    print("\n=== Estatísticas finais ===")
    print(f"Custo da combinatória achado: {combinatory_value:.5f}") 
    print(f"Custo do caminho achado: {adjusted_path_cost:.2f}")
    total_found = combinatory_value + adjusted_path_cost
    print(f"Custo total achado: {total_found:.5f}")

    with open('final_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("\n=== Atribuição de personagens por evento ===\n")
        for lbl in sequence:
            if lbl in label_to_sa_index:
                ev_idx = label_to_sa_index[lbl]
                chars = [character_names[i] for i in assignment[ev_idx]]
                coords = specials[lbl]
                f.write(f"Evento {lbl} em {coords}: {', '.join(chars)}\n")

        f.write("\n=== Estatísticas finais ===\n")
        f.write(f"Custo da combinatória achado: {combinatory_value:.5f}\n")
        f.write(f"Custo do caminho achado: {adjusted_path_cost:.2f}\n")
        total_found = combinatory_value + adjusted_path_cost
        f.write(f"Custo total achado: {total_found:.5f}\n")

    map         = read_map('mapa.txt')
    path         = read_path('optimal_path.txt')
    final_events = read_final_eventos('final_statistics.txt')
    specials     = find_special_points(map)
    start_pos    = path[0]
    end_pos      = path[-1]

    offset_x = max(0, min(start_pos[1] - VISIBLE_COLS // 2,
                          len(map[0]) - VISIBLE_COLS))
    offset_y = max(0, min(start_pos[0] - VISIBLE_ROWS // 2,
                          len(map) - VISIBLE_ROWS))

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Caminho Final – Interface")

    path_so_far = []
    for pos in path:
        path_so_far.append(pos)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        draw_map(screen, map, path_so_far, specials, final_events,
                 agent_pos=pos, start_pos=start_pos, end_pos=end_pos,
                 offset_x=offset_x, offset_y=offset_y)
        pygame.display.flip()
        time.sleep(0.02)

    while True:
        if pygame.event.wait().type == pygame.QUIT:
            pygame.quit(); sys.exit()

  

if __name__ == "__main__":
    main()