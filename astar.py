import heapq

costs = {
    'M': 50,
    'A': 20,
    'N': 15,
    'F': 10,
    'R': 5,
    '.': 1,
    'P': 1,
    '0': 1
}

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

def dijkstra(grid, start):
    n, m = len(grid), len(grid[0])
    dist = [[float('inf')]*m for _ in range(n)]
    prev = [[None]*m for _ in range(n)]
    hq = []
    si, sj = start
    dist[si][sj] = 0
    heapq.heappush(hq, (0, si, sj))
    while hq:
        d, i, j = heapq.heappop(hq)
        if d>dist[i][j]: continue
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0<=ni<n and 0<=nj<m:
                if grid[ni][nj] == '#':
                    continue
                c = costs.get(grid[ni][nj], 0)
                nd = d+c
                if nd<dist[ni][nj]:
                    dist[ni][nj]=nd
                    prev[ni][nj]=(i,j)
                    heapq.heappush(hq,(nd,ni,nj))
    return dist, prev

def compute_all_pairs(specials, grid):
    nodes = list(specials.keys())
    idx = {n:i for i,n in enumerate(nodes)}
    N=len(nodes)
    distM=[[float('inf')]*N for _ in range(N)]
    pathM=[[None]*N for _ in range(N)]
    for u in nodes:
        ui=idx[u]
        dist, prev=dijkstra(grid,specials[u])
        for v in nodes:
            vi=idx[v]
            pi,pj=specials[v]
            distM[ui][vi]=dist[pi][pj]
            p=[]
            if dist[pi][pj]<float('inf'):
                cur=(pi,pj)
                while cur!=specials[u]:
                    p.append(cur)
                    cur=prev[cur[0]][cur[1]]
                p.append(specials[u])
                p.reverse()
            pathM[ui][vi]=p
    return nodes, idx, distM, pathM

def mst_cost(unvis, distM):
    if not unvis: return 0
    visited={unvis[0]}
    edges=[]
    cost=0
    for v in unvis[1:]: heapq.heappush(edges,(distM[unvis[0]][v],v))
    while len(visited)<len(unvis):
        d,u=heapq.heappop(edges)
        if u in visited: continue
        visited.add(u)
        cost+=d
        for v in unvis:
            if v not in visited:
                heapq.heappush(edges,(distM[u][v],v))
    return cost

def heuristic(node, mask, k, end, distM):
    full_mask=(1<<k)-1
    if mask==full_mask:
        return distM[node][end]
    unvis=[i+1 for i in range(k) if not(mask&(1<<i))]
    h=0
    dmin=min(distM[node][u] for u in unvis)
    h+=dmin
    h+=mst_cost(unvis,distM)
    h+=min(distM[u][end] for u in unvis)
    return h


def solve_tsp_astar(distM, start, end, k):
    full_mask = (1 << k) - 1
    pq = []
    h0 = heuristic(start, 0, k, end, distM)
    heapq.heappush(pq, (h0, 0, start, 0, [start]))

    frontier_log = [(start, 0)]
    visited_log = []
    seen = {}
    step = 0
    while pq:
        f, g, u, mask, path = heapq.heappop(pq)
        visited_log.append((u, step))
        if seen.get((u, mask), float('inf')) <= g:
            step += 1
            continue
        seen[(u, mask)] = g
        if u == end and mask == full_mask:
            return g, path, visited_log, frontier_log

        for i in range(1, k + 1):
            if not (mask & (1 << (i - 1))):
                ng = g + distM[u][i]
                nmask = mask | (1 << (i - 1))
                h = heuristic(i, nmask, k, end, distM)
                frontier_log.append((i, step))  # i está entrando na fila neste passo
                heapq.heappush(pq, (ng + h, ng, i, nmask, path + [i]))
        if mask == full_mask:
            ng = g + distM[u][end]
            frontier_log.append((end, step))
            heapq.heappush(pq, (ng, ng, end, mask, path + [end]))
        step += 1
    return float('inf'), [], visited_log, frontier_log

def format_steps(full_path, specials):
    coord2label = {coord: lbl for lbl, coord in specials.items()}
    lines = [f"start: {specials['0']}"]
    for coord in full_path:
        msg = f"go to {coord}"
        lbl = coord2label.get(coord)
        if lbl and lbl != '0':
            tag = "arrived at destination" if lbl == 'P' else "found event"
            msg += f" → {tag} {lbl}"
        lines.append(msg)
    return lines

def main():
    grid=read_map('mapa.txt')
    specials=find_special_points(grid)
    nodes, idx, distM, pathM=compute_all_pairs(specials,grid)
    start='0'
    end='P'
    evts=[n for n in nodes if n not in (start,end)]
    ordered=[start]+evts+[end]
    new_idx={n:i for i,n in enumerate(ordered)}
    k=len(evts)
    N2=len(ordered)
    D=[[0]*N2 for _ in range(N2)]
    Pm=[[None]*N2 for _ in range(N2)]
    for u in ordered:
        for v in ordered:
            ui,vi=new_idx[u],new_idx[v]
            D[ui][vi]=distM[idx[u]][idx[v]]
            Pm[ui][vi]=pathM[idx[u]][idx[v]]

    cost, path_idx, visited_log, frontier_log = solve_tsp_astar(D,0,N2-1,k)

    full_path=[]
    cur=0
    for nxt in path_idx[1:]:
        seg=Pm[cur][nxt]
        full_path+=seg[1:]
        cur=nxt

    seq=[ordered[i] for i in path_idx]
    start_coord = specials['0']
    coord2label = {coord: label for label, coord in specials.items()}

    steps = format_steps(full_path, specials)

    with open('optimal_path.txt', 'w', encoding='utf-8') as f:
        for line in steps:
            f.write(line + '\n')

    print('Cost:', cost)
    print('Sequence:', seq)

if __name__=='__main__':
    main()