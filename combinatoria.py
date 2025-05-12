import math
import random
import time
from copy import deepcopy

n_personagens = 6
max_vidas     = 5
m_atividades  = 19
poder         = [1.8, 1.6, 1.4, 1.3, 1.2, 1.0]
dif           = [55, 60, 65, 70, 75, 80, 85, 90, 95,
                 120,125,130,135,150,155,160,170,180,100]
uso_total     = n_personagens * max_vidas - 1  # 29

def inicializa_gulosa():
    estado = [[] for _ in range(m_atividades)]
    usos_rest = [max_vidas]*n_personagens
    idxs = sorted(range(m_atividades), key=lambda j: dif[j], reverse=True)
    total = 0
    for j in idxs:
        while total < uso_total and len(estado[j]) < n_personagens:
            cand = [i for i in range(n_personagens) if usos_rest[i]>0 and i not in estado[j]]
            if not cand: break
            i = max(cand, key=lambda i: poder[i])
            estado[j].append(i); usos_rest[i]-=1; total+=1
    while total < uso_total:
        i = random.randrange(n_personagens); j = random.randrange(m_atividades)
        if usos_rest[i]>0 and i not in estado[j] and len(estado[j])<n_personagens:
            estado[j].append(i); usos_rest[i]-=1; total+=1
    return estado

def tempo_total(estado):
    t = 0.0
    for j, equipe in enumerate(estado):
        if not equipe:
            t += 1e6
        else:
            soma_p = sum(poder[i] for i in equipe)
            t += dif[j] / soma_p
    return t

def vizinho_random(estado):
    novo = deepcopy(estado)
    j_from = random.randrange(m_atividades)
    if not novo[j_from]: return novo
    i = random.choice(novo[j_from]); novo[j_from].remove(i)
    candidatos = [j for j in range(m_atividades)
                  if i not in novo[j] and len(novo[j])<n_personagens]
    if candidatos:
        novo[random.choice(candidatos)].append(i)
    return novo

def vizinho_heuristico(estado):
    novo = deepcopy(estado)
    ratios = [ (dif[j]/sum(poder[i] for i in eq)) if eq else float('inf')
               for j, eq in enumerate(novo) ]
    j_high = max(range(m_atividades), key=lambda j: ratios[j])
    j_low  = min(range(m_atividades), key=lambda j: ratios[j])
    if novo[j_low]:
        i_low = min(novo[j_low], key=lambda i: poder[i])
        if i_low not in novo[j_high] and len(novo[j_high])<n_personagens:
            novo[j_low].remove(i_low)
            novo[j_high].append(i_low)
    return novo

def vizinho_swap(estado):
    novo = deepcopy(estado)
    j1, j2 = random.sample(range(m_atividades), 2)
    if not novo[j1] or not novo[j2]:
        return novo
    i1 = random.choice(novo[j1])
    i2 = random.choice(novo[j2])
    if (i2 not in novo[j1] and i1 not in novo[j2]):
        novo[j1].remove(i1); novo[j2].remove(i2)
        novo[j1].append(i2); novo[j2].append(i1)
    return novo

def vizinho_bestof(estado, k=5):
    best_n = None
    best_eval = float('inf')
    for _ in range(k):
        
        r = random.random()
        if r < 1/3:
            cand = vizinho_random(estado)
        elif r < 2/3:
            cand = vizinho_heuristico(estado)
        else:
            cand = vizinho_swap(estado)
        val = tempo_total(cand)
        if val < best_eval:
            best_eval = val
            best_n = cand
    return best_n

def simulated_annealing_heuristico(
    time_limit=300.0, temp0=100.0, alpha=0.995,
    Tmin=0.1, p_h=0.3, p_s=0.2, p_b=0.3
):
    start = time.time()
    current = inicializa_gulosa()
    current_eval = tempo_total(current)
    best, best_eval = deepcopy(current), current_eval
    T = temp0; step = 0

    while time.time() - start < time_limit:
        r = random.random()
        if r < p_b:
            viz = vizinho_bestof(current, k=5)
        elif r < p_b + p_h:
            viz = vizinho_heuristico(current)
        elif r < p_b + p_h + p_s:
            viz = vizinho_swap(current)
        else:
            viz = vizinho_random(current)

        viz_eval = tempo_total(viz)
        delta = viz_eval - current_eval

        if delta < 0 or random.random() < math.exp(-delta / T):
            current, current_eval = viz, viz_eval
            if viz_eval < best_eval:
                best, best_eval = deepcopy(viz), viz_eval

        T *= alpha; step += 1
        if T < Tmin:
            T = temp0
            current, current_eval = deepcopy(best), best_eval

        if step % 10000 == 0:
            elapsed = time.time() - start
            print(f"[{elapsed:.1f}s] it={step}, T={T:.2f}, best={best_eval:.3f}")

    return best, best_eval

if __name__ == "__main__":
    sol, tempo = simulated_annealing_heuristico(
        time_limit=500.0,
        temp0=50.0,
        alpha=0.998,
        Tmin=0.5,
        p_h=0.3,
        p_s=0.2,
        p_b=0.3
    )
    print("Melhor tempo total:", tempo)
    for j, eq in enumerate(sol):
        print(f"Ativ {j:2d} (dif={dif[j]}):", eq)