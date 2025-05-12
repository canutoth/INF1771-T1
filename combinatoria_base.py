from ortools.linear_solver import pywraplp
import itertools

# 1) Parâmetros
n_personagens = 6
max_vidas     = 5
m_atividades  = 19
poder         = [1.8, 1.6, 1.4, 1.3, 1.2, 1.0]
dif           = [55, 60, 65, 70, 75, 80, 85, 90, 95,
                 120,125,130,135,150,155,160,170,180,100]
uso_total     = n_personagens * max_vidas - 1  

# 2) Cria o solver CBC
solver = pywraplp.Solver.CreateSolver('CBC')
if not solver:
    raise Exception("Não foi possível inicializar o CBC")

# 3) Variáveis de decisão
# x[i,j] = 1 se personagem i for usado na atividade j
x = {}
for i in range(n_personagens):
    for j in range(m_atividades):
        x[i,j] = solver.IntVar(0, 1, f'x[{i},{j}]')

# s[j] = soma dos poderes na atividade j  (real contínua)
max_s = sum(poder)
s = {j: solver.NumVar(0.0, max_s, f's[{j}]')
     for j in range(m_atividades)}

# tau[j] = tempo da atividade j  (real contínua)
tau = {j: solver.NumVar(0.0, solver.infinity(), f'tau[{j}]')
       for j in range(m_atividades)}

# 4) Restrições básicas

# a) Máximo de 6 personagens por atividade
for j in range(m_atividades):
    solver.Add(sum(x[i,j] for i in range(n_personagens)) <= 6)

# b) Cada personagem i é usado no máximo max_vidas vezes
for i in range(n_personagens):
    solver.Add(sum(x[i,j] for j in range(m_atividades)) <= max_vidas)

# c) Total de usos ≤ uso_total
solver.Add(sum(x[i,j]
               for i in range(n_personagens)
               for j in range(m_atividades))
           <= uso_total)

# d) Definição de s[j]
for j in range(m_atividades):
    solver.Add(
        s[j] == sum(poder[i] * x[i,j]
                    for i in range(n_personagens))
    )

# 5) Linearização exata de tau[j] = dif[j] / s[j]
#    usando as restrições de tangentes em TODOS os possíveis valores de s[j].

# 5.1) Primeiro, liste todos os valores possíveis de soma de poderes
valores_s = set()
for r in range(1, n_personagens + 1):
    for comb in itertools.combinations(range(n_personagens), r):
        valores_s.add(sum(poder[i] for i in comb))
valores_s = sorted(valores_s)  # ≈ 2^6−1 = 63 valores

# 5.2) Para cada atividade j, e para cada ponto de tangência s0:
#      tau[j] ≥ f(s0) + f'(s0) * ( s[j] – s0 ),
#      onde f(s) = dif[j]/s
for j in range(m_atividades):
    for s0 in valores_s:
        f0      = dif[j] / s0
        fprime0 = - dif[j] / (s0 * s0)
        # constrói a inequação
        solver.Add(
            tau[j] >= f0 + fprime0 * (s[j] - s0)
        )

# 6) Objetivo: minimizar o tempo total
solver.Minimize(solver.Sum(tau[j] for j in range(m_atividades)))

# 7) Parâmetros do solve e execução
solver.set_time_limit(30 * 1000000000)  # 30 s em ms
status = solver.Solve()

# 8) Leitura de resultados
if status == pywraplp.Solver.OPTIMAL:
    print(f"❯ Tempo total ótimo = {solver.Objective().Value():.4f}\n")
    for j in range(m_atividades):
        usados = [i for i in range(n_personagens)
                  if x[i,j].solution_value() > 0.5]
        print(f"Atividade {j:2d}: personagens {', '.join(map(str, usados)):>10}  "
              f"soma_poder={s[j].solution_value():.2f}  "
              f"tempo={tau[j].solution_value():.2f}")
else:
    print("❯ O solver não encontrou solução ótima dentro do limite de tempo.")