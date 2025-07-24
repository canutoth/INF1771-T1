# INF1771 – Trabalho 1: Busca Heurística (A*)

## Descrição
Este repositório reúne a implementação de um agente de busca heurística (A*) capaz de guiar personagens através de 19 eventos no mundo de Skyrim, encontrando a rota global de menor custo desde a cidade inicial (Whiterun) até o destino final (próximo ao lago Hinalta em Falkreath).

## Mapa e Terrenos
- **Dimensões**: matriz de 400 × 153 (`map.txt`)  
- **Terrenos** (custo ao entrar):
  - Montanha (`M`): 50
  - Água (`A`): 20
  - Neve (`N`): 15
  - Floresta (`F`): 10
  - Rochoso (`R`): 5
  - Livre (`.`): 1
- Bloqueados (`#`) não são navegáveis.

## Eventos e Personagens
- **19 eventos** com dificuldade específica (por exemplo, Helgen – 55, Riverwood – 60, …, Batalha contra Alduin – 180).  
- **Personagens** e seus factores de poder:
  - Dragonborn: 1.8  
  - Ralof ou Hadvar: 1.6  
  - Lydia: 1.4  
  - Farengar Secret‑Fire: 1.3  
  - Balgruuf, o Grande: 1.2  
  - Delphine: 1.0  

- Cada personagem possui 5 pontos de energia; perde 1 por evento e fica em hibernação ao chegar a 0. Deve restar pelo menos um personagem activo ao chegar ao destino.

## Funcionalidades
- Planeamento de caminho óptimo com A* (movimento ortogonal apenas).  
- Planeamento local de batalhas, escolhendo a equipa de personagens ideal.  
- Visualização simples implementada em pygame, exibindo:
- Posição actual, fronteira, estados visitados e caminho final.  
- Custo incremental à medida que o agente avança e custo total.  
- Parâmetros (mapa, terrenos, eventos e personagens) configuráveis via ficheiros de texto ou código.

Vídeo Relatório:
https://drive.google.com/file/d/1zmGCeGDubRZRZuj2UEt-pecTNs8Nf0fL/view?usp=sharing
