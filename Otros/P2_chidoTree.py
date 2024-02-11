from xmlrpc.client import boolean
import pygame
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

print("Seleccione el algoritmo de búsqueda")
print("1) Búsqueda por profundidad")
print("2) Búsqueda por anchura")
opcion = int(input())
print("Ingrese la prioridad de las acciones, ejemplo (Arriba,Abajo,Izquierda,Derecha)")
prio = input()

def leer_lab(filename):
    with open(filename, 'r') as f:
        content = f.read().splitlines()
    lab = []
    for line in content:
        row = [int(x) for x in line]
        lab.append(row)
    return lab

continuar = boolean

# Definir los colores que se utilizarán
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Definir el tamaño de los bloques que formarán el laberinto
BLOCK_WIDTH = 40
BLOCK_HEIGHT = 40

# Leer el laberinto desde un archivo de texto
filename = 'laberinto.txt'
maze = leer_lab(filename)

# Obtener el número de filas y columnas del laberinto
NUM_ROWS = len(maze)
NUM_COLS = len(maze[0])

# Calcular el tamaño de la pantalla en función del tamaño del laberinto
SCREEN_WIDTH = NUM_COLS * BLOCK_WIDTH
SCREEN_HEIGHT = NUM_ROWS * BLOCK_HEIGHT

# Inicializar Pygame
pygame.init()

# Crear la pantalla
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Laberinto")

# Dibujar el laberinto
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        if maze[row][col] == 0:
            pygame.draw.rect(screen, WHITE, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        else:
            pygame.draw.rect(screen, BLACK, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])

# Actualizar la pantalla
pygame.display.flip()

def bfs(maze, start, end, prio):
    queue = [start, ]
    visited = set()
    prodecessors = {}
    while queue:
        current = queue.pop(0)
        visited.add(current)
        if current == end:
            path = []
            while current != start:
                path.append(current)
                current = prodecessors[current]
            path.append(start)
            return prodecessors, path[::-1]
        for neighbor in get_neighbors(maze, current, priority=prio):
            if neighbor not in visited:
                queue.append(neighbor)
                prodecessors[neighbor] = current
                pygame.display.flip()
    return None

def dfs(maze, start, end, prio):
    nodo_raiz = start
    prodecessors = {}
    if end == nodo_raiz:
        return nodo_raiz
    frontera = [nodo_raiz, ]
    explorados = set()
    while frontera:
        nodo = frontera.pop()
        explorados.add(nodo)
        if nodo == end:
            path = []
            while nodo != start:
                path.append(nodo)
                nodo = prodecessors[nodo]
            path.append(start)
            return prodecessors, path[::-1]
        neighbors = get_neighbors(maze, nodo, prio)
        for neighbor in neighbors:
            if neighbor not in explorados and neighbor not in frontera:
                frontera.append(neighbor)
                prodecessors[neighbor] = nodo
                if neighbor == end:
                    path = []
                    while neighbor != start:
                        path.append(neighbor)
                        neighbor = prodecessors[neighbor]
                    path.append(start)
                    return prodecessors, path[::-1]
    return None

def print_search_tree(prodecessors):
    G = nx.DiGraph()
    for child, parent in prodecessors.items():
        G.add_edge(child, parent)
    pos = nx.spring_layout(G, pos=nx.spectral_layout(G), scale=3)
    nx.draw(G, pos, with_labels=True, node_color='grey', node_size=700, font_size=8, arrowsize=10)
    plt.show()

# Esperar a que el usuario cierre la ventana
start_pos = None
end_pos = None
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            row = pos[1] // BLOCK_HEIGHT
            col = pos[0] // BLOCK_WIDTH
            if start_pos is None:
                start_pos = (row, col)
                pygame.draw.rect(screen, GREEN, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
            elif end_pos is None and (row, col) != start_pos:
                end_pos = (row, col)
                pygame.draw.rect(screen, RED, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
                done = True
    pygame.display.flip()

print("El punto inicial es:", start_pos)
print("La meta es:", end_pos)

def plot_graph(prodecessors):
    G = nx.DiGraph()
    for child, parent in prodecessors.items():
        G.add_edge(parent, child)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()

def get_neighbors(maze, point, priority):
    my_string = priority
    my_list = my_string.split(",")
    neighbors = []
    row, col = point
    for prio in my_list:
        if prio == "Arriba":
            if row > 0 and not maze[row - 1][col]:
                neighbors.append((row - 1, col))
        if prio == "Izquierda":
            if col > 0 and not maze[row][col - 1]:
                neighbors.append((row, col - 1))
        if prio == "Abajo":
            if row < len(maze) - 1 and not maze[row + 1][col]:
                neighbors.append((row + 1, col))
        if prio == "Derecha":
            if col < len(maze[0]) - 1 and not maze[row][col + 1]:
                neighbors.append((row, col + 1))
    return neighbors

# Define una función para verificar si dos nodos están a una distancia de 1
def is_near(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return abs(x1 - x2) + abs(y1 - y2) == 1

def draw_search_path(screen, prodecessors):
    for node, parent in prodecessors.items():
        if parent is not None:
            x1, y1 = parent
            x2, y2 = node
            pygame.draw.line(screen, GREEN, [y1 * BLOCK_WIDTH + BLOCK_WIDTH // 2, x1 * BLOCK_HEIGHT + BLOCK_HEIGHT // 2], [y2 * BLOCK_WIDTH + BLOCK_WIDTH // 2, x2 * BLOCK_HEIGHT + BLOCK_HEIGHT // 2], 2)
            pygame.display.flip()
            neighbors = get_neighbors(maze, node, prio)
            if len(neighbors) > 1:
                time.sleep(2)  # Espera 2 segundos en una bifurcación
            else:
                time.sleep(0.1)

def draw_path(screen, path, color):
    for i in range(len(path) - 1):
        current_square = path[i]
        next_square = path[i + 1]
        pygame.draw.rect(screen, color, [next_square[1] * BLOCK_WIDTH, next_square[0] * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        pygame.draw.line(screen, GREEN, (current_square[1] * BLOCK_WIDTH + BLOCK_WIDTH // 2, current_square[0] * BLOCK_HEIGHT + BLOCK_HEIGHT // 2),
                         (next_square[1] * BLOCK_WIDTH + BLOCK_WIDTH // 2, next_square[0] * BLOCK_HEIGHT + BLOCK_HEIGHT // 2), 4)
        pygame.display.flip()
        pygame.time.delay(200)

if opcion == 2:
    prodecessors, path = bfs(maze, start_pos, end_pos, prio)
    long = len(path)
    print("El costo es de:", long - 1)
    print("El camino por medio de búsqueda por anchura es", path)
    draw_search_path(screen, prodecessors)
    draw_path(screen, path, BLUE)
    print_search_tree(prodecessors)
else:
    prodecessors, path = dfs(maze, start_pos, end_pos, prio)
    long = len(path)
    print("El costo es de:", long - 1)
    print("El camino por medio de búsqueda por profundidad es", path)
    draw_search_path(screen, prodecessors)
    draw_path(screen, path, BLUE)
    print_search_tree(prodecessors)

pygame.quit()
