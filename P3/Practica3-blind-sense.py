from queue import PriorityQueue
import math
import pygame
from pygame.locals import *
import time
from os import system
import networkx as nx
import matplotlib.pyplot as plt



print("**********************************************")
print("Elige un personaje que usará :)")
print("1- El mono")
print("2- El octoupus")
personaje= int(input())
if personaje==1:
    print("Eres el mono...")
    time.sleep(2)
    ch_montaña = 99
    ch_tierra = 2
    ch_agua = 4
    ch_arena = 3
    ch_bosque = 1    

elif personaje==2:
    print("Eres el octopus")
    time.sleep(2)
    ch_montaña = 99
    ch_tierra = 2
    ch_agua = 1
    ch_arena = 0
    ch_bosque=3
    
# Definir los colores que se utilizarán
MONTANA = (0, 0, 0)
BOSQUE = (40, 180, 99)
AGUA = (52, 152, 219)
ARENA = (241, 196, 15)
TIERRA=(240, 178, 122)
PULPO=(125, 60, 152)
HUMANO=(30, 223, 226)
LLAVE = (255, 255, 0)
BLACK = (0, 0, 0)

class MapGraph:
    def __init__(self, maze):
        self.maze = maze
        self.width = len(maze[0])
        self.height = len(maze)

    def neighbors(self, pos):
        (x, y) = pos
        result = []

        if x > 0:
            result.append((x - 1, y))
        if x < self.width - 1:
            result.append((x + 1, y))
        if y > 0:
            result.append((x, y - 1))
        if y < self.height - 1:
            result.append((x, y + 1))

        return result

    def cost(self, current, next, personaje):
        (x1, y1) = current
        (x2, y2) = next
        terrain_type = self.maze[x2][y2]
        if personaje == 1:
            if terrain_type == 0:
                return 5  # Montaña para el agente A
            elif terrain_type == 1:
                return 2  # Tierra para el agente A
            elif terrain_type == 2:
                return 4  # Agua para el agente A
            elif terrain_type == 3:
                return 3  # Arena para el agente A
            elif terrain_type == 4:
                return 1  # Bosque para el agente A
            else:
                raise ValueError('Agente desconocido')

        elif personaje == 2:
            if terrain_type == 0:
                return 5  # Montaña para el agente B
            elif terrain_type == 1:
                return 2  # Tierra para el agente B
            elif terrain_type == 2:
                return 1  # Agua para el agente B
            elif terrain_type == 3:
                return 5  # Arena para el agente B
            elif terrain_type == 4:
                return 3  # Bosque para el agente B
            else:
                raise ValueError('Agente desconocido')

    def heuristic(self, pos, goal):
        (x1, y1) = pos
        (x2, y2) = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)



def astar_search(start, goal, graph, agent):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next, agent)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + graph.heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def leer_lab(filename):
    with open(filename,'r') as f:
        content = f.read().splitlines()
    lab = []
    for line in content:
        row = [int(x)for x in line]
        lab.append(row)
    return lab

def print_search_tree(came_from, explored_nodes):
    G = nx.DiGraph()
    labels = {}

    for node in came_from:
        G.add_node(node)
        labels[node] = str(node)
        if came_from[node] is not None:
            G.add_edge(came_from[node], node)

    pos = nx.spring_layout(G, k=0.2)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx(G, pos, with_labels=False, arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color='black')

    # Highlight explored nodes in a different color
    nx.draw_networkx_nodes(G, pos, nodelist=explored_nodes, node_color='red', node_size=50)

    plt.title('Árbol de búsqueda')
    plt.axis('off')
    plt.show()

#Imprimir los niveles de búsqueda por A*
def print_search_levels(graph, start, goal, explored_nodes):
    visited = set()
    current_level = 0
    current_level_nodes = [start]
    paraimprimir = [start]

    while current_level_nodes:
        print("Nivel", current_level)
        print("Nodos: ", current_level_nodes)
        next_level_nodes = []

        for node in current_level_nodes:
            if node == goal:
                # Si se alcanza el objetivo, terminamos la función
                return

            if node not in visited:
                visited.add(node)

                neighbors = graph.neighbors(node)
                if neighbors:  # Verificar si los vecinos existen
                    for neighbor in neighbors:
                        if neighbor not in visited and neighbor not in next_level_nodes:
                            next_level_nodes.append(neighbor)
                            paraimprimir.append(neighbor)

        current_level_nodes = next_level_nodes
        current_level += 1

    # Highlight explored nodes in a different color
    for node in explored_nodes:
        row, col = node
        pygame.draw.rect(screen, ROJO, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        pygame.display.flip()

# Leer el laberinto desde un archivo de texto
filename = '.\P3\laberinto.txt'
maze = leer_lab(filename)
print(maze[0][0])

# Definir el tamaño de los bloques que formarán el laberinto
BLOCK_WIDTH = 50
BLOCK_HEIGHT = 50
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

font = pygame.font.Font(None, 20)
pygame.display.set_caption("Laberinto")

'''
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        if maze[row][col] == 0:
            pygame.draw.rect(screen, MONTANA, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        if maze[row][col] == 1:
            pygame.draw.rect(screen, TIERRA, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        if maze[row][col] == 2:
            pygame.draw.rect(screen, AGUA, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        if maze[row][col] == 3:
            pygame.draw.rect(screen, ARENA, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        if maze[row][col] == 4:
            pygame.draw.rect(screen, BOSQUE, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])

'''
# Dibujar el laberinto


#Dibujar el camino recorrido por el agente a cada punto
# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 0, 255)
AMARILLO = (255, 255, 0)
AGENTE=(128,64,0)




# Actualizar la pantalla
pygame.display.flip()

# Crear una fuente para el texto


# Definir el texto a escribir para cada cuadrado
Inicio_text = 'I'
Final_text= 'F'

# Definir las posiciones iniciales
start_pos = None
final_pos = None
done = False
# Esperar a que el usuario cierre la ventana
print("A continuacion seleccione la ubicacion de:")
print("-Agente\n-Final\n")
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Obtener la posición del clic del mouse
            pos = pygame.mouse.get_pos()
            # Convertir la posición en índices de fila y columna
            row = pos[1] // BLOCK_HEIGHT
            col = pos[0] // BLOCK_WIDTH
            if start_pos is None and maze[row][col] != 1:
                start_pos = (row,col)
                pygame.draw.rect(screen, AGENTE, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
                text_surface = font.render(Inicio_text, True, BLACK)
                text_x = col * BLOCK_WIDTH + (BLOCK_WIDTH // 2) - (text_surface.get_width() // 2)
                text_y = row * BLOCK_HEIGHT + (BLOCK_HEIGHT // 2) - (text_surface.get_height() // 2)
                screen.blit(text_surface, (text_x, text_y)) 
            elif final_pos is None:
                final_pos = (row,col)
                pygame.draw.rect(screen, AMARILLO, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
                text_surface = font.render(Final_text, True, BLACK)
                text_x = col * BLOCK_WIDTH + (BLOCK_WIDTH // 2) - (text_surface.get_width() // 2)
                text_y = row * BLOCK_HEIGHT + (BLOCK_HEIGHT // 2) - (text_surface.get_height() // 2)
                screen.blit(text_surface, (text_x, text_y))      
                done = True

    
    
    pygame.display.flip()
print("Calculando...")
time.sleep(5)
screen.fill("#0A0A0A")


# Salir de Pygame
graph = MapGraph(maze)
AcumuladoH=0
AcumuladoP=0
print ("El punto inicial es: ",start_pos)
print ("El punto final es: ",final_pos)
print ("El costo para el agente en la \ntierra es de: ",ch_tierra,"\n por el agua es de: ",ch_agua,"\n por la arena es de: ",ch_arena,"\n por el bosque es de: ",ch_bosque)



def draw_path(screen, paths, maze):

    visitado = set()
    """
    Dibuja el camino de un agente en la interfaz gráfica.
    
    Args:
        screen (pygame.Surface): Superficie de la pantalla donde se dibujará.
        path (list): Lista de posiciones (coordenadas) que forman el camino.
        color (tuple): Color RGB del camino.
    """
    

    for path in paths:        
        x,y=path
        vecinos = graph.neighbors(path)
        for nodo in vecinos:
            i, j = nodo
            if maze[i][j] == 0:
                pygame.draw.rect(screen, MONTANA, [j * BLOCK_WIDTH, i * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
            if maze[i][j] == 1:
                pygame.draw.rect(screen, TIERRA, [j * BLOCK_WIDTH, i * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
            if maze[i][j] == 2:
                pygame.draw.rect(screen, AGUA, [j * BLOCK_WIDTH, i * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
            if maze[i][j] == 3:
                pygame.draw.rect(screen, ARENA, [j * BLOCK_WIDTH, i * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
            if maze[i][j] == 4:
                pygame.draw.rect(screen, BOSQUE, [j * BLOCK_WIDTH, i * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        #pygame.draw.rect(screen, ROJO, (pos[0] *BLOCK_WIDTH, pos[1] * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT))
    for square in paths:
        x, y = square
        pygame.draw.rect(screen, ROJO, [y * BLOCK_WIDTH, x * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        text = font.render("v" , True, "#0A0A0A")
        text_rect = text.get_rect(center=(x * BLOCK_WIDTH // 2, y * BLOCK_HEIGHT // 2))
        pygame.display.flip()
    time.sleep(2)







#Humano I-P
start = start_pos
came_fromIF, cost_so_farIF = astar_search(start, final_pos, graph, personaje)
pathIF = [final_pos]
explored_nodesIF = list(came_fromIF.keys()) + [final_pos]
while pathIF[-1] != start:
    pathIF.append(came_fromIF[pathIF[-1]])
pathIF.reverse()
#print(came_fromIF)
#print("Path if:", pathIF)
print("Camino encontrado para el agente I-F:", pathIF)
print("Costo total:", cost_so_farIF[final_pos])
print("Niveles de búsqueda:")
print_search_levels(graph, start, final_pos, explored_nodesIF)
draw_path(screen, pathIF, maze=maze)  # Dibujar camino del agente
print_search_tree(came_fromIF, explored_nodesIF)

