from queue import PriorityQueue
import math
import pygame
import time
from os import system
import networkx as nx
import matplotlib.pyplot as plt


print("**********************************************")
print("+++++HUMANO++++++")
print("Indique el costo que va tener el humano si atraviesa por la tierra:")
ch_tierra = int(input())
print("Indique el costo que va tener el humano si atraviesa por el agua:")
ch_agua = int(input())
print("Indique el costo que va tener el humano si atraviesa por la arena:")
ch_arena = int(input())
print("Indique el costo que va tener el humano si atraviesa por el bosque:")
ch_bosque = int(input())
#system("cls")
print("++++++Pulpo+++++")
print("Indique el costo que va tener el pulpo si atraviesa por la tierra:")
cp_tierra = int(input())
print("Indique el costo que va tener el pulpo si atraviesa por el agua:")
cp_agua = int(input())
print("Indique el costo que va tener el pulpo si atraviesa por la arena:")
cp_arena = int(input())
print("Indique el costo que va tener el pulpo si atraviesa por el bosque:")
cp_bosque = int(input())
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
GREEN = (0,255,0)
ROJO = (255, 0, 0)

def prueba():
    return ch_agua

print(prueba())
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
            if terrain_type == 1:
                return 999
            elif terrain_type == 2:
                return ch_tierra  # Tierra para el agente A
            elif terrain_type == 3:
                return ch_agua  # Agua para el agente A
            elif terrain_type == 4:
                return ch_arena  # Arena para el agente A
            elif terrain_type == 5:
                return ch_bosque  # Bosque para el agente A
        elif personaje == 2:
            if terrain_type == 1:
                return 999  # Montaña para el agente B
            elif terrain_type == 2:
                return cp_tierra  # Tierra para el agente B
            elif terrain_type == 3:
                return cp_agua  # Agua para el agente B
            elif terrain_type == 4:
                return cp_arena  # Arena para el agente B
            elif terrain_type == 5:
                return cp_bosque  # Bosque para el agente B
        else:
            raise ValueError('Agente desconocido')

    def heuristic(self, pos, goal):
        (x1, y1) = pos
        (x2, y2) = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def astar_search(start, goal, graph,agent):
    frontier = PriorityQueue() #Cola de prioridad donde se almacenan los nodos a explorar
    frontier.put(start, 0)
    came_from = {} #Almacena los nodos visitados y sus nodos anteriores en el camino mas corto
    cost_so_far = {} #Almacena los costos acumulados desde el nodo de inicio hasta cada nodo visitado
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get() #Se extrae el nodo de mayor prioridad

        if current == goal:
            break

        for next in graph.neighbors(current):
            #Se calcula el nuevo costo acumulado desde el nodo inicial hasta el vecino actual sumando el 
            #costo acumulado desde el nodo de inicio hasta "current" y el costo para moverse de current a "next"
            new_cost = cost_so_far[current] + graph.cost(current, next,agent)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                #Se calcula la prioridad del vecino next sumando el nuevo costo acumulado con la estimación heuristica
                priority = new_cost + graph.heuristic(next, goal)
                #Se agrega el vecino a frontier con la prioridad calculada
                frontier.put(next, priority)
                #Se actualiza came lo que indica que el nodo current es el nodo anterior en el camino más corto hacia "next"
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

# Leer el laberinto desde un archivo de texto
filename = 'laberinto.txt'
maze = leer_lab(filename)

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
# Dibujar el laberinto
for row in range(NUM_ROWS):
    for col in range(NUM_COLS):
        if maze[row][col] == 1:
            pygame.draw.rect(screen, MONTANA, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        if maze[row][col] == 2:
            pygame.draw.rect(screen, TIERRA, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        if maze[row][col] == 3:
            pygame.draw.rect(screen, AGUA, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        if maze[row][col] == 4:
            pygame.draw.rect(screen, ARENA, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        if maze[row][col] == 5:
            pygame.draw.rect(screen, BOSQUE, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        


# Actualizar la pantalla
pygame.display.flip()

# Definir el texto a escribir para cada cuadrado
pulpo_text = 'O'
humano_text = 'H'
llave_text = 'K'
portal_text = 'P'
temple_text = 'D'

# Definir las posiciones iniciales
pulpo_pos = None
humano_pos = None
llave_pos = None
portal_pos = None
temple_pos = None 
done = False
# Esperar a que el usuario cierre la ventana
print("A continuacion seleccione la ubicacion inicial del")
print("-Pulpo\n-Humano\n-La llave\n-Portal\n-Templo oscuro")
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
            if pulpo_pos is None and maze[row][col] != 1:
                pulpo_pos = (row,col)
                pygame.draw.rect(screen, PULPO, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
                text_surface = font.render(pulpo_text, True, BLACK)
                text_x = col * BLOCK_WIDTH + (BLOCK_WIDTH // 2) - (text_surface.get_width() // 2)
                text_y = row * BLOCK_HEIGHT + (BLOCK_HEIGHT // 2) - (text_surface.get_height() // 2)
                screen.blit(text_surface, (text_x, text_y)) 
            elif humano_pos is None and maze[row][col] != 1:
                humano_pos = (row,col)
                pygame.draw.rect(screen, HUMANO, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
                text_surface = font.render(humano_text, True, BLACK)
                text_x = col * BLOCK_WIDTH + (BLOCK_WIDTH // 2) - (text_surface.get_width() // 2)
                text_y = row * BLOCK_HEIGHT + (BLOCK_HEIGHT // 2) - (text_surface.get_height() // 2)
                screen.blit(text_surface, (text_x, text_y))      
            elif llave_pos is None and maze[row][col] != 1:
                llave_pos = (row,col)
                pygame.draw.rect(screen, LLAVE, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
                text_surface = font.render(llave_text, True, BLACK)
                text_x = col * BLOCK_WIDTH + (BLOCK_WIDTH // 2) - (text_surface.get_width() // 2)
                text_y = row * BLOCK_HEIGHT + (BLOCK_HEIGHT // 2) - (text_surface.get_height() // 2)
                screen.blit(text_surface, (text_x, text_y))           
            elif portal_pos is None and maze[row][col] != 1:
                portal_pos = (row,col)
                text_surface = font.render(portal_text, True, BLACK)
                pygame.draw.rect(screen, LLAVE, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])               
                text_x = col * BLOCK_WIDTH + (BLOCK_WIDTH // 2) - (text_surface.get_width() // 2)
                text_y = row * BLOCK_HEIGHT + (BLOCK_HEIGHT // 2) - (text_surface.get_height() // 2)
                screen.blit(text_surface, (text_x, text_y)) 
            elif temple_pos is None and maze[row][col] != 1:
                temple_pos = (row,col)
                pygame.draw.rect(screen, LLAVE, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
                text_surface = font.render(temple_text, True, BLACK)
                text_x = col * BLOCK_WIDTH + (BLOCK_WIDTH // 2) - (text_surface.get_width() // 2)
                text_y = row * BLOCK_HEIGHT + (BLOCK_HEIGHT // 2) - (text_surface.get_height() // 2)
                screen.blit(text_surface, (text_x, text_y))
                done = True
    pygame.display.flip()
time.sleep(2)


# Salir de Pygame
graph = MapGraph(maze)

print ("El punto inicial del pulpo es: ",pulpo_pos)
print ("El punto inicial del humano es: ",humano_pos)
print ("La posicion de la llave es: ",llave_pos)
print ("La posicion del templo oscuro ", temple_pos)
print ("La posicion del portal es: ",portal_pos)
print ("El costo para el humano en la \ntierra es de: ",ch_tierra,"\n por el agua es de: ",ch_agua,"\n por la arena es de: ",ch_arena,"\n por el bosque es de: ",ch_bosque)
print ("El costo para el pulpo en la \ntierra es de: ",cp_tierra,"\n por el agua es de: ",cp_agua,"\n por la arena es de: ",cp_arena,"\n por el bosque es de: ",cp_bosque)

def draw_path(screen, path):
    """
    Dibuja el camino de un agente en la interfaz gráfica.
    
    Args:
        screen (pygame.Surface): Superficie de la pantalla donde se dibujará.
        path (list): Lista de posiciones (coordenadas) que forman el camino.
        color (tuple): Color RGB del camino.
    """
    for square in path:        
        x,y=square
        #pygame.draw.rect(screen, ROJO, (pos[0] *BLOCK_WIDTH, pos[1] * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT))
        pygame.draw.rect(screen, ROJO, [y * BLOCK_WIDTH, x * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
        pygame.display.flip()
    time.sleep(2)
        





#Humano I-P
start = humano_pos
came_fromIPH, cost_so_farIPH = astar_search(start, portal_pos, graph,1)
pathIPH = [portal_pos]
while pathIPH[-1] != start:
    pathIPH.append(came_fromIPH[pathIPH[-1]])
pathIPH.reverse()
print("Camino encontrado para el humano I-P:", pathIPH)
print("Costo total:", cost_so_farIPH[portal_pos])

#Humano I-K
start = humano_pos
came_fromIKH, cost_so_farIKH = astar_search(start, llave_pos, graph,1)
pathIKH = [llave_pos]
while pathIKH[-1] != start:
    pathIKH.append(came_fromIKH[pathIKH[-1]])
pathIKH.reverse()
print("Camino encontrado para el humano I-K:", pathIKH)
print("Costo total:", cost_so_farIKH[llave_pos])

#Humano I-D
start = humano_pos
came_fromIDH, cost_so_farIDH = astar_search(start, temple_pos, graph,1)
pathIDH = [temple_pos]
while pathIDH[-1] != start:
    pathIDH.append(came_fromIDH[pathIDH[-1]])
pathIDH.reverse()
print("Camino encontrado para el humano I-D:", pathIDH)
print("Costo total:", cost_so_farIDH[temple_pos])

#Humano K-D
came_fromKDH, cost_so_farKDH = astar_search(llave_pos, temple_pos, graph,1)
pathKDH = [temple_pos]
while pathKDH[-1] != llave_pos:
    pathKDH.append(came_fromKDH[pathKDH[-1]])
pathKDH.reverse()
print("Camino encontrado para el humano K-D:", pathKDH)
print("Costo total:", cost_so_farKDH[temple_pos])

#Humano D-K
came_fromDKH, cost_so_farDKH = astar_search(temple_pos, llave_pos, graph,1)
pathDKH = [llave_pos]
while pathDKH[-1] != temple_pos:
    pathDKH.append(came_fromDKH[pathDKH[-1]])
pathDKH.reverse()
print("Camino encontrado para el humano D-K:", pathDKH)
print("Costo total:", cost_so_farDKH[llave_pos])

#Humano K-P
came_fromKPH, cost_so_farKPH = astar_search(llave_pos, portal_pos, graph,1)
pathKPH = [portal_pos]
while pathKPH[-1] != llave_pos:
    pathKPH.append(came_fromKPH[pathKPH[-1]])
pathKPH.reverse()
print("Camino encontrado para el humano K-P:", pathKPH)
print("Costo total:", cost_so_farKPH[portal_pos])

#Humano D-P
came_fromDPH, cost_so_farDPH = astar_search(temple_pos, portal_pos, graph,1)
pathDPH = [portal_pos]
while pathDPH[-1] != temple_pos:
    pathDPH.append(came_fromDPH[pathDPH[-1]])
pathDPH.reverse()
print("Camino encontrado para el humano D-P:", pathDPH)
print("Costo total:", cost_so_farDPH[portal_pos])


#************************Pulpo I-P
start = pulpo_pos
came_fromIPP, cost_so_farIPP = astar_search(start, portal_pos, graph,2)
pathIPP = [portal_pos]
while pathIPP[-1] != start:
    pathIPP.append(came_fromIPP[pathIPP[-1]])
pathIPP.reverse()
print("Camino encontrado para el pulpo I-P:", pathIPP)
print("Costo total:", cost_so_farIPP[portal_pos])

#Pulpo I-K
came_fromIKP, cost_so_farIKP = astar_search(start, llave_pos, graph,2)
pathIKP = [llave_pos]
while pathIKP[-1] != start:
    pathIKP.append(came_fromIKP[pathIKP[-1]])
pathIKP.reverse()
print("Camino encontrado para el pulpo I-K:", pathIKP)
print("Costo total:", cost_so_farIKP[llave_pos])

#Pulpo I-D
came_fromIDP, cost_so_farIDP = astar_search(start, temple_pos, graph,2)
pathIDP = [temple_pos]
while pathIDP[-1] != start:
    pathIDP.append(came_fromIDP[pathIDP[-1]])
pathIDP.reverse()
print("Camino encontrado para el pulpo I-D:", pathIDP)
print("Costo total:", cost_so_farIDP[temple_pos])

#Pulp K-D
came_fromKDP, cost_so_farKDP = astar_search(llave_pos, temple_pos, graph,2)
pathKDP = [temple_pos]
while pathKDP[-1] != llave_pos:
    pathKDP.append(came_fromKDP[pathKDP[-1]])
pathKDP.reverse()
print("Camino encontrado para el pulpo k-D:", pathKDP)
print("Costo total:", cost_so_farKDP[temple_pos])

#Pulpo D-K
came_fromDKP, cost_so_farDKP = astar_search(temple_pos, llave_pos, graph,2)
pathDKP = [llave_pos]
while pathDKP[-1] != temple_pos:
    pathDKP.append(came_fromDKP[pathDKP[-1]])
pathDKP.reverse()
print("Camino encontrado para el pulpo D-K:", pathDKP)
print("Costo total:", cost_so_farDKP[llave_pos])

#Pulpo K-P
came_fromKPP, cost_so_farKPP = astar_search(llave_pos, portal_pos, graph,2)
pathKPP = [portal_pos]
while pathKPP[-1] != llave_pos:
    pathKPP.append(came_fromKPP[pathKPP[-1]])
pathKPP.reverse()
print("Camino encontrado para el pulpo k-P:", pathKPP)
print("Costo total:", cost_so_farKPP[portal_pos])

#Pulpo D-P
came_fromDPP, cost_so_farDPP = astar_search(temple_pos, portal_pos, graph,2)
pathDPP = [portal_pos]
while pathDPP[-1] != temple_pos:
    pathDPP.append(came_fromDPP[pathDPP[-1]])
pathDPP.reverse()
print("Camino encontrado para el pulpo D-P:", pathDPP)
print("Costo total:", cost_so_farDPP[portal_pos])



#Costo Humano de I-P
costoHumanoIP= cost_so_farIPH[portal_pos]
#Costo Humano de I-K-P
costoHumanoIKP = cost_so_farIKH[llave_pos]+cost_so_farKPH[portal_pos]
#Costo Humano de I-D-P
costoHumanoIDP = cost_so_farIDH[temple_pos]+cost_so_farDPH[portal_pos]
#Costo Humano de I-K-D-P
costoHumanoIKDP=cost_so_farIKH[llave_pos]+cost_so_farKDH[temple_pos]+cost_so_farDPH[portal_pos]
#Costo Humano de I-D-K-P
costoHumanoIDKP=cost_so_farIDH[temple_pos]+cost_so_farDKH[llave_pos]+cost_so_farKPH[portal_pos]


#Costo Pulpo de I-P
costoPulpoIP= cost_so_farIPP[portal_pos]
#Costo Pulpo de I-K-P
costoPulpoIKP = cost_so_farIKP[llave_pos]+cost_so_farKPP[portal_pos]
#Costo Pulpo de I-D-P
costoPulpoIDP = cost_so_farIDP[temple_pos]+cost_so_farDPP[portal_pos]
#Costo Pulpo de I-K-D-P
costoPulpoIKDP=cost_so_farIKP[llave_pos]+cost_so_farKDP[temple_pos]+cost_so_farDPP[portal_pos]
#Costo Pulpo de I-D-K-P
costoPulpoIDKP=cost_so_farIDP[temple_pos]+cost_so_farDKP[llave_pos]+cost_so_farKPP[portal_pos]


print("Humano:")
print
print("Ruta: I-P        Costo:",costoHumanoIP)
print("Ruta I-K-P       Costo:",costoHumanoIKP)
print("Ruta I-D-P       Costo:",costoHumanoIDP)
print("Ruta I-K-D-P     Costo:",costoHumanoIKDP)
print("Ruta I-D-K-P     Costo:",costoHumanoIDKP)


print("Pulpó:")
print
print("Ruta: I-P        Costo:",costoPulpoIP)
print("Ruta I-K-P       Costo:",costoPulpoIKP)
print("Ruta I-D-P       Costo:",costoPulpoIDP)
print("Ruta I-K-D-P     Costo:",costoPulpoIKDP)
print("Ruta I-D-K-P     Costo:",costoPulpoIDKP)


if (costoPulpoIDP + costoHumanoIKP) < (costoPulpoIKDP + costoHumanoIP) and \
   (costoPulpoIDP + costoHumanoIKP) < (costoPulpoIDKP + costoHumanoIP) and \
   (costoPulpoIDP + costoHumanoIKP) < (costoHumanoIDKP + costoPulpoIP) and \
   (costoPulpoIDP + costoHumanoIKP) < (costoHumanoIKDP + costoPulpoIP) and\
   (costoPulpoIDP + costoHumanoIKP) < (costoHumanoIDP+costoPulpoIKP):
    #pathfinal.update([pathIDP, pathDPP, pathIKH, pathKPH])
    print("El pulpo tomará la ruta I-D-P y el humano I-K-P")
    print("Costo:",costoPulpoIDP+costoHumanoIKP)
    print("Camino del humano:")
    print("Del punto inicial a la llave",pathIKH)
    draw_path(screen,pathIKH)
    print("De la llave al portal",pathKPH)
    draw_path(screen,pathKPH)
    print("")
    print("Camino del pulpo")
    print("Del punto inicial al templo oscuro:",pathIDP)
    draw_path(screen,pathIDP)
    print("De el templo oscuro al portal:",pathDPP)
    draw_path(screen,pathDPP)
elif (costoHumanoIDP + costoPulpoIKP) < (costoPulpoIKDP + costoHumanoIP) and \
     (costoHumanoIDP + costoPulpoIKP) < (costoPulpoIDKP + costoHumanoIP) and \
     (costoHumanoIDP + costoPulpoIKP) < (costoHumanoIDKP + costoPulpoIP) and \
     (costoHumanoIDP + costoPulpoIKP) < (costoHumanoIKDP + costoPulpoIP) and \
     (costoHumanoIDP + costoPulpoIKP) < (costoPulpoIDP + costoHumanoIKP):
    #pathfinal.update([pathIDH, pathDPH, pathIKP, pathKPP])
    print("El pulpo tomará la ruta I-K-P y el humano I-D-P")
    print("Costo",costoHumanoIDP + costoPulpoIKP)
    print("Camino del humano:")
    print("Punto inicial a Templo oscuro:",pathIDH)
    draw_path(screen,pathIDH)
    print("Del templo oscuro al Portal:",pathDPH)
    draw_path(screen,pathDPH)
    print("")
    print("Camino del pulpo:")
    print("Punto inical a la llave:",pathIKP)
    draw_path(screen,pathIKP)
    print("De la llave al portal",pathKPP)
    draw_path(screen,pathKPP)
elif (costoPulpoIKDP + costoHumanoIP) < (costoPulpoIDKP + costoHumanoIP) and \
     (costoPulpoIKDP + costoHumanoIP) < (costoHumanoIKDP + costoPulpoIP) and \
     (costoPulpoIKDP + costoHumanoIP) < (costoHumanoIDKP + costoPulpoIP) and \
     (costoPulpoIKDP + costoHumanoIP) < (costoHumanoIKP + costoPulpoIDP) and \
     (costoPulpoIKDP + costoHumanoIP) < (costoHumanoIDP + costoPulpoIKP):
    #pathfinal.update([pathIKP, pathKDP, pathDPP, pathIPH])
    print("El pulpo tomará la ruta I-K-D-P y el humano I-P")
    print("Costo:",costoPulpoIKDP + costoHumanoIP)
    print("Camino del humano")
    print("Del punto inicial al portal:",pathIPH)
    draw_path(screen,pathIPH)
    print("")
    print("Camino del pulpo")
    print("Del punto inical a la llave:",pathIKP)
    draw_path(screen,pathIKP)
    print("De la llave al templo oscuro",pathKDP)
    draw_path(screen,pathKDP)
    print("Del templo oscuro al portal",pathDPP)
    draw_path(screen,pathDPP)

elif (costoPulpoIDKP + costoHumanoIP) < (costoHumanoIKDP + costoPulpoIP) and \
     (costoPulpoIDKP + costoHumanoIP) < (costoHumanoIDKP + costoPulpoIP) and \
     (costoPulpoIDKP + costoHumanoIP) < (costoPulpoIKDP + costoHumanoIP) and \
     (costoPulpoIDKP + costoHumanoIP) < (costoHumanoIKP + costoPulpoIDP) and \
     (costoPulpoIDKP + costoHumanoIP) < (costoHumanoIDP + costoPulpoIKP) :
    #pathfinal.update([pathIDP, pathDKP, pathKPP, pathIPH])
    print("El pulpo tomará la ruta I-D-K-P y el humano I-P")
    print("Costo:",costoPulpoIDKP + costoHumanoIP)
    print("Camino del humano")
    print("Del punto inicial al portal:",pathIPH)
    draw_path(screen,pathIPH)
    print("")
    print("Camino del pulpo")
    print("Del punto inical al templo oscuro:",pathIDP)
    draw_path(screen,pathIDP)
    print("Del templo oscuro a la llave",pathDKP)
    draw_path(screen,pathDKP)
    print("De la llave al portal",pathKPP)
    draw_path(screen,pathKPP)
elif (costoHumanoIKDP + costoPulpoIP) < (costoPulpoIKDP + costoHumanoIP) and \
     (costoHumanoIKDP + costoPulpoIP) < (costoPulpoIDKP + costoHumanoIP) and \
     (costoHumanoIKDP + costoPulpoIP) < (costoHumanoIDKP + costoPulpoIP) and \
     (costoHumanoIKDP + costoPulpoIP) < (costoHumanoIDP + costoPulpoIKP) and \
     (costoHumanoIKDP + costoPulpoIP) < (costoHumanoIKP + costoPulpoIDP):
    #pathfinal.update([pathIKH, pathKDH, pathDPH, pathIPP])
    print("El humano tomará la ruta I-K-D-P y el pulpo I-P")
    print("Costo",costoHumanoIKDP + costoPulpoIP)
    print("Camino del humano")
    print("Del punto inical a la llave:",pathIKH)
    draw_path(screen,pathIKH)
    print("De la llave al templo oscuro",pathKDH)
    draw_path(screen,pathKDH)
    print("Del templo oscuro al portal",pathDPH)
    draw_path(screen,pathDPH)
    print("")
    print("Camino del pulpo")
    print("Del punto inicial al portal:",pathIPP)
    draw_path(screen,pathIPP)
   
elif (costoHumanoIDKP + costoPulpoIP) < (costoHumanoIKDP + costoPulpoIP) and \
     (costoHumanoIDKP + costoPulpoIP) < (costoPulpoIDKP + costoHumanoIP) and \
     (costoHumanoIDKP + costoPulpoIP) < (costoPulpoIKDP + costoHumanoIP) and \
     (costoHumanoIDKP + costoPulpoIP) < (costoHumanoIDP + costoPulpoIKP) and \
     (costoHumanoIDKP + costoPulpoIP) < (costoHumanoIKP + costoPulpoIDP) :
    #pathfinal.update([pathIDH, pathDKH, pathKPH, pathIPP])
    print("El humano tomará la ruta I-D-K-P y el pulpo I-P")
    print("Costo: ",costoHumanoIDKP + costoPulpoIP)
    print("Camino del humano")
    print("Del punto inical al templo oscuro:",pathIDH)
    draw_path(screen,pathIDH)
    print("Del templo oscuro a la llave",pathDKH)
    draw_path(screen,pathDKH)
    print("De la llave al portal",pathKPH)
    draw_path(screen,pathKPH)
    print("")
    print("Camino del pulpo")
    print("Del punto inicial al portal:",pathIPP)
    draw_path(screen,pathIPP)


