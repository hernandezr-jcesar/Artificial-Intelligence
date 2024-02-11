import pygame
import math
from queue import PriorityQueue

# Definir el tamaño del laberinto y la ventana
WIDTH = 600
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Resolución de Laberinto")

# Colores
WHITE = (255, 255, 255)
GRAY = (192, 192, 192)
PEACH = (255, 218, 185)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PURPLE_BLACK = (194, 55, 219)

# Mapeo de valores a colores
COLOR_MAP = {
    0: GRAY,
    1: PEACH,
    2: BLUE,
    3: YELLOW,
    4: GREEN,
    8: RED,
    9: PURPLE_BLACK
}

# Clase para representar cada nodo en el laberinto
class Node:
    def __init__(self, row, col, width, value):
        self.row = row
        self.col = col
        self.x = col * width  # Cambio en la coordenada X
        self.y = row * width  # Cambio en la coordenada Y
        self.color = COLOR_MAP[value]
        self.neighbors = []
        self.width = width

    def get_position(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == (0, 128, 0)

    def is_end(self):
        return self.color == (255, 165, 0)

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = (0, 128, 0)

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = (255, 165, 0)

    def make_path(self):
        self.color = (0, 255, 0)

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid, maze):
        self.neighbors = []
        if self.row < len(grid) - 1 and maze[self.row + 1][self.col] != 0:
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and maze[self.row - 1][self.col] != 0:
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < len(grid[0]) - 1 and maze[self.row][self.col + 1] != 0:
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and maze[self.row][self.col - 1] != 0:
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


# Función para calcular la heurística (distancia Manhattan)
def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


# Algoritmo de búsqueda A*
def astar(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = h(start.get_position(), end.get_position())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_position(), end.get_position())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


# Función para reconstruir el camino
def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


# Función principal para configurar el laberinto y la interfaz gráfica
def main(win, width, maze_file):
    # Lee el laberinto desde el archivo de texto y limpia los datos
    with open(maze_file, 'r') as file:
        maze = [[int(char) for char in line.strip() if char in '0123489'] for line in file]

    ROWS = len(maze)  # Número de filas y columnas en el laberinto
    grid = []
    gap = width // ROWS

    start = None
    end = None

    for i in range(ROWS):
        grid.append([])
        for j in range(ROWS):
            node = Node(i, j, gap, maze[i][j])
            grid[i].append(node)

            if maze[i][j] == 8:  # 2 representa el inicio
                start = node
            elif maze[i][j] == 9:  # 3 representa el final
                end = node

    run = True
    started = False

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            """""
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                col, row = get_clicked_position(pos, ROWS, width)  # Cambio en el orden de col y row
                node = grid[row][col]  # Cambio en el orden de col y row
                if not start and node != end:
                    start = node
                    start.make_start()
                elif not end and node != start:
                    end = node
                    end.make_end()
                elif node != end and node != start:
                    node.make_barrier()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                col, row = get_clicked_position(pos, ROWS, width)  # Cambio en el orden de col y row
                node = grid[row][col]  # Cambio en el orden de col y row
                node.reset()
                if node == start:
                    start = None
                elif node == end:
                    end = None
            """
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not started:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid, maze)  # Pasamos la matriz maze como argumento

                    astar(lambda: draw(win, grid, ROWS, width), grid, start, end)

    pygame.quit()


# Función para dibujar el laberinto y los nodos
def draw(win, grid, rows, width):
    win.fill(GRAY)  # Fondo gris

    for row in grid:
        for node in row:
            node.draw(win)

    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, BLACK, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, width))

    pygame.display.update()


# Función para obtener la posición del clic del mouse
"""""
def get_clicked_position(pos, rows, width):
    gap = width // rows
    x, y = pos

    row = x // gap
    col = y // gap

    return row, col
"""""

if __name__ == "__main__":
    maze_file = ".\P1\laberinto.txt"  # Reemplaza con la ruta de tu archivo de laberinto
    main(WIN, WIDTH, maze_file)
