import pygame
import pygame_gui
import sys

# Configurar colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

GRAY = (192, 192, 192)
DARK_GRAY = (150, 150, 150)

BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

RED = (255, 0, 0)


# Configurar colores
COLORES = {
    '0': DARK_GRAY,
    '1': WHITE,
}

#with open('laberinto2.txt', 'r') as archivo:
#    numeros = [linea.split() for linea in archivo]
def leer_lab(filename):
    with open(filename,'r') as f:
        content = f.read().splitlines()
    lab = []
    for line in content:
        row = [int(x)for x in line]
        lab.append(row)
    return lab

def get_neighbor(maze,point):
    #Devuelve una lista de los vecinos válidos de un punto en el laberinto
    neighbors=[]
    row,col = point
    if row > 0 and not maze[row - 1][col]:
        neighbors.append((row - 1, col))  # Arriba
    if col > 0 and not maze[row][col - 1]:
        neighbors.append((row, col - 1))  # Izquierda
    if row < len(maze) - 1 and not maze[row + 1][col]:
        neighbors.append((row + 1, col))  # Abajo
    if col < len(maze[0]) - 1 and not maze[row][col + 1]:
        neighbors.append((row, col + 1))  # Derecha
    return neighbors

def moverlaberinto(maze, start, end):

    #explorados = set()  # Para ver si el nodo ya esta explorado o nel
    #path = [start, ]
    #if end == start:
    #    return start
    #neighbors = get_neighbor(maze, nodo)
    # Dimensiones del tablero
    ANCHO, ALTO = 600, 600
    TAMANO_CELDA = 40
    NUM_CELDAS = 15

    # Colores
    BLANCO = (255, 255, 255)
    NEGRO = (0, 0, 0)
    VERDE = (0, 255, 0)
    BLUE = (0, 0, 255)

    # Inicializa Pygame
    pygame.init()
    VENTANA = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption('Laberinto')

    # Posición inicial del jugador
    row, col = start_pos

    # Meta
    end_x, end_y = end_pos
    meta = (end_x, end_y)

    # Crea una lista para almacenar el camino
    camino = []

    while True:
        print(camino)
        actual = (row, col)
        print("Actual:", actual)
        # Obtiene los eventos del teclado
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # actual = (x,y)
            # Si el evento es una tecla presionada
            if actual == meta:
                # El jugador ha llegado a la meta
                print("¡Has ganado!")
                pygame.quit()
                sys.exit()

            if evento.type == pygame.KEYDOWN:

                if evento.key == pygame.K_UP and row > 0 and maze[row - 1][col] == 0:
                    if actual not in camino:
                        camino.append(actual)
                    row -= 1
                elif evento.key == pygame.K_DOWN and row < NUM_CELDAS - 1 and maze[row + 1][col] == 0:
                    if actual not in camino:
                        camino.append(actual)
                    row += 1
                elif evento.key == pygame.K_LEFT and col > 0 and maze[row][col - 1] == 0:
                    if actual not in camino:
                        camino.append(actual)
                    col -= 1
                elif evento.key == pygame.K_RIGHT and col < NUM_CELDAS - 1 and maze[row][col + 1] == 0:
                    if actual not in camino:
                        camino.append(actual)
                    col += 1

        # Limpia la pantalla
        VENTANA.fill(BLANCO)

        # Dibuja el laberinto
        for fila in range(NUM_CELDAS):
            for columna in range(NUM_CELDAS):
                celda = (fila, columna)
                if fila == row and columna == col:
                    pygame.draw.rect(VENTANA, VERDE,
                                     (columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA))
                    pygame.draw.rect(VENTANA, NEGRO,
                                     [columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA], 1)
                elif fila == end_x and columna == end_y:
                    pygame.draw.rect(VENTANA, RED,
                                     (columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA))
                    pygame.draw.rect(VENTANA, NEGRO,
                                     [columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA], 1)

                elif maze[fila][columna] == 1:
                    pygame.draw.rect(VENTANA, NEGRO,
                                     (columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA))
                elif celda in camino:
                    pygame.draw.rect(VENTANA, BLUE,
                                     (columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA))

                elif maze[fila][columna] == 0:
                    pygame.draw.rect(VENTANA, BLANCO,
                                     (columna * TAMANO_CELDA, fila * TAMANO_CELDA, TAMANO_CELDA, TAMANO_CELDA))

        # Actualiza la pantalla
        pygame.display.update()
    return None




# Definir la posición donde quieres imprimir el array
posicion_x = 40
posicion_y = 40

# Definir el tamaño de los bloques que formarán el laberinto
BLOCK_WIDTH = 40
BLOCK_HEIGHT = 40



# Leer el laberinto desde un archivo de texto
filename = ".\Practicas\P1\laberinto.txt"
maze = leer_lab(filename)
#print(maze)

# Obtener el número de filas y columnas del laberinto
NUM_ROWS = len(maze)
NUM_COLS = len(maze[0])

# Calcular el tamaño del tabl9era en función del tamaño del laberinto
ANCHO_TABLERO  = (NUM_COLS + 1) * BLOCK_WIDTH
ALTO_TABLERO = (NUM_ROWS + 1) * BLOCK_HEIGHT

# Calcular el tamaño de la pantalla
ANCHO_VENTANA  = NUM_COLS * BLOCK_WIDTH + BLOCK_WIDTH + 400
ALTO_VENTANA = NUM_ROWS * BLOCK_HEIGHT + BLOCK_HEIGHT



# Definir el tamaño del tablero y de cada celda
TAMANO_CELDA = BLOCK_WIDTH // NUM_ROWS + 1

# Inicializar Pygame
pygame.init()

# Inicializar la ventana
ventana = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
pygame.display.set_caption('Laberinto')

# Inicializar el administrador de interfaz de usuario de pygame_gui
administrador = pygame_gui.UIManager((ANCHO_VENTANA, ALTO_VENTANA))

# Crear una fuente para el texto
fuente_info = pygame.font.Font(None, 24)

# Crear una fuente para entrada de texto
fuente_input = pygame.font.Font(None, 32)


# Función para obtener la letra de la columna a partir de su posición numérica
def obtener_letra_columna(columna):
    return chr(columna - 1 + ord('A'))


# Crear una lista de números del 1 al 15
numeros = [str(i) for i in range(1, 16)]

# Crear una lista de letras de la A a la O
letras = [chr(65 + i) for i in range(15)]

# Crear un menú desplegable para elegir números
menu_numeros = pygame_gui.elements.UIDropDownMenu(options_list=numeros,
                                                  starting_option=numeros[0],
                                                  relative_rect=pygame.Rect(
                                                      (ANCHO_TABLERO + 100, 330, 100, 30)),
                                                  manager=administrador)

# Crear un menú desplegable para elegir letras
menu_letras = pygame_gui.elements.UIDropDownMenu(options_list=letras,
                                                 starting_option=letras[0],
                                                 relative_rect=pygame.Rect(
                                                     (ANCHO_TABLERO + 200, 330, 100, 30)),
                                                 manager=administrador)

# Crear un botón para realizar la acción
boton_cambiar = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((ANCHO_TABLERO + 120, 110, 200, 30)),
    text='Cambiar valor', manager=administrador)

# Crear un botón para realizar la acción
#boton_seleccionar_inicio = pygame_gui.elements.UIButton(
#    relative_rect=pygame.Rect((ANCHO_TABLERO + 120, 190, 200, 30)),
#    text='Seleccionar Inicio', manager=administrador)

# Crear un botón para realizar la acción
#boton_seleccionar_final = pygame_gui.elements.UIButton(
#    relative_rect=pygame.Rect((ANCHO_TABLERO + 120, 270, 200, 30)),
#    text='Seleccionar Final', manager=administrador)

# Crear un botón para empezar movimiento
boton_iniciar = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((ANCHO_TABLERO + 120, 20, 200, 30)),
    text='INICIAR', manager=administrador)
###################################################################################################################
camino = []
start_pos=None
end_pos = None
done = False
# Loop principal del juego
ejecutando = True
while ejecutando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            # Salir del juego
            pygame.quit()
            sys.exit()
        if evento.type == pygame.MOUSEBUTTONDOWN:
            # Obtener la posición del clic del mouse
            pos = pygame.mouse.get_pos()
            # Convertir la posición en índices de fila y columna
            row = (pos[1] // BLOCK_HEIGHT)-1
            col = (pos[0] // BLOCK_WIDTH)-1
            if start_pos is None:
                start_pos = (row,col)
                # Resaltar la casilla seleccionada
                #pygame.draw.rect(ventana, GREEN, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
            elif end_pos is None and (row,col) != start_pos:
                end_pos = (row,col)
                # Resaltar la casilla seleccionada
                #pygame.draw.rect(ventana, [col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT])
                done = True
        # Actualizar el administrador de interfaz de usuario con eventos
        administrador.process_events(evento)

        # Obtener los valores seleccionados de los menús desplegables
        numero_seleccionado = menu_numeros.selected_option
        letra_seleccionada = menu_letras.selected_option

        col = int(ord(letra_seleccionada) - ord('A') + 1) - 1
        row = int(numero_seleccionado) - 1
        # Verificar si se hace clic en el botón de cambiar
        if evento.type == pygame_gui.UI_BUTTON_PRESSED:
            if evento.ui_element == boton_cambiar:
                #print(col,row)
                if maze[row][col] == 1:
                    maze[row][col] = 0
                else:
                    maze[row][col] = 1
                print(f'Cambiando valor de la celda: {numero_seleccionado}{letra_seleccionada}')
            """
            if evento.ui_element == boton_seleccionar_inicio:
                if start_pos == None:

                    #start_pos = (row, col)
                    #print(f'Celda de inicio seleccionada: {numero_seleccionado}{letra_seleccionada}')
                    #print(row, col)

                    if maze[row][col] == 0:
                        start_pos = (row, col)
                        print(f'Celda de inicio seleccionada: {numero_seleccionado}{letra_seleccionada}')
                        print(row, col)
                        print(start_pos)
                    else:
                        print("Ese no es un camino")
                        print(row, col)


            if evento.ui_element == boton_seleccionar_final:
                if end_pos == None:
                    if maze[row][col] == 0:
                        end_pos = (row, col)
                        print(f'Celda final seleccionada: {numero_seleccionado}{letra_seleccionada}')
                        print(row, col)
                    else:
                        print("Ese no es un camino")
                        print(row, col)
            """
            if evento.ui_element == boton_iniciar and start_pos != None and end_pos != None:
                print(f'Jugando!!!: {numero_seleccionado}{letra_seleccionada}')
                # Salir del juego
                pygame.quit()
                moverlaberinto(maze,start_pos,end_pos)



        ###########################################################################################################
        # Dibujar tablero
        # Actualizar la interfaz de usuario
        administrador.update(30)
        # Limpiar pantalla
        ventana.fill(GRAY)
        # Dibujar la interfaz de usuario
        administrador.draw_ui(ventana)


        # Dibujar letras (columnas)
        letra = ord('A')
        for i in range(1, 16):
            pygame.draw.rect(ventana, BLACK, [i * BLOCK_WIDTH, 0, BLOCK_WIDTH, BLOCK_HEIGHT], 1)
            fuente = pygame.font.Font(None, 36)
            texto = fuente.render(chr(letra), True, BLACK)
            ventana.blit(texto, (i * BLOCK_WIDTH + 15, 5))
            letra += 1

        # Dibujar filas (números)
        for i in range(16):
            if i != 0:
                pygame.draw.rect(ventana, BLACK, [0, i * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT], 1)
                fuente = pygame.font.Font(None, 36)
                texto = fuente.render(str(i), True, BLACK)
                ventana.blit(texto, (5, i * BLOCK_WIDTH + 10))

        # Dibujar el laberinto
        for fila in range(NUM_ROWS):
            for columna in range(NUM_COLS):
                x = posicion_x + (columna * BLOCK_WIDTH)  # Ajusta el tamaño del elemento del array según sea necesario
                y = posicion_y + (fila * BLOCK_HEIGHT)
                """
                elif maze[row][col] == 2:
                    pygame.draw.rect(ventana, GREEN, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT])
                    pygame.draw.rect(ventana, BLACK, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT], 1)
                elif maze[row][col] == 3:
                    pygame.draw.rect(ventana, RED, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT])
                    pygame.draw.rect(ventana, BLACK, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT], 1)
                """
                nodo = (fila,columna)
                if nodo == start_pos:
                    pygame.draw.rect(ventana, GREEN, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT])
                    pygame.draw.rect(ventana, BLACK, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT], 1)
                elif nodo == end_pos:
                    pygame.draw.rect(ventana, RED, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT])
                    pygame.draw.rect(ventana, BLACK, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT], 1)
                elif nodo in camino:
                    pygame.draw.rect(ventana, BLUE, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT])
                    pygame.draw.rect(ventana, BLACK, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT], 1)
                elif maze[fila][columna] == 0:
                    pygame.draw.rect(ventana, WHITE, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT])
                    pygame.draw.rect(ventana, BLACK, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT], 1)
                else:
                    pygame.draw.rect(ventana, DARK_GRAY, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT])
                    pygame.draw.rect(ventana, BLACK, [x, y, BLOCK_WIDTH, BLOCK_HEIGHT], 1)

        #################################################################################################################
        # para cambiar valores
        cambiarValor_texto = fuente_info.render(f'¿Deseas cambiar algun valor de este laberinto?', True, (0, 0, 0))
        ventana.blit(cambiarValor_texto, (ANCHO_TABLERO + 10, 80))

        #################################################################################################################
        #PARA PONER TEXTOS
        # para poner inicio y final
        #guardarInicio_texto = fuente_info.render(f'¿Deseas seleccionar el punto de inicio?', True, (0, 0, 0))
        guardarInicio_texto = fuente_info.render(f'Punto de inicio:', True, (0, 0, 0))
        ventana.blit(guardarInicio_texto, (ANCHO_TABLERO + 10, 160))


        if start_pos != None:
            fila, columna = start_pos
            letra_columna = obtener_letra_columna(columna+1)
            posicion = str(fila+1) + str(letra_columna)
            guardarInicio_texto = fuente_info.render(posicion, True, (0, 0, 0))
            ventana.blit(guardarInicio_texto, (ANCHO_TABLERO + 150, 190))

        #guardarInicio_texto = fuente_info.render(f'¿Deseas seleccionar el punto final?', True, (0, 0, 0))
        guardarInicio_texto = fuente_info.render(f'Punto Final:', True, (0, 0, 0))
        ventana.blit(guardarInicio_texto, (ANCHO_TABLERO + 10, 240))

        if end_pos != None:
            fila, columna = end_pos
            letra_columna = obtener_letra_columna(columna+1)
            posicion = str(fila+1) + str(letra_columna)
            guardarInicio_texto = fuente_info.render(posicion, True, (0, 0, 0))
            ventana.blit(guardarInicio_texto, (ANCHO_TABLERO + 150, 270))

        Seleccionar_celda_texto = fuente_info.render(f'SELECCION DE CELDA:', True, (0, 0, 0))
        ventana.blit(Seleccionar_celda_texto, (ANCHO_TABLERO + 10, 310))
        #################################################################################################################
        #Para ver informacion de celda seleccionada

        # Obtener la posición del mouse
        pos_mouse = pygame.mouse.get_pos()
        x, y = pos_mouse
        print(x,y)
        # Calcular las coordenadas de la celda seleccionada
        fila_seleccionada = y // BLOCK_WIDTH
        columna_seleccionada = x // BLOCK_HEIGHT
        print(fila_seleccionada,columna_seleccionada)

        # Obtener el valor de la celda seleccionada del archivo de números
        if (1 <= fila_seleccionada < 16 and 1 <= columna_seleccionada < 16):
            # Dibujar un contorno alrededor de la celda seleccionada
            pygame.draw.rect(ventana, (255, 0, 0), (
                columna_seleccionada * BLOCK_WIDTH, fila_seleccionada * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT),
                             3)  # Contorno rojo de grosor 3
            # Obtener valores de celda seleccionada
            valor_celda = maze[fila_seleccionada-1][columna_seleccionada-1]

            # Obtener la letra de la columna y la descripción del color
            letra_columna = obtener_letra_columna(columna_seleccionada)
            #print(letra_columna, fila_seleccionada)

            info_texto = fuente_info.render(f'Coordenada: ({fila_seleccionada}, {letra_columna})', True, (0, 0, 0))
            ventana.blit(info_texto, (ANCHO_TABLERO + 10, 500))

            if valor_celda == 1:
                tipo = "pared"
            else:
                tipo = "camino"
            valor_texto = fuente_info.render(f'Valor: {tipo}', True, (0, 0, 0))
            ventana.blit(valor_texto, (ANCHO_TABLERO + 10, 540))


    # Actualizar la pantalla
    pygame.display.flip()

# Salir de Pygame
pygame.quit()
