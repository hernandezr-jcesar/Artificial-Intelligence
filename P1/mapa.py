import pygame

# Inicializar Pygame
pygame.init()

# Configurar dimensiones del tablero y celdas
ANCHO_CELDA = 40
ANCHO_TABLERO = 16 * ANCHO_CELDA
ALTO_TABLERO = 16 * ANCHO_CELDA

# Definir la posición donde quieres imprimir el array
posicion_x = 40
posicion_y = 40

# Configurar colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

GRAY = (192, 192, 192)
DARK_GRAY = (150, 150, 150)
PEACH = (255, 218, 185)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PURPLE_BLACK = (194, 55, 219)

# Configurar colores
COLORES = {
   # '0': DARK_GRAY,
   # '1': BLANCO,

    '0': DARK_GRAY,
    '1': PEACH,
    '2': BLUE,
    '3': YELLOW,
    '4': GREEN
   # Añadir más colores según sea necesario
}


# Leer números desde el archivo
with open('mapa.txt', 'r') as archivo:
    numeros = [linea.split() for linea in archivo]

# Obtener el número de filas y columnas del laberinto
NUM_ROWS = len(numeros)
NUM_COLS = len(numeros[0])

# Crear una fuente para el texto
fuente_info = pygame.font.Font(None, 24)

# Inicializar la ventana
ventana = pygame.display.set_mode((ANCHO_TABLERO + 200, ALTO_TABLERO))
pygame.display.set_caption('Tablero 15x15')

# Función para obtener la letra de la columna a partir de su posición numérica
def obtener_letra_columna(columna):
    return chr(columna - 1 + ord('A'))

start_pos=None
end_pos = None
done = False
# Loop principal del juego
ejecutando = True
while ejecutando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False
    if evento.type == pygame.MOUSEBUTTONDOWN:
        # Obtener la posición del clic del mouse
        pos = pygame.mouse.get_pos()
        # Convertir la posición en índices de fila y columna
        row = (pos[1] // ANCHO_CELDA) - 1
        col = (pos[0] // ANCHO_CELDA) - 1
        if start_pos is None:
            start_pos = (row, col)
        elif end_pos is None and (row, col) != start_pos:
            end_pos = (row, col)
            done = True

    # Limpiar pantalla
    ventana.fill(GRAY)

    # Dibujar letras (columnas)
    letra = ord('A')
    for i in range(1,16):
        pygame.draw.rect(ventana, NEGRO, (i * ANCHO_CELDA, 0, ANCHO_CELDA, ANCHO_CELDA), 1)
        fuente = pygame.font.Font(None, 36)
        texto = fuente.render(chr(letra), True, NEGRO)
        ventana.blit(texto, (i * ANCHO_CELDA + 15, 5))
        letra += 1

    # Dibujar filas (números)
    for i in range(16):
        if i != 0:
            pygame.draw.rect(ventana, NEGRO, (0, i * ANCHO_CELDA, ANCHO_CELDA, ANCHO_CELDA),1)
            fuente = pygame.font.Font(None, 36)
            texto = fuente.render(str(i), True, NEGRO)
            ventana.blit(texto, (5, i * ANCHO_CELDA + 10))


    # Dibujar celdas con colores desde el archivo
    for fila in range(NUM_ROWS):
        for columna in range(NUM_COLS):
            x = posicion_x + (columna * ANCHO_CELDA)  # Ajusta el tamaño del elemento del array según sea necesario
            y = posicion_y + (fila * ANCHO_CELDA)
            color = COLORES.get(numeros[fila][columna],BLANCO)  # Blanco por defecto si no hay coincidencia
            pygame.draw.rect(ventana, NEGRO, (x, y, ANCHO_CELDA, ANCHO_CELDA),1)  # Borde negro
            pygame.draw.rect(ventana, color, (x + 1, y + 1, ANCHO_CELDA-2, ANCHO_CELDA - 2))



    # Obtener la posición del mouse
    pos_mouse = pygame.mouse.get_pos()
    mouse_x, mouse_y = pos_mouse

    # Calcular las coordenadas de la celda seleccionada
    fila_seleccionada = mouse_y // ANCHO_CELDA
    columna_seleccionada = mouse_x // ANCHO_CELDA

    # Obtener el valor de la celda seleccionada del archivo de números
    if 1 <= fila_seleccionada < 16 and 1 <= columna_seleccionada < 16:
        # Dibujar un contorno alrededor de la celda seleccionada
        pygame.draw.rect(ventana, (255, 0, 0), (
            columna_seleccionada * ANCHO_CELDA, fila_seleccionada *ANCHO_CELDA, ANCHO_CELDA, ANCHO_CELDA),3)  # Contorno rojo de grosor 3

        # Obtener valores de celda seleccionada
        valor_celda = numeros[fila_seleccionada - 1][columna_seleccionada - 1]

        # Obtener la letra de la columna y la descripción del color
        letra_columna = obtener_letra_columna(columna_seleccionada)
        descripcion_color = COLORES.get(valor_celda,'Desconocido')  # Valor por defecto si el color no está en el diccionario

        if valor_celda is not None:
            # Mostrar la información de la celda seleccionada al lado de la ventana
            info_texto = fuente_info.render(f'Coordenada: ({fila_seleccionada}, {letra_columna})', True, (0, 0, 0))
            ventana.blit(info_texto, (ANCHO_TABLERO + 10, 20))

            valor_texto = fuente_info.render(f'Valor: {valor_celda}', True, (0, 0, 0))
            ventana.blit(valor_texto, (ANCHO_TABLERO + 10, 50))

            color_texto = fuente_info.render(f'Color: {descripcion_color}', True, (0, 0, 0))
            ventana.blit(color_texto, (ANCHO_TABLERO + 10, 80))

            col_des = {
                '0': "MONTAÑA",
                '1': "TIERRA",
                '2': "AGUA",
                '3': "ARENA",
                '4': "BOSQUE"
            }
            descripcion = col_des.get(valor_celda)

            descripcion_texto = fuente_info.render(f'Descripcion: {descripcion}', True, (0, 0, 0))
            ventana.blit(descripcion_texto, (ANCHO_TABLERO + 10, 110))
        else:
            vacio = fuente_info.render(f'', True, (0, 0, 0))
            ventana.blit(vacio, (ANCHO_TABLERO + 10, 50))

    # Actualizar pantalla
    pygame.display.flip()

# Salir del juego
pygame.quit()
