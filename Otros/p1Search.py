import pygame


import sys
import numpy as np


def itsAvailable(Px, Py, field):
    if (0 <= Px < field.shape[1]) & (0 <= Py < field.shape[0]):
        return  True
    return False

def itsRoad(Px, Py, field):
    if(itsAvailable(Px, Py, field)):
        noRoad = 0
        if field[Py, Px] != noRoad:
            return True
    return False

def mostrarValorCelda(x1, y1, field):
    terrenos = ["Montaña", "Camino", "Agua", "Arena", "Bosque", "" , "", "", "", "Limite"]
    valor = field[y1][x1]
    pygame.display.set_caption(f"Coordenada ({x1}, {y1}) | Valor: {valor} | Terreno: {terrenos[valor]}" )


def coloresTipo(i, j, field):
    # Asignación de colores a las celdas de acuerdo a su valor inicial
    if field[i, j] == 0:
        color = "#6F6A69"
        tipo = "Montaña"
    if field[i, j] == 1:
        color = "#834625"
        tipo = "Tierra"
    if field[i, j] == 2:
        color = "#3B9FD0"
        tipo = "Agua"
    if field[i, j] == 3:
        color = "#E5E5C5"
        tipo = "Arena"
    if field[i, j] == 4:
        color = "#37B33B"
        tipo = "Bosque"
    if field[i, j] == 8:
        color = "#8338ec"
        tipo = "Visitado"
    if field[i, j] == 9:
        color = "#9B9B9B"
        tipo = "Limite"
    return color, tipo

def personajes(seleccion, field, i, j):
    costo = 0
    # Costes personaje: humano
    if (seleccion == "humano"):
        if (field[i, j] == 0):
            costo = 99
        if (field[i, j] == 1):
            costo = 1
        if (field[i, j] == 2):
            costo = 2
        if (field[i, j] == 3):
            costo = 3
        if (field[i, j] == 4):
            costo = 4
        if (field[i, j] == 5):
            costo = 5
        if (field[i, j] == 6):
            costo = 5
    # Costes personaje: mono
    elif (seleccion == "mono"):
        if (field[i, j] == 0):
            costo = 99
        if (field[i, j] == 1):
            costo = 2
        if (field[i, j] == 2):
            costo = 4
        if (field[i, j] == 3):
            costo = 3
        if (field[i, j] == 4):
            costo = 1
        if (field[i, j] == 5):
            costo = 5
        if (field[i, j] == 6):
            costo = 99
    # Costes personaje: pulpo
    elif (seleccion == "pulpo"):
        if (field[i, j] == 0):
            costo = 99
        if (field[i, j] == 1):
            costo = 2
        if (field[i, j] == 2):
            costo = 1
        if (field[i, j] == 3):
            costo = 99
        if (field[i, j] == 4):
            costo = 3
        if (field[i, j] == 5):
            costo = 2
        if (field[i, j] == 6):
            costo = 99
    # Costes personaje: pulpo
    elif (seleccion == "pie grande"):
        if (field[i, j] == 0):
            costo = 15
        if (field[i, j] == 1):
            costo = 4
        if (field[i, j] == 2):
            costo = 99
        if (field[i, j] == 3):
            costo = 99
        if (field[i, j] == 4):
            costo = 4
        if (field[i, j] == 5):
            costo = 5
        if (field[i, j] == 6):
            costo = 3

    return costo
              

def mapaUNO():
    map = np.loadtxt('./map.txt', dtype = int)

    # Configuración de la ventana y el laberinto
    WIDTH, HEIGHT = map.shape[1] * 50, map.shape[0] * 50
    CELL_SIZE = 50  # Tamaño de una celda

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Laberinto (Controlado por usuario)")
    clock = pygame.time.Clock()
    # Posición del jugador
    posX, posY = 2, 2  # Inicio del jugador

    font = pygame.font.Font(None, 36)

    posXcopy, posYcopy = 2, 2  # Inicio del jugador
    map[posY, posX] = 8
    finalX, finalY = 9, 9
    FPS = 1
    visitado = set()
    contador = 0
    # Bucle principal del juego
    running = itsRoad(posX, posY, map) & itsRoad(finalX, finalY, map)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse = pygame.mouse.get_pos()
                x, y = mouse[0] // 50, mouse[1] // 50
                if 0 <= x < len(map[0]) and 0 <= y < len(map):
                    mostrarValorCelda(x, y, map)
            elif event.type == pygame.KEYDOWN:
                '''
                Reiniciamos entorno en caso: Jugador atrapado
                '''
                if event.key == pygame.K_r:
                    map = np.loadtxt('./map.txt', dtype = int)
                    posX, posY = int(posXcopy), int(posYcopy)
                    map[posY, posX] = 8
                    contador = 0
                if event.key == pygame.K_ESCAPE:
                    running = False
                    

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] and map[posY][posX - FPS] == 1:
            map[posY][posX - FPS] = 8 
            contador = contador + 1
            posX -= FPS
        if keys[pygame.K_d] and map[posY][posX + FPS] == 1:
            map[posY][posX + FPS] = 8
            contador = contador + 1
            posX += FPS
        if keys[pygame.K_w] and map[posY - FPS][posX] == 1:
            map[posY - FPS][posX] = 8
            contador = contador + 1
            posY -= FPS
        if keys[pygame.K_s] and map[posY + FPS][posX] == 1:
            map[posY + FPS][posX] = 8
            contador = contador + 1
            posY += FPS

        clock.tick(10)
        screen.fill("#0A0A0A")
        # Dibujar el laberinto
        for y in range(max(0, posY - 1), min(len(map), posY + 1 + 1)):
            for x in range(max(0, posX - 1), min(len(map[y]), posX + 1 + 1)):
                    if map[y][x] == 1:
                        pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    if map[y][x] == 9:
                        pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for celda in visitado:
            x, y = celda
            pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))      

        msj = "X"

        if(posX == finalX & posY == finalY):
            pygame.display.set_caption(f"HAS LLEGADO A LA META!" )
            msj = f"{str(contador)}"

        visitado.add((posX, posY))  

        # Dibujar al jugador
        pygame.draw.rect(screen, "#2a9d8f", (posX * CELL_SIZE, posY * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        text = font.render(f"{str(contador)}" , True, "#0A0A0A")
        text_rect = text.get_rect(center=(posX * CELL_SIZE + CELL_SIZE // 2, posY * CELL_SIZE + CELL_SIZE // 2))
        screen.blit(text, text_rect)


        pygame.draw.rect(screen, "#780000", (finalX * CELL_SIZE, finalY * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        text = font.render(msj , True, "#0A0A0A")
        text_rect = text.get_rect(center=(finalX * CELL_SIZE + CELL_SIZE // 2, finalY * CELL_SIZE + CELL_SIZE // 2))
        screen.blit(text, text_rect)

        pygame.display.flip()


def mapaDOS(pj):
    map = np.loadtxt('./map2.txt', dtype = int)

    Mensaje = 0
    # Configuración de la ventana y el laberinto
    WIDTH, HEIGHT = map.shape[1] * 50, map.shape[0] * 50
    CELL_SIZE = 50  # Tamaño de una celda

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Mundo abierto (Controlado por usuario) - {pj}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    # Posición del jugador
    posX, posY = 2, 2  # Inicio del jugador
    finalX, finalY = 9, 9
    contador = 0
    FPS = 1
    visitado = set()
    # Bucle principal del juego

    running = itsRoad(posX, posY, map) & itsRoad(finalX, finalY, map)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse = pygame.mouse.get_pos()
                x, y = mouse[0] // 50, mouse[1] // 50
                if 0 <= x < len(map[0]) and 0 <= y < len(map):
                    mostrarValorCelda(x, y, map)

        #print((personajes(pj, map ,posY, posX - FPS)))
        '''
        Podría encapsular estos ifs para darle caracteristicas a cada personaje

        con contador implementar un sistema de costos para cada personaje :p
        '''
        keys = pygame.key.get_pressed()
        if (keys[pygame.K_a] and map[posY][posX - FPS] != 9) and (personajes(pj, map ,posY, posX - FPS) != 99):
            posX -= FPS
            contador = contador + personajes(pj, map ,posY, posX)

        if (keys[pygame.K_d] and map[posY][posX + FPS] != 9) and (personajes(pj, map ,posY, posX + FPS) != 99):
            posX += FPS
            contador = contador + personajes(pj, map ,posY, posX)

        if (keys[pygame.K_w] and map[posY - FPS][posX] != 9) and (personajes(pj, map ,posY - FPS, posX) != 99):
            posY -= FPS
            contador = contador + personajes(pj, map ,posY, posX)

        if (keys[pygame.K_s] and map[posY + FPS][posX] != 9) and (personajes(pj, map ,posY + FPS, posX) != 99):
            posY += FPS
            contador = contador + personajes(pj, map, posY, posX)

        clock.tick(10)
        screen.fill("#0A0A0A")

        # Dibujar el laberinto
        for y in range(max(0, posY - 1), min(len(map), posY + 1 + 1)):
            for x in range(max(0, posX - 1), min(len(map[y]), posX + 1 + 1)):
                    if map[y][x] == 0:
                        pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    if map[y][x] == 1:
                        pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    if map[y][x] == 2:
                        pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    if map[y][x] == 3:
                        pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    if map[y][x] == 4:
                        pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))    
                    if map[y][x] == 9:
                        pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for celda in visitado:
            x, y = celda
            pygame.draw.rect(screen, coloresTipo(y, x, map)[0], (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))      

        msj = "X"
        if(posX == finalX & posY == finalY):
            pygame.display.set_caption(f"HAS LLEGADO A LA META!" )
            msj = str(contador)
        visitado.add((posX, posY))  

        pygame.draw.rect(screen, "#2a9d8f", (posX * CELL_SIZE, posY * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        text = font.render(f"{str(contador)}" , True, "#0A0A0A")
        text_rect = text.get_rect(center=(posX * CELL_SIZE + CELL_SIZE // 2, posY * CELL_SIZE + CELL_SIZE // 2))
        screen.blit(text, text_rect)


        pygame.draw.rect(screen, "#780000", (finalX * CELL_SIZE, finalY * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        text = font.render(msj , True, "#0A0A0A")
        text_rect = text.get_rect(center=(finalX * CELL_SIZE + CELL_SIZE // 2, finalY * CELL_SIZE + CELL_SIZE // 2))
        screen.blit(text, text_rect)


        pygame.display.flip()




def main():
    pygame.init()

    # Configuración de la ventana del menú
    MENU_WIDTH, MENU_HEIGHT = 250, 150
    menu_screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
    pygame.display.set_caption("Menú")

    # Fuentes
    font = pygame.font.Font(None, 36)

    # Opciones del menú
    menu_options = ["LABERINTO", "MUNDO ABIERTO"]
    selected_option = 0  # Índice de la opción seleccionada

    # Bucle principal del menú
    menu_running = True
    while menu_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                menu_running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(menu_options)
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(menu_options)
                elif event.key == pygame.K_RETURN:
                    selected_mode = menu_options[selected_option]
                    menu_running = False

        menu_screen.fill("#2a9d8f")
        for i, option in enumerate(menu_options):
            color = "#780000" if i == selected_option else "#edf2f4"
            text = font.render(option, True, color)
            text_rect = text.get_rect(center=(MENU_WIDTH // 2, MENU_HEIGHT // 2 + i * 40))
            menu_screen.blit(text, text_rect)
        pygame.display.flip()

    # Iniciar la función o modo de juego seleccionado
    if selected_mode == "LABERINTO":
        # Selecion modo laberinto
        mapaUNO()
    elif selected_mode == "MUNDO ABIERTO":
        pygame.init()

        selWIDTH, selHEIGHT = 250, 250
        menu_screen = pygame.display.set_mode((selWIDTH, selHEIGHT))
        pygame.display.set_caption("Menú")

        font = pygame.font.Font(None, 36)

        pjs = ["humano", "mono", "pulpo", "pie grande"]
        opcion = 0

        menu_running = True
        while menu_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    menu_running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        opcion = (opcion - 1) % len(pjs)
                    elif event.key == pygame.K_DOWN:
                        opcion = (opcion + 1) % len(pjs)
                    elif event.key == pygame.K_RETURN:
                        selected_mode = pjs[opcion]
                        menu_running = False

            menu_screen.fill("#2a9d8f")
            for i, option in enumerate(pjs):
                color = "#780000" if i == opcion else "#edf2f4"
                text = font.render(option, True, color)
                text_rect = text.get_rect(center=(selWIDTH // 2, selHEIGHT // 3 + i * 40))
                menu_screen.blit(text, text_rect)
            pygame.display.flip()
        print(type(pjs[opcion]))
        mapaDOS(pjs[opcion])

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()