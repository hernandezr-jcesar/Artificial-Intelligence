import numpy as np
import pandas as pd

PATH = "./dataset/IrisPlant_modificada.csv"

atypical_state = "Desconocido"
remaining_state = "Desconocido"



def add_dataset(path):
    try:
        # Leemos los datos desde el archivo CSV informando la dirección
        data = pd.read_csv(path)
        # Obtenemos los nombres de las columnas
        column_names = data.columns.tolist()
        # Se convierte el dataframe a numpy-array los datos a un NumPy array
        data_array = data.to_numpy()
        # separamos el dataset en las columnas de entrada y salida (x, y)
        input_data = data_array[:, :-1].astype(float)
        output_data = data_array[:, -1]

        return input_data, output_data, column_names
    except Exception as e:
        print(f"Error al cargar el conjunto de datos: {e}")
        return None, None, None
    
def add_data_point(input_vector, output_value):
        # Agrega un punto de datos al conjunto de entrenamiento
        input_data = np.vstack([input_data, input_vector])
        output_data = np.append(output_data, output_value)
    
def obtener_clases(x, y):
    ## Obtenemos la clase con base en sus etiquetas [y]
    clases_unicas = np.unique(y)

    ## Fragmentamos las clases en arrays diferentes 1 x clase
    subsets = [x[y == cls] for cls in clases_unicas]
    return clases_unicas, subsets


def detectar_valores_faltantes(datos):
    ## Se crea una máscara booleana que indica dónde hay valores nulos
    valores_faltantes = np.isnan(datos)
    ## Se obtienen las coordenadas (índices) donde hay valores NaN
    indices_faltantes = np.argwhere(valores_faltantes)
    ##
    print("Porcentaje de valores faltantes respecto a la muestra dada")
    print(f"{round((indices_faltantes.shape[0]/datos.shape[0])*100, 3)} %")
    state = "No existen valores faltantes"
    if indices_faltantes.shape[0] != 0:
        state = "Faltan datos"
        print("Coordenadas de valores faltantes")
        print(indices_faltantes) 
        print("\nNumero de valores faltantes")
        print(indices_faltantes.shape[0])
    return state

def calculos(datos, n):
    ## Convertimos el conjunto de datos a flotante por si no lo está
    datos.astype('float64')
    mean = 0.0
    std_mean = 0.0
    ## Se calculan el promedio y la desviación estandar por columna y posteriormente se devuelve
    mean = np.nansum(datos[:, n]) / len(datos)
    std_mean = np.nanstd(datos[:, n])
    return mean, std_mean

def detectar_valores_atipicos(datos, n):
    mean, std_mean = calculos(datos, n)
    lim_sup = mean + std_mean
    lim_inf = mean - std_mean

    print(f"Limite superior para el rasgo: {lim_sup}")
    print(f"Limite inferior para el rasgo: {lim_inf}")
    columna = datos[:, n]
    estado = "NO existen valores atípicos"

    for indice, valor in enumerate(columna):
        #z = (i - mean) / std_mean
        if valor < lim_inf or valor > lim_sup:
            estado = "Existen valores atípicos"
            print(f"Dato {valor} atípico detectado en el indice: [{indice}, {n})]")
        return estado


def normalizar_atipicos(datos, n):
    mean, std_mean = calculos(datos, n)
    lim_sup = mean + std_mean
    lim_inf = mean - std_mean
    ## Reemplazar los valores NaN con el promedio del rasgo con base en los calculos del promedio
    datos[:, n] = np.clip(datos[:, n], lim_inf, lim_sup)

    return datos

def imputacion_faltantes(datos, n):
    mean, y = calculos(datos, n)
    ## Reemplazar los valores NaN con el promedio del rasgo con base en los calculos del promedio
    datos[np.isnan(datos[:, n]), n] = mean

    return datos

def despliegue():
    print("Comprobar registros              (1)")
    print("Comprobar valores faltantes      (2)")
    print("Calculos por clase               (3)")
    print("Imputar datos faltantes          (4)")
    print("Comprobar valores atipicos       (5)")
    print("Normalizar valores atipicos      (6)")
    print("Modificar set de datos (x, y)    (7)")

if __name__ == "__main__":

    x, y, column_names = add_dataset(PATH)

    clases, subsets_por_clase = obtener_clases(x, y)

    menu = True
    while(menu):
        print(f"Estatus atipicos: {atypical_state}")
        print(f"Estatus faltantes: {remaining_state}")
        print("Acciones a realizar para el dataset\n")
        despliegue()
        opciones = int(input("Seleccione una opción > "))
        

        if(opciones == 1):
            ## Obtenemos las clases por dataset con
            if x is not None and y is not None and column_names is not None:
                print("\nNombres de las columnas:")
                print(column_names)
                for i, subset in zip(np.unique(y), subsets_por_clase):
                    print(f"Muestras: {subset.shape[0]}")
                    print(f"Rasgos: {subset.shape[1]}")
                    temp = subset.shape[0]/x.shape[0]
                    print(f"    Forma de la clase: {i} (Muestras: {subset.shape[0]} la cual representa {round(temp * 100, 3)}% del dataset)\n")
            else:
                print("No se pudo cargar el conjunto de datos.")

        elif(opciones == 2):
            for i, subset in zip(np.unique(y), subsets_por_clase):
                print(f"\n###### Comprobar faltantes  CLASE: {i} ######")
                remaining_state = detectar_valores_faltantes(subset)
                

        elif(opciones == 3):
            for i, subset in zip(np.unique(y), subsets_por_clase):
                print(f"\n###### Calculos  CLASE: {i} ######")
                print(f"Resultados para el promedio y desviacion\n")
                for k in range(0, len(column_names)-1):
                    print(f"Rasgo({column_names[k]}) -> promedio: {calculos(subset, k)[0]}, desviación estandar: {calculos(subset, k)[1]}")

        elif(opciones == 4 and remaining_state != "OK!"):
            
            if (remaining_state == "Desconocido"):
                print("NO se ha comprobado el estado de datos faltantes")
            elif(remaining_state == "No existen valores faltantes"):
                print("NO es necesario hacer una imputacion de datos")
            else:
                for i, subset in zip(np.unique(y), subsets_por_clase):
                    print(f"Inserción de datos en la clase {i}")
                    for k in range(0, len(column_names)-1):
                        subset = imputacion_faltantes(subset, k)
                remaining_state = "OK!"

        elif(opciones == 5):
            if (atypical_state != "OK!"):
                for i, subset in zip(np.unique(y), subsets_por_clase):
                    print(f"\n###### Comprobar atipicos  CLASE: {i} ######")
                    for k in range(0, len(column_names)-1):
                        print(f"Detectando valores atipicos dentro de los rasgos: {column_names[k]}")
                        atypical_state = detectar_valores_atipicos(subset, k)
            print("Los datos atipicos están o fueron normalizados")

        elif(opciones == 6):
            if (remaining_state == "Desconocido"):
                print("NO se ha comprobado el estado de datos faltantes")
            for i, subset in zip(np.unique(y), subsets_por_clase):
                    print(f"Inserción de datos en la clase {i}")
                    for k in range(0, len(column_names)-1):
                        subset = normalizar_atipicos(subset, k)
            atypical_state = "OK!"
            
        elif(opciones == 7):
            
            new_x = np.concatenate(subsets_por_clase)
            new_y = y
            
            print(new_x.shape)

            print(f"Tenemos la siguiente imagen el vector de entrada {new_x.shape[1]}")
            print(f"Para habilitar 1, deshabilitar 0")
            input_size = (input("Habilite los rasgos (columas) que desea tener en su vector de entrada. Ej. 0, 0, 1, 0: \n"))

            # Solicita al usuario los tamaños de salida
            print(f"Tenemos la siguiente imagen el vector de salida {new_x.shape[0]}")
            print(f"Use la notacion de segmentación, Ej. [X:N], [:N], [X:], , etc")
            output_size = (input("Habilite los rasgos (filas) que desea tener en su vector de salida.\n"))
            
            habilitar = input_size.split(',')
            habilitar = [int(splits) for splits in habilitar]
            counter = 0
            to_delete = []
            for split in habilitar:
                if split == 0:
                    to_delete.append(counter)
                counter = counter + 1
            print(to_delete)

            new_xMod = np.delete(new_x, to_delete, axis=1)
            print(new_xMod)
                
   
            filas_seleccionadas = output_size.split(":")
            filas_seleccionadas = [int(splits) for splits in filas_seleccionadas]

            output_dataMod = y[filas_seleccionadas[0]: filas_seleccionadas[1]+1]
            new_xMod= new_xMod[filas_seleccionadas[0]: filas_seleccionadas[1]+1]
            print(new_xMod)

            print("Conjunto de datos creado: \n")
            print(np.column_stack((new_xMod, output_dataMod)))

        
        else:
            menu = False
            

            
            

                 
            

 
        
    


    

    
    
    




    