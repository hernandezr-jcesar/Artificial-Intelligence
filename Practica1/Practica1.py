# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:10:59 2023

@author: David
"""

import numpy as np

def convertir_a_numerico(dato):
    try:
        return int(dato) if "." not in dato else float(dato)
    except ValueError:
        return dato

def obtener_nombre_tipo(tipo):
    return str(tipo).split("'")[1]

def definir_tipos_de_datos(data):
    for columna in range(len(data[0])):
        # Convertir la columna a tipos numéricos si es posible
        data_columna = [convertir_a_numerico(row[columna]) for row in data]

        # Obtener el tipo de dato de la columna
        tipo_de_dato = type(data_columna[0])

        # Obtener el nombre del tipo sin "<class '...'>"
        nombre_tipo = obtener_nombre_tipo(tipo_de_dato)

        # Determinar la medida de la columna
        if np.issubdtype(tipo_de_dato, np.number):
            if all(isinstance(dato, int) for dato in data_columna):
                medida = 'discreta'
            else:
                medida = 'continua'
        elif tipo_de_dato == str:
            medida = 'nominal'  # Determinar si es string
        else:
            medida = 'discreta'

        # Imprimir el dato a analizar, el tipo de dato y su medida
        print(f"Columna {columna + 1}: {nombre_tipo} ({medida})")

    return data



class SistemaCargaDatos:
    def __init__(self, archivo, delimitador=','):
        self.archivo = archivo
        self.delimitador = delimitador
        self.data = None
        self.num_atributos = None
        self.num_patrones = None

    def cargar_datos(self, filas_a_seleccionar):
        try:
            # Cargar datos desde el archivo CSV
            with open(self.archivo, 'r') as file:
                lines = file.readlines()
                self.data = [line.strip().split(self.delimitador) for line in lines]

            # Definir automáticamente los tipos de dato y medida
            self.data = definir_tipos_de_datos(self.data)

            # Obtener el número de atributos y patrones
            self.num_atributos = len(self.data[0])
            self.num_patrones = len(self.data)

            print(f"\nNúmero de atributos: {self.num_atributos}\nNúmero de patrones: {self.num_patrones}")

            # Obtener los datos como una matriz
            matriz_datos = np.array(self.data)
            print("\nMatriz de datos original:")
            print(matriz_datos)

            # Seleccionar un subconjunto de renglones
            subconjunto_datos = [self.data[i] for i in filas_a_seleccionar]
            print(f"\nSubconjunto de datos seleccionando filas {filas_a_seleccionar}:")
            for row in subconjunto_datos:
                print(row)


        except FileNotFoundError:
            print(f"Error: El archivo {self.archivo} no fue encontrado.")

        except Exception as e:
            print(f"Error: {e}")

    def guardar_subconjunto(self, subconjunto_datos, nuevo_archivo):
        try:
            with open(nuevo_archivo, 'w') as file:
                for row in subconjunto_datos:
                    file.write(self.delimitador.join(map(str, row)) + '\n')

            print(f"Subconjunto de datos guardado exitosamente en {nuevo_archivo}")

        except Exception as e:
            print(f"Error al guardar el subconjunto de datos en {nuevo_archivo}: {e}")

    def seleccionar_atributos(self, columnas_seleccionadas, archivo_guardar=None):
        try:
            # Seleccionar un subconjunto de atributos
            subconjunto_atributos = [[row[i] for i in columnas_seleccionadas] for row in self.data]
    
            print(f"\nSubconjunto de atributos seleccionados:")
            for row in subconjunto_atributos:
                print(row)
    
            if archivo_guardar:
                self.guardar_subconjunto(subconjunto_atributos, archivo_guardar)
                #print(f"Subconjunto de atributos guardado en {archivo_guardar}")
    
            return subconjunto_atributos
    
        except Exception as e:
            print(f"Error al seleccionar subconjunto de atributos: {e}")
            return None



# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia del sistema de carga de datos
    sistema = SistemaCargaDatos(archivo="glass.data", delimitador=",")

    filas_seleccionadas = [1, 2, 5]
    # Cargar datos y mostrar información
    sistema.cargar_datos(filas_seleccionadas)

    # Guardar el subconjunto de datos en un nuevo archivo CSV
    nuevo_archivo_csv = "subconjunto_datos.csv"
    sistema.guardar_subconjunto(subconjunto_datos=[sistema.data[i] for i in filas_seleccionadas], nuevo_archivo=nuevo_archivo_csv)

    subconjunto_atributo = "subconjunto_atributo.csv"
    columnas_seleccionadas = [1, 2]
    subconjunto_atributos = sistema.seleccionar_atributos(columnas_seleccionadas, archivo_guardar=subconjunto_atributo)

