import numpy as np
import pandas as pd

# Describir cada uno de los atributos al momento
def punto1(dataset_path):  

    # Cargar el conjunto de datos usando pandas, indicando que la primera columna es el índice
    df = pd.read_csv(dataset_path)

    # Obtener los atributos y la etiqueta
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
  
    # Descripción de los atributos del vector de entrada X
    for idx, column in enumerate(X.columns):
        print(f"\nAtributo {idx + 1}: {column}")
        print(f"Tipo de dato: {X[column].dtype}")

        if X[column].dtype == 'object':
            # Si es categórico
            print(f"Categorías: {X[column].unique()}")
        else:
            # Si es numérico
            print(f"Mínimo: {X[column].min()}")
            print(f"Máximo: {X[column].max()}")
            print(f"Promedio: {X[column].mean():.2f}")            
            print(f"Desviación Estándar: {X[column].std():.2f}")

    # Descripción de la etiqueta de salida Y
    print(f"\nEtiqueta Y:")
    print(f"Tipo de dato: {y.dtype}")
    print(f"Categorías: {y.unique()}")
    
    return df, X, y

# Definir los atributos del vector de entrada X y de salida(clase) Y
def punto2(df):      

    # Obtener los atributos y la etiqueta
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Por cada clase en Y
    clases = y.unique()
    for clase in clases:
        print(f"\nEstadísticas para la Clase {clase}:")
        # Filtrar las instancias correspondientes a la clase actual
        instancias_clase = df[df.iloc[:, -1] == clase].iloc[:, :-1]

        # Obtener las estadísticas
        for idx, column in enumerate(instancias_clase.columns):
            print(f"\nAtributo {idx + 1}: {column}")
            print(f"Máximo: {instancias_clase[column].max()}")
            print(f"Mínimo: {instancias_clase[column].min()}")
            print(f"Promedio: {instancias_clase[column].mean():.2f}")
            print(f"Desviación Estándar: {instancias_clase[column].std():.2f}")
            
            if instancias_clase[column].dtype == 'object':
                # Si es categórico
                print(f"Categorías: {instancias_clase[column].unique()}")

    # Descripción de la etiqueta de salida Y
    print(f"\nEtiqueta Y:")
    print(f"Tipo de dato: {y.dtype}")
    print(f"Categorías: {y.unique()}")

# En caso de ser necesario hacer un preprocesamiento a la base de datos, describirlo
def punto3(df):           
    # Manejo de datos faltantes
    df_imputed = df.copy()
    for column in df_imputed.columns:
        if df_imputed[column].dtype == np.float64:
            # Calcular la media de la columna
            mean_value = df_imputed[column].mean()
            # Reemplazar los valores faltantes con la media
            df_imputed[column] = df_imputed[column].fillna(mean_value)

    # Codificación de variables categóricas (si es necesario)
    df_encoded = df_imputed.copy()
    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'object':
            # Mapear valores únicos a números
            unique_values = df_encoded[column].unique()
            mapping = {value: index for index, value in enumerate(unique_values)}
            df_encoded[column] = df_encoded[column].map(mapping)

    # Escalado de características (si es necesario)
    df_scaled = df_encoded.copy()
    for column in df_scaled.columns[:-1]:  # Excluyendo la columna de etiquetas
        # Calcular la media y la desviación estándar de la columna
        mean_value = df_scaled[column].mean()
        std_dev = df_scaled[column].std()
        # Estandarizar la columna
        df_scaled[column] = (df_scaled[column] - mean_value) / std_dev

    # Dividir los datos en conjuntos de entrenamiento y prueba
    np.random.seed(42)  # Establecer la semilla para reproducibilidad
    mask = np.random.rand(len(df_scaled)) < 0.8
    train_data = df_scaled[mask]
    test_data = df_scaled[~mask]

    X_train = train_data.iloc[:, :-1].to_numpy()
    y_train = train_data.iloc[:, -1].to_numpy()
    X_test = test_data.iloc[:, :-1].to_numpy()
    y_test = test_data.iloc[:, -1].to_numpy()    

    return X_train, X_test, y_train, y_test

def euclidean_distance(x1, x2):
    # Calcular la distancia euclidiana entre dos puntos
    return np.linalg.norm(x1 - x2)

def manhattan_distance(x1, x2):
    # Calcular la distancia de Manhattan entre dos puntos
    return np.sum(np.abs(x1 - x2))

# def min_distance_classifier(train_X, train_y, test_X, distance_func=euclidean_distance):
#     # Implementación del clasificador de mínima distancia
#     predictions = []
#     for sample in test_X:
#         # Calcular la distancia con cada instancia de entrenamiento
#         distances = [distance_func(sample, train_sample) for train_sample in train_X]
#         # Obtener la clase de la instancia más cercana
#         predicted_class = train_y[np.argmin(distances)]
#         predictions.append(predicted_class)
#     return np.array(predictions)

def min_distance_classifier(train_X, train_y, test_X, distance_func=euclidean_distance):
    # Calcula el valor promedio para cada atributo en el conjunto de entrenamiento para cada clase
    class_averages = {cls: np.mean(train_X[train_y == cls], axis=0) for cls in set(train_y)}

    # Clasificación
    predictions = []
    for sample in test_X:
        # Calcula la distancia con los valores promedio de cada clase
        distances = [distance_func(sample, class_averages[cls]) for cls in class_averages]
        # Asigna la clase cuyo valor promedio está más cerca
        predicted_class = list(class_averages.keys())[np.argmin(distances)]
        predictions.append(predicted_class)

    return np.array(predictions)

def knn_classifier(train_X, train_y, test_X, k, distance_func=euclidean_distance):
    # Implementación del clasificador KNN
    predictions = []
    for sample in test_X:
        # Calcular la distancia con cada instancia de entrenamiento
        distances = [distance_func(sample, train_sample) for train_sample in train_X]
        # Obtener las k instancias más cercanas
        nearest_neighbors_indices = np.argsort(distances)[:k]
        # Obtener las clases de las k instancias más cercanas
        nearest_neighbors_classes = train_y[nearest_neighbors_indices]
        # Obtener la clase más común entre las k instancias más cercanas
        predicted_class = np.bincount(nearest_neighbors_classes).argmax()
        predictions.append(predicted_class)
    return np.array(predictions)

def calculate_accuracy(y_true, y_pred):
    # Calcular la precisión del clasificador
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy

def calculate_error(y_true, y_pred):
    # Calcular la tasa de error del clasificador
    incorrect_predictions = np.sum(y_true != y_pred)
    total_samples = len(y_true)
    error_rate = incorrect_predictions / total_samples
    return error_rate

# Punto 4: Clasificador de Mínima Distancia
def punto4(X_train, X_test, y_train, y_test, distance_func=euclidean_distance):
     # a. Entrenamiento y Prueba
    predictions_train_test = min_distance_classifier(X_train, y_train, X_test, distance_func)
    accuracy_train_test = calculate_accuracy(y_test, predictions_train_test)
    error_train_test = calculate_error(y_test, predictions_train_test)
    
    # b. K-fold Cross Validation (por ejemplo, con K=5)
    k_fold = 30
    fold_size = len(X_train) // k_fold
    accuracies_cv = []
    errors_cv = []
    
    for i in range(k_fold):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
    
        # Conjunto de prueba actual
        cv_X_test = X_train[start_idx:end_idx]
        cv_y_test = y_train[start_idx:end_idx]
    
        # Conjunto de entrenamiento actual
        cv_X_train = np.concatenate([X_train[:start_idx], X_train[end_idx:]])
        cv_y_train = np.concatenate([y_train[:start_idx], y_train[end_idx:]])
    
        # Clasificación
        predictions_cv = min_distance_classifier(cv_X_train, cv_y_train, cv_X_test, distance_func)
        
        # Métricas
        accuracy_cv = calculate_accuracy(cv_y_test, predictions_cv)
        error_cv = calculate_error(cv_y_test, predictions_cv)
    
        accuracies_cv.append(accuracy_cv)
        errors_cv.append(error_cv)
    
    # c. Bootstrap
    num_bootstrap_samples = 100
    accuracies_bootstrap = []
    errors_bootstrap = []
    
    for _ in range(num_bootstrap_samples):
        # Muestreo bootstrap
        bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
        bootstrap_X_train = X_train[bootstrap_indices]
        bootstrap_y_train = y_train[bootstrap_indices]
    
        # Clasificación
        predictions_bootstrap = min_distance_classifier(bootstrap_X_train, bootstrap_y_train, X_test, distance_func)
        
        # Métricas
        accuracy_bootstrap = calculate_accuracy(y_test, predictions_bootstrap)
        error_bootstrap = calculate_error(y_test, predictions_bootstrap)
    
        accuracies_bootstrap.append(accuracy_bootstrap)
        errors_bootstrap.append(error_bootstrap)
    
    # Resultados para el clasificador de mínima distancia
    print("\n   Resultados Clasificador de Mínima Distancia:")
    print(f"    Métrica utilizada: {distance_func.__name__}")
    print("\n   a. Entrenamiento y Prueba:")
    print(f"        Precisión: {accuracy_train_test:.2%}")
    print(f"        Tasa de Error: {error_train_test:.2%}")
    
    print("\n   b. K-fold Cross Validation:")
    print(f"        Precisión Promedio: {np.mean(accuracies_cv):.2%}")
    print(f"        Tasa de Error Promedio: {np.mean(errors_cv):.2%}")
    
    print("\n   c. Bootstrap:")
    print(f"        Precisión Promedio: {np.mean(accuracies_bootstrap):.2%}")
    print(f"        Tasa de Error Promedio: {np.mean(errors_bootstrap):.2%}")

# Punto 5: Clasificador KNN
def punto5(X_train, X_test, y_train, y_test,distance_func=euclidean_distance):
    
    # a. Entrenamiento y Prueba
    
    k_fold = 20
    fold_size = len(X_train) // k_fold
    
    k_value = 10  # Puedes ajustar este valor según tu elección
    predictions_knn_train_test = knn_classifier(X_train, y_train, X_test, k_value, distance_func)
    accuracy_knn_train_test = calculate_accuracy(y_test, predictions_knn_train_test)
    error_knn_train_test = calculate_error(y_test, predictions_knn_train_test)
    
    # b. K-fold Cross Validation (usando el mismo valor de k)
    accuracies_knn_cv = []
    errors_knn_cv = []
    
    for i in range(k_fold):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
    
        # Conjunto de prueba actual
        cv_X_test = X_train[start_idx:end_idx]
        cv_y_test = y_train[start_idx:end_idx]
    
        # Conjunto de entrenamiento actual
        cv_X_train = np.concatenate([X_train[:start_idx], X_train[end_idx:]])
        cv_y_train = np.concatenate([y_train[:start_idx], y_train[end_idx:]])
    
        # Clasificación
        predictions_knn_cv = knn_classifier(cv_X_train, cv_y_train, cv_X_test, k_value, distance_func)
        
        # Métricas
        accuracy_knn_cv = calculate_accuracy(cv_y_test, predictions_knn_cv)
        error_knn_cv = calculate_error(cv_y_test, predictions_knn_cv)
    
        accuracies_knn_cv.append(accuracy_knn_cv)
        errors_knn_cv.append(error_knn_cv)
    
    # c. Bootstrap (usando el mismo valor de k)
    accuracies_knn_bootstrap = []
    errors_knn_bootstrap = []
    
    num_bootstrap_samples = 100
    
    for _ in range(num_bootstrap_samples):
        # Muestreo bootstrap
        bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
        bootstrap_X_train = X_train[bootstrap_indices]
        bootstrap_y_train = y_train[bootstrap_indices]
    
        # Clasificación
        predictions_knn_bootstrap = knn_classifier(bootstrap_X_train, bootstrap_y_train, X_test, k_value, distance_func)
        
        # Métricas
        accuracy_knn_bootstrap = calculate_accuracy(y_test, predictions_knn_bootstrap)
        error_knn_bootstrap = calculate_error(y_test, predictions_knn_bootstrap)
    
        accuracies_knn_bootstrap.append(accuracy_knn_bootstrap)
        errors_knn_bootstrap.append(error_knn_bootstrap)
    
    # Resultados para el clasificador KNN
    print("\n   Resultados Clasificador KNN:")
    print(f"    Métrica utilizada: {distance_func.__name__}")
    print("\n    a. Entrenamiento y Prueba:")
    print(f"        Precisión: {accuracy_knn_train_test:.2%}")
    print(f"        Tasa de Error: {error_knn_train_test:.2%}")
    
    print("\n   b. K-fold Cross Validation:")
    print(f"        Precisión Promedio: {np.mean(accuracies_knn_cv):.2%}")
    print(f"        Tasa de Error Promedio: {np.mean(errors_knn_cv):.2%}")
    
    print("\n   c. Bootstrap:")
    print(f"        Precisión Promedio: {np.mean(accuracies_knn_bootstrap):.2%}")
    print(f"        Tasa de Error Promedio: {np.mean(errors_knn_bootstrap):.2%}")

# Eliminacion de 2 atributos para mejorar la eficiencia
def punto6(df,atributos, clase):
    print("\n DataFrame Original:")
    print(df)
    
    print("\n   Tienes los siguientes atributos:")
    # Asigna un número a cada columna
    numeros_columnas = {i: columna for i, columna in enumerate(atributos.columns)}

    # Imprime la relación entre el número y el nombre de la columna
    for numero, columna in numeros_columnas.items():
        print(f'    Atributos {numero}: {columna}')
    
    colAeliminar1 = int(input("\n    Ingresa el numero del primer atributo a eliminar:"))
    colAeliminar2 = int(input("\n    Ingresa el numero del segundo atributo a eliminar:"))
    

    print("\n a.Eliminar el primer Atributo")
    # Verifica que el índice sea válido
    if colAeliminar1 < len(atributos.columns):
        # Utiliza el método drop para eliminar la columna por su índice
        newdf1 = df.drop(df.columns[colAeliminar1], axis=1)        
        # Muestra el DataFrame después de eliminar la columna
        print(newdf1)
    else:
        print(f"El índice {colAeliminar1} no es válido para las columnas del DataFrame.")
    
    X_train, X_test, y_train, y_test = punto3(newdf1)
    # Punto 4
    punto4(X_train, X_test, y_train, y_test, distance_func=euclidean_distance)
    # Punto 5
    punto5(X_train, X_test, y_train, y_test,distance_func=euclidean_distance)

    
    print("\nb.Eliminar el segundo Atributo")    

    # Verifica que el índice sea válido
    if colAeliminar2 < len(atributos.columns):
        # Utiliza el método drop para eliminar la columna por su índice
        newdf2 = df.drop(df.columns[colAeliminar2], axis=1)     
        # Muestra el DataFrame después de eliminar la columna
        print(newdf2)
    else:
        print(f"El índice {colAeliminar2} no es válido para las columnas del DataFrame.")
    
    
    X_train, X_test, y_train, y_test = punto3(newdf2)
    # Punto 4
    punto4(X_train, X_test, y_train, y_test, distance_func=euclidean_distance)
    # Punto 5
    punto5(X_train, X_test, y_train, y_test,distance_func=euclidean_distance)
    
    print("\nc.Eliminar los dos atributos")
    # Verifica que el índice sea válido
    if colAeliminar1 < len(atributos.columns) and colAeliminar2 < len(atributos.columns):
        # Utiliza el método drop para eliminar la columna por su índice
        newdf = df.drop(df.columns[colAeliminar1], axis=1)     
        newdf3 = newdf.drop(df.columns[colAeliminar2], axis=1)     
        # Muestra el DataFrame después de eliminar la columna
        print(newdf3)
    else:
        print(f"El índice {colAeliminar1}  o el {colAeliminar2} no son válidos para las columnas del DataFrame.")
    
    
    X_train, X_test, y_train, y_test = punto3(newdf3)
    # Punto 4
    punto4(X_train, X_test, y_train, y_test, distance_func=euclidean_distance)
    # Punto 5
    punto5(X_train, X_test, y_train, y_test,distance_func=euclidean_distance)


# Eliminacion de uno o mas muestras para mejorar la eficiencia    
def punto7(df,atributos, y, subsets_por_clase):
    new_x = np.concatenate(subsets_por_clase)    
    print(new_x.shape)

    # Solicita al usuario los tamaños de salida
    print(f"Tenemos la siguiente cantidad de muestras {new_x.shape[0]}")
    print(f"Use la notacion de segmentación, Ej. [X:N], [:N], [X:], , etc")
    output_size = (input("Habilite los atributos (filas) que desea tener en su vector de salida.\n"))

    filas_seleccionadas = output_size.split(":")
    filas_seleccionadas = [int(splits) for splits in filas_seleccionadas]

    output_dataMod = y[filas_seleccionadas[0]: filas_seleccionadas[1]+1]
    new_x= new_x[filas_seleccionadas[0]: filas_seleccionadas[1]+1]
    #print(new_xMod)

    

    print("Conjunto de datos creado: \n")
    nuevodf = np.column_stack((new_x, output_dataMod))
    print(nuevodf)
    """
    #return nuevodf, new_x, output_dataMod
    column_names = atributos.columns
    column_names = column_names.append('Clase')
    print(column_names)
    
    # Extrae los nombres de las columnas de la primera fila
    #column_names = nuevodf[0, :-1].tolist() + ['Clase']

    # Crea un DataFrame utilizando los nombres de las columnas y los datos restantes
    #df = pd.DataFrame(nuevodf[1:], columns=column_names)

    df = pd.DataFrame(nuevodf, columns=column_names)

    X_train, X_test, y_train, y_test = punto3(df)
    # Punto 4
    punto4(X_train, X_test, y_train, y_test, distance_func=euclidean_distance)
    # Punto 5
    punto5(X_train, X_test, y_train, y_test,distance_func=euclidean_distance)
    """
def obtener_clases(x, y):
    ## Obtenemos la clase con base en sus etiquetas [y]
    clases_unicas = np.unique(y)

    ## Fragmentamos las clases en arrays diferentes 1 x clase
    subsets = [x[y == cls] for cls in clases_unicas]
    return clases_unicas, subsets
###########################################################################################################
if __name__ == "__main__":
    
    # Ubicacion DataSet
    dataset_path = 'iris.data'

    
    # Punto 1
    print("\n############################################################################")    
    print("\nPunto 1:")
    df, atributos, clases= punto1(dataset_path)

    # Punto 2
    print("\n############################################################################")    
    print("\nPunto 2:")
    punto2(df)

    # Punto 3    
    print("\n############################################################################")    
    print("\nPunto 3:")
    X_train, X_test, y_train, y_test = punto3(df)

    # Punto 4
    print("\n############################################################################")    
    print("\nPunto 4:")
    punto4(X_train, X_test, y_train, y_test, distance_func=euclidean_distance)

    # Punto 5
    print("\n############################################################################")    
    print("\nPunto 5:")
    punto5(X_train, X_test, y_train, y_test,distance_func=euclidean_distance)

    # Punto 6
    print("\n############################################################################")    
    print("\nPunto 6:")
    punto6(df, atributos, clases)

    # Punto 7
    print("\n############################################################################")    
    print("\nPunto 7:")
     # Obtener clases
    allclases, subsets_por_clase = obtener_clases(atributos, clases)
    punto7(df,atributos, clases, subsets_por_clase)

        