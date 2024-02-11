import numpy as np

class CustomClassifier:
    def __init__(self, input_size, output_size):
        # Inicialización de la clase con el tamaño de entrada y salida
        self.input_size = input_size
        self.output_size = output_size
        # Inicialización de matrices para almacenar datos de entrada y salida
        self.input_data = np.array([]).reshape(0, input_size)
        self.output_data = np.array([])

    def add_data_point(self, input_vector, output_value):
        # Agrega un punto de datos al conjunto de entrenamiento
        self.input_data = np.vstack([self.input_data, input_vector])
        self.output_data = np.append(self.output_data, output_value)

    def euclidean_distance(self, x1, x2):
        # Calcula la distancia euclidiana entre dos vectores
        return np.sqrt(np.sum((x1 - x2)**2))

    def manhattan_distance(self, x1, x2):
        # Calcula la distancia de Manhattan entre dos vectores
        return np.sum(np.abs(x1 - x2))

    def train_knn(self, k):
        # Establece el valor de k para el algoritmo k-NN
        self.k = k

    def predict_knn(self, input_vector):
        # Predice la clase utilizando el algoritmo k-NN
        distances = [self.euclidean_distance(input_vector, x) for x in self.input_data]
        indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.output_data[indices]
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def train_min_distance(self):
        # Calcular los promedios de los datos de entrenamiento para Mínima Distancia por clase
        unique_classes = np.unique(self.output_data)
        self.class_means = {label: np.mean(self.input_data[self.output_data == label], axis=0) for label in unique_classes}

    def predict_min_distance(self, input_vector):
        # Predice la clase utilizando el algoritmo de Mínima Distancia
        distances = {label: self.euclidean_distance(input_vector, mean) for label, mean in self.class_means.items()}
        min_class = min(distances, key=distances.get)
        return min_class

def load_data_from_file(file_path):
    # Carga datos desde un archivo CSV
    data = np.loadtxt(file_path, delimiter=',', dtype='str')
    input_data = data[:, :-1].astype(float)
    output_data = data[:, -1]
    return input_data, output_data

if __name__ == "__main__":
    # Solicita al usuario los tamaños de entrada y salida
    input_size = int(input("Ingrese el tamaño del vector de entrada: "))
    output_size = int(input("Ingrese el tamaño del vector de salida: "))

    # Crea un clasificador personalizado
    classifier = CustomClassifier(input_size, output_size)

    # Solicita al usuario la ruta del archivo de datos de entrenamiento
    file_path = input("Ingrese la ruta del archivo de datos de entrenamiento (CSV): ")
    input_data, output_data = load_data_from_file(file_path)

    # Agrega los datos de entrenamiento al clasificador
    for i in range(len(output_data)):
        classifier.add_data_point(input_data[i], output_data[i])

    # Solicita al usuario elegir entre k-NN y Mínima Distancia
    classifier_type = input("Elija el clasificador (knn / min): ")

    # Entrenamiento según el tipo de clasificador elegido
    if classifier_type == 'knn':
        k_value = int(input("Ingrese el valor de K para KNN: "))
        classifier.train_knn(k_value)
    elif classifier_type == 'min':
        classifier.train_min_distance()

    # Realiza predicciones según el tipo de clasificador elegido
    num_predictions = int(input("Ingrese el número de predicciones a realizar: "))
    for _ in range(num_predictions):
        # Solicita al usuario ingresar el vector de entrada para predecir
        input_vector = np.array([float(input(f"Ingrese el valor {i+1} del vector de entrada para predecir: ")) for i in range(input_size)])
        
        # Realiza la predicción y muestra el resultado
        if classifier_type == 'knn':
            prediction = classifier.predict_knn(input_vector)
        elif classifier_type == 'min':
            prediction = classifier.predict_min_distance(input_vector)

        print(f"La predicción es: {prediction}")
