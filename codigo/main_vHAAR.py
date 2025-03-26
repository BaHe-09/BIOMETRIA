# IMPORTAR LIBRERÍAS
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# INICIALIZACIÓN
facenet = FaceNet()

# Cargar los embeddings y el modelo de clasificación
embeddings_caras = np.load("faces_embeddings_done_4classes.npz")
X = embeddings_caras['arr_0']  # Embeddings de entrenamiento
Y = embeddings_caras['arr_1']
codificador = LabelEncoder()
codificador.fit(Y)
clasificador_haar = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
modelo = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Configurar umbral de distancia para rostros desconocidos
UMBRAL_DISTANCIA = 0.8  # Ajusta este valor según tus necesidades

# Función para calcular la distancia mínima a los embeddings conocidos
def distancia_minima_embedding(embedding, embeddings_conocidos):
    distancias = np.linalg.norm(embeddings_conocidos - embedding, axis=1)
    return np.min(distancias)

# Habilitar ejecución en modo eager de TensorFlow
tf.compat.v1.enable_eager_execution()

# INICIAR CÁMARA
captura = cv.VideoCapture(0)  # Cambia el índice si es necesario

# Verificar si la cámara se ha abierto correctamente
if not captura.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# BUCLE PRINCIPAL
while captura.isOpened():
    _, cuadro = captura.read()
    imagen_rgb = cv.cvtColor(cuadro, cv.COLOR_BGR2RGB)
    imagen_gris = cv.cvtColor(cuadro, cv.COLOR_BGR2GRAY)
    caras = clasificador_haar.detectMultiScale(imagen_gris, 1.3, 5)
    
    for x, y, w, h in caras:
        rostro = imagen_rgb[y:y+h, x:x+w]
        rostro = cv.resize(rostro, (160, 160))  # 1x160x160x3
        rostro = np.expand_dims(rostro, axis=0)
        embedding = facenet.embeddings(rostro)
        
        # Calcular distancia al embedding más cercano
        distancia = distancia_minima_embedding(embedding, X)
        
        if distancia > UMBRAL_DISTANCIA:
            nombre_final = "Sin identificar"
        else:
            nombre_predicho = modelo.predict(embedding)
            nombre_final = codificador.inverse_transform(nombre_predicho)[0]
        
        cv.rectangle(cuadro, (x, y), (x+w, y+h), (255, 0, 255), 10)
        cv.putText(cuadro, str(nombre_final), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Reconocimiento Facial", cuadro)
    if cv.waitKey(1) & ord('q') == 27:  # Presiona 'q' para salir
        break

# LIBERAR RECURSOS
captura.release()
cv.destroyAllWindows()