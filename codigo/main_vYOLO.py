import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. Cargar Modelos ---
modelo_yolo = YOLO('yolov8n-face-lindevs.pt')  # Modelo YOLOv8 para rostros
facenet = FaceNet()  # Extracción de embeddings

# --- 2. Cargar Base de Datos de Rostros Conocidos ---
embeddings_data = np.load("faces_embeddings_done_4classes.npz")  # Ajusta tu archivo
X = embeddings_data['arr_0']  # Embeddings
Y = embeddings_data['arr_1']   # Nombres

# --- 3. Entrenar k-NN ---
codificador = LabelEncoder()
Y_codificado = codificador.fit_transform(Y)
modelo_knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
modelo_knn.fit(X, Y_codificado)

# --- 4. Umbral para Rostros Desconocidos ---
UMBRAL_DISTANCIA = 0.7  # Ajustar según pruebas

# --- 5. Iniciar Cámara ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 6. Detección con YOLOv8 ---
    resultados = modelo_yolo(frame, verbose=False)
    
    for r in resultados:
        boxes = r.boxes.xyxy.cpu().numpy()  # Coordenadas de los bounding boxes
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # --- 7. Extraer ROI del rostro ---
            rostro = frame[y1:y2, x1:x2]
            if rostro.size == 0:  # Evitar errores si el ROI está vacío
                continue
                
            # --- 8. Preprocesamiento para FaceNet ---
            rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
            rostro_resized = cv2.resize(rostro_rgb, (160, 160))
            rostro_expanded = np.expand_dims(rostro_resized, axis=0)
            
            # --- 9. Extraer Embedding ---
            embedding = facenet.embeddings(rostro_expanded)
            
            # --- 10. Comparar con rostros conocidos ---
            distancias, indices = modelo_knn.kneighbors(embedding, n_neighbors=1)
            distancia_minima = distancias[0][0]
            
            # --- 11. Clasificar ---
            if distancia_minima > UMBRAL_DISTANCIA:
                nombre = "DESCONOCIDO"
                color = (0, 0, 255)  # Rojo
            else:
                etiqueta = modelo_knn.predict(embedding)
                nombre = codificador.inverse_transform(etiqueta)[0]
                color = (0, 255, 0)  # Verde
            
            # --- 12. Dibujar resultados ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, nombre, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # --- 13. Mostrar frame ---
    cv2.imshow("Reconocimiento Facial (YOLOv8 + FaceNet)", frame)
    
    # --- 14. Salir con 'q' ---
    if cv2.waitKey(1) == ord('q'):
        break

# --- 15. Liberar recursos ---
cap.release()
cv2.destroyAllWindows()