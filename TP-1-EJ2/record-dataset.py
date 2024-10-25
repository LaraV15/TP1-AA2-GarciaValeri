import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Lista para almacenar los datos (id, etiqueta, coordenadas)
dataset = []

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Contador para el ID de cada muestra
id_contador = 0

# Inicializar la detección de manos
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("No se pudo acceder a la cámara.")
            break

        # Convertir BGR a RGB (MediaPipe trabaja con imágenes en RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen en busca de manos
        results = hands.process(frame_rgb)

        # Si se detectan manos, dibujar los puntos clave en la imagen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostrar el video con las manos detectadas
        cv2.imshow('Manos detectadas', frame)

        # Capturar la tecla presionada
        key = cv2.waitKey(5) & 0xFF

        # Si se presiona '0', '1' o '2'
        if key in [ord('0'), ord('1'), ord('2')]:
            # Obtener el identificador (0: piedra, 1: papel, 2: tijera)
            etiqueta = int(chr(key))

            # Si hay manos detectadas
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Obtener las coordenadas de los landmarks
                    coordenadas = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
                    # Agregar datos al dataset: [id, etiqueta, coordenadas_normalizadas]
                    dataset.append([id_contador, etiqueta, coordenadas])
                    id_contador += 1  # Incrementar el ID
                    # Notificar que se han guardado las coordenadas
                    print(f"Datos guardados - ID: {id_contador}, Etiqueta: {etiqueta}")
        # Salir del bucle con la tecla 'q'
        if key == ord('q'):
            break
# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Almacenar los datos en el archivo rps_dataset.npy
dataset_array = np.array(dataset, dtype=object)
np.save('rps_dataset.npy', dataset_array)
print("Dataset guardado en 'rps_dataset.npy'")