import cv2
import mediapipe as mp
import numpy as np
import os

# Inicializar MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Listas para almacenar los datos y etiquetas
data = []
labels = []

# Verificar si los archivos ya existen para continuar agregando datos
if os.path.exists('rps_dataset.npy') and os.path.exists('rps_labels.npy'):
    data = np.load('rps_dataset.npy').tolist()
    labels = np.load('rps_labels.npy').tolist()

# Inicializar la detección de manos
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
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

                # Obtener las coordenadas de los landmarks
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                print("Landmarks detectados:", landmarks)

                # Guardar coordenadas y esperar la etiqueta
                print("Presiona 0 (piedra), 1 (papel), 2 (tijeras), o 's' para saltar:")
                key = cv2.waitKey(0) & 0xFF  # Espera una tecla para etiquetar

                if key in [ord('0'), ord('1'), ord('2')]:
                    label = int(chr(key))  # Convertir la tecla presionada a número
                    data.append(landmarks)
                    labels.append(label)
                    print(f"Etiqueta {label} guardada.")

        # Mostrar el video con las manos detectadas
        cv2.imshow('Manos detectadas', frame)

        # Salir del bucle con la tecla 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Guardar los datos en archivos .npy
np.save('rps_dataset.npy', np.array(data))
np.save('rps_labels.npy', np.array(labels))

print("Datos guardados exitosamente.")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()