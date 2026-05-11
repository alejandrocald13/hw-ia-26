import math
import cv2
import mediapipe as mp
import numpy as np

def dedo_levantado(landmarks, dedo_tip, dedo_dip, dedo_pip, dedo_mcp):
    return (landmarks[dedo_tip].y < landmarks[dedo_dip].y and
            landmarks[dedo_dip].y < landmarks[dedo_pip].y and
            landmarks[dedo_pip].y < landmarks[dedo_mcp].y)

def distancia_dedos(landmarks, tip1, tip2, image_w, image_h):
    """Retorna distancia en píxeles y coordenadas de ambos puntos"""
    x1 = int(landmarks[tip1].x * image_w)
    y1 = int(landmarks[tip1].y * image_h)
    x2 = int(landmarks[tip2].x * image_w)
    y2 = int(landmarks[tip2].y * image_h)
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist, (x1, y1), (x2, y2)

def dibujar_barra(image, distancia, dist_min=20, dist_max=300):
    """Dibuja una barra que crece/decrece según la distancia"""
    h, w = image.shape[:2]

    # Normalizar distancia entre 0 y 1
    ratio = min(max((distancia - dist_min) / (dist_max - dist_min), 0), 1)

    # Configuración de la barra
    barra_x = 30
    barra_y_top = 100
    barra_y_bottom = h - 100
    barra_alto_total = barra_y_bottom - barra_y_top
    barra_ancho = 30

    # Fondo de la barra (gris)
    cv2.rectangle(image,
                  (barra_x, barra_y_top),
                  (barra_x + barra_ancho, barra_y_bottom),
                  (80, 80, 80), -1)

    # Relleno según distancia
    relleno_alto = int(barra_alto_total * ratio)
    relleno_y = barra_y_bottom - relleno_alto

    # Color: verde cuando abierto, rojo cuando cerrado
    color = (
        int(255 * (1 - ratio)),  # R
        int(255 * ratio),         # G
        50                        # B
    )

    cv2.rectangle(image,
                  (barra_x, relleno_y),
                  (barra_x + barra_ancho, barra_y_bottom),
                  color, -1)

    # Borde de la barra
    cv2.rectangle(image,
                  (barra_x, barra_y_top),
                  (barra_x + barra_ancho, barra_y_bottom),
                  (255, 255, 255), 2)

    # Texto con la distancia
    cv2.putText(image, f"{int(distancia)}px",
                (barra_x - 5, barra_y_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print('No se pudo obtener la imagen')
            continue

        image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                dedos = {
                    'pulgar': (4, 3, 2, 1),
                    'indice': (8, 7, 6, 5),
                    'medio':  (12, 11, 10, 9),
                    'anular': (16, 15, 14, 13),
                    'peque':  (20, 19, 18, 17)
                }

                dedos_levantados = []
                for nombre, (tip, dip, pip, mcp) in dedos.items():
                    if dedo_levantado(landmarks, tip, dip, pip, mcp):
                        dedos_levantados.append(nombre)

                cv2.putText(image, f"Dedos: {', '.join(dedos_levantados)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # ── Distancia pulgar-índice ──────────────────────────
                dist, pt_pulgar, pt_indice = distancia_dedos(landmarks, 4, 8, w, h)

                # Línea entre los dos dedos
                cv2.line(image, pt_pulgar, pt_indice, (0, 200, 255), 3)

                # Círculos en las puntas
                cv2.circle(image, pt_pulgar, 10, (0, 200, 255), -1)
                cv2.circle(image, pt_indice, 10, (0, 200, 255), -1)

                # Etiqueta en el punto medio
                mid = ((pt_pulgar[0] + pt_indice[0]) // 2,
                       (pt_pulgar[1] + pt_indice[1]) // 2)
                cv2.putText(image, f"{int(dist)}px", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                # Barra lateral
                dibujar_barra(image, dist)

        cv2.imshow("Deteccion de manos", image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Deteccion de manos", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()