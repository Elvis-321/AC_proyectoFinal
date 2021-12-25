import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import ctypes

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
color_mouse_pointer = (255, 0, 255)

# Puntos de la pantalla
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
ancho, alto = user32.GetSystemMetrics(0),user32.GetSystemMetrics(1)
SCREEN_GAME_X_INI = 0
SCREEN_GAME_Y_INI = 0
SCREEN_GAME_X_FIN = ancho
SCREEN_GAME_Y_FIN = alto

# relacion de aspecto entre ancho y alto de la pantalla
aspect_ratio_screen = (SCREEN_GAME_X_FIN - SCREEN_GAME_X_INI) / (SCREEN_GAME_Y_FIN - SCREEN_GAME_Y_INI)

X_Y_INI = 120 #espacio entre el marco del frame y el rectangulo

def calculate_distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p1 - p2)

def detect_clic(hand_landmarks):
    clic = False
    color_base = (255, 0, 0)
    color_index = (0, 255, 0)

    x_base1 = int(hand_landmarks.landmark[0].x * width)
    y_base1 = int(hand_landmarks.landmark[0].y * height)

    x_base2 = int(hand_landmarks.landmark[4].x * width)
    y_base2 = int(hand_landmarks.landmark[4].y * height)

    x_index = int(hand_landmarks.landmark[8].x * width)
    y_index = int(hand_landmarks.landmark[8].y * height)

    d_base = calculate_distance(x_base1, y_base1, x_base2, y_base2)
    d_base_index = calculate_distance(x_base1, y_base1, x_index, y_index)
    if d_base_index < d_base:
        clic = True
        color_base = (255, 0, 255)
        color_index = (255, 0, 255)

    cv2.circle(frame, (x_base1, y_base1), 10, color_base, 3)
    cv2.circle(frame, (x_index, y_index), 10, color_index, 3)
    cv2.line(frame, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
    cv2.line(frame, (x_base1, y_base1), (x_index, y_index), color_index, 3)
    return clic

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)

        # Dibujando un área proporcional a la pantalla
        area_width = width - X_Y_INI * 2
        area_height = int(area_width / aspect_ratio_screen)
        aux_image = np.zeros(frame.shape, np.uint8)
        aux_image = cv2.rectangle(aux_image, (X_Y_INI, X_Y_INI), (X_Y_INI + area_width, X_Y_INI +area_height), (255, 0, 0), 0)
        frame = cv2.addWeighted(frame, 1, aux_image, 0.7, 0)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:

                # ubicacion de los puntos "x" y "y" para el mouse
                x = int(hand_landmarks.landmark[4].x * width)
                y = int(hand_landmarks.landmark[4].y * height)

                # movimiento de x y y dentro del area
                xm = np.interp(x, (X_Y_INI, X_Y_INI + area_width), (SCREEN_GAME_X_INI, SCREEN_GAME_X_FIN))
                ym = np.interp(y, (X_Y_INI, X_Y_INI + area_height), (SCREEN_GAME_Y_INI, SCREEN_GAME_Y_FIN))

                pyautogui.moveTo(int(xm), int(ym)) # mover el mouse
                if detect_clic(hand_landmarks):
                   pyautogui.click()

                #visualizacion de los circulos
                cv2.circle(frame, (x, y), 10, color_mouse_pointer, 3)
        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
