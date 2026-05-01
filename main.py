import mediapipe as mp
import cv2, math
from tkinter import *
from PIL import Image, ImageTk

win = Tk()
width = win.winfo_screenwidth()
height = win.winfo_screenheight()
win.geometry("%dx%d" % (width, height))
win.title('Traductor Lenguaje de Señas')
win.configure(bg='#000000')

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class TraductorLenguajeSeñas:
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.current_gesture = None
        self.CountGesture = StringVar()
        # Historial de posiciones para detectar movimiento (J, Z, Ñ)
        self.landmark_history = []

    def detect_gesture(self, image):
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                self.landmark_history.append({
                    'pinky': (lm.landmark[20].x, lm.landmark[20].y),
                    'index': (lm.landmark[8].x, lm.landmark[8].y),
                    'middle': (lm.landmark[12].x, lm.landmark[12].y),
                })
                if len(self.landmark_history) > 20:
                    self.landmark_history.pop(0)
                self.current_gesture = self.get_gesture(lm)
            else:
                self.landmark_history.clear()

    def get_gesture(self, hand_landmarks):
        def letraA():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            dedos_hacia_abajo = (
                index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y
            )
            pulgar_estirado_lateral = abs(thumb_tip.x - thumb_mcp.x) > 0.06
            return dedos_hacia_abajo and pulgar_estirado_lateral

        def letraB():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            index_mcp = hand_landmarks.landmark[5]
            cuatro_dedos_up = (
                index_tip.y < index_pip.y + 0.05 and
                middle_tip.y < middle_pip.y + 0.05 and
                ring_tip.y < ring_pip.y + 0.05 and
                pinky_tip.y < pinky_pip.y + 0.05
            )
            dedos_juntos = (
                abs(index_tip.x - middle_tip.x) < 0.08 and
                abs(middle_tip.x - ring_tip.x) < 0.08 and
                abs(ring_tip.x - pinky_tip.x) < 0.08
            )
            pulgar_doblado = thumb_tip.y > index_mcp.y - 0.05
            return cuatro_dedos_up and dedos_juntos and pulgar_doblado

        def letraC():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            thumb_mcp = hand_landmarks.landmark[2]
            dedos_curvados = (
                index_pip.y - 0.05 < index_tip.y < index_pip.y + 0.1 and
                middle_pip.y - 0.05 < middle_tip.y < middle_pip.y + 0.1 and
                ring_pip.y - 0.05 < ring_tip.y < ring_pip.y + 0.1 and
                pinky_pip.y - 0.05 < pinky_tip.y < pinky_pip.y + 0.1
            )
            dedos_juntos = (
                abs(index_tip.x - middle_tip.x) < 0.08 and
                abs(middle_tip.x - ring_tip.x) < 0.08 and
                abs(ring_tip.x - pinky_tip.x) < 0.08
            )
            pulgar_curvado = (thumb_tip.y > thumb_ip.y - 0.05) and (thumb_ip.y > thumb_mcp.y - 0.05)
            return dedos_curvados and dedos_juntos and pulgar_curvado

        def letraD():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            index_up = index_tip.y < index_pip.y - 0.04
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            thumb_near_middle = math.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2) < 0.12
            return index_up and middle_down and ring_down and pinky_down and thumb_near_middle

        def letraE():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            all_hooked = (
                index_tip.y > index_pip.y - 0.02 and
                middle_tip.y > middle_pip.y - 0.02 and
                ring_tip.y > ring_pip.y - 0.02 and
                pinky_tip.y > pinky_pip.y - 0.02
            )
            tips_at_mcp = (
                abs(index_tip.y - index_mcp.y) < 0.1 and
                abs(middle_tip.y - middle_mcp.y) < 0.1
            )
            return all_hooked and tips_at_mcp

        def letraF():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            index_down = index_tip.y > index_pip.y
            middle_up = middle_tip.y < middle_pip.y - 0.03
            ring_up = ring_tip.y < ring_pip.y - 0.03
            pinky_up = pinky_tip.y < pinky_pip.y - 0.03
            thumb_near_index = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) < 0.1
            return index_down and middle_up and ring_up and pinky_up and thumb_near_index

        def letraG():
            # Índice Y pulgar apuntando al costado (como pistola horizontal)
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_mcp = hand_landmarks.landmark[5]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            index_horizontal = abs(index_tip.x - index_mcp.x) > abs(index_tip.y - index_mcp.y) * 1.5
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            # El pulgar también se extiende (no está doblado)
            thumb_extended = abs(thumb_tip.x - thumb_ip.x) > 0.02
            return index_horizontal and middle_down and ring_down and pinky_down and thumb_extended

        def letraH():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_horizontal = abs(index_tip.x - index_mcp.x) > abs(index_tip.y - index_mcp.y)
            middle_horizontal = abs(middle_tip.x - middle_mcp.x) > abs(middle_tip.y - middle_mcp.y)
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            fingers_aligned = abs(index_tip.y - middle_tip.y) < 0.06
            return index_horizontal and middle_horizontal and ring_down and pinky_down and fingers_aligned

        def letraI():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            pinky_up = pinky_tip.y < pinky_pip.y - 0.04
            index_down = index_tip.y > index_pip.y
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            return pinky_up and index_down and middle_down and ring_down

        def letraJ():
            # Posición I (meñique arriba) + movimiento en J con el meñique
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            pinky_up = pinky_tip.y < pinky_pip.y - 0.04
            index_down = index_tip.y > index_pip.y
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            static_I = pinky_up and index_down and middle_down and ring_down
            if not static_I or len(self.landmark_history) < 10:
                return False
            # El meñique traza una J: movimiento vertical + componente horizontal al final
            pinky_ys = [h['pinky'][1] for h in self.landmark_history[-15:]]
            pinky_xs = [h['pinky'][0] for h in self.landmark_history[-15:]]
            y_range = max(pinky_ys) - min(pinky_ys)
            x_range = max(pinky_xs) - min(pinky_xs)
            return y_range > 0.08 and x_range > 0.05

        def letraK():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            thumb_tip = hand_landmarks.landmark[4]
            index_up = index_tip.y < index_pip.y - 0.04
            middle_up = middle_tip.y < middle_pip.y - 0.04
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            fingers_spread = abs(index_tip.x - middle_tip.x) > 0.05
            thumb_between = (min(index_mcp.x, middle_mcp.x) - 0.05 < thumb_tip.x < max(index_mcp.x, middle_mcp.x) + 0.05)
            return index_up and middle_up and ring_down and pinky_down and fingers_spread and thumb_between

        def letraL():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            index_up = index_tip.y < index_pip.y - 0.04
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            thumb_extended = abs(thumb_tip.x - thumb_mcp.x) > 0.07
            return index_up and middle_down and ring_down and pinky_down and thumb_extended

        def letraM():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            index_mcp = hand_landmarks.landmark[5]
            all_down = (
                index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y
            )
            three_together = (
                abs(index_tip.x - middle_tip.x) < 0.07 and
                abs(middle_tip.x - ring_tip.x) < 0.07
            )
            pinky_separate = abs(pinky_tip.x - ring_tip.x) > 0.04
            thumb_tucked = thumb_tip.y > index_mcp.y - 0.05
            return all_down and three_together and pinky_separate and thumb_tucked

        def letraN():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            index_mcp = hand_landmarks.landmark[5]
            all_down = (
                index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y
            )
            two_together = abs(index_tip.x - middle_tip.x) < 0.07
            ring_separate = abs(ring_tip.x - middle_tip.x) > 0.07
            thumb_tucked = thumb_tip.y > index_mcp.y - 0.05
            return all_down and two_together and ring_separate and thumb_tucked

        def letraN_tilde():
            # Posición N + movimiento lateral (ondulación)
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            index_mcp = hand_landmarks.landmark[5]
            all_down = (
                index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y
            )
            two_together = abs(index_tip.x - middle_tip.x) < 0.07
            ring_separate = abs(ring_tip.x - middle_tip.x) > 0.07
            thumb_tucked = thumb_tip.y > index_mcp.y - 0.05
            static_N = all_down and two_together and ring_separate and thumb_tucked
            if not static_N or len(self.landmark_history) < 10:
                return False
            # Detectar ondulación: el índice oscila horizontalmente
            index_xs = [h['index'][0] for h in self.landmark_history[-15:]]
            x_range = max(index_xs) - min(index_xs)
            return x_range > 0.07

        def letraO():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            all_curved = (
                abs(index_tip.y - index_pip.y) < 0.06 and
                abs(middle_tip.y - middle_pip.y) < 0.06 and
                abs(ring_tip.y - ring_pip.y) < 0.06 and
                abs(pinky_tip.y - pinky_pip.y) < 0.06
            )
            thumb_close = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) < 0.1
            return all_curved and thumb_close

        def letraP():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            index_points_down = index_tip.y > index_mcp.y + 0.05
            middle_points_down = middle_tip.y > middle_mcp.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            thumb_out = abs(thumb_tip.x - thumb_ip.x) > 0.04
            return index_points_down and middle_points_down and ring_down and pinky_down and thumb_out

        def letraQ():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_mcp = hand_landmarks.landmark[5]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            index_points_down = index_tip.y > index_mcp.y + 0.05
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            thumb_down = thumb_tip.y > thumb_mcp.y
            return index_points_down and middle_down and ring_down and pinky_down and thumb_down

        def letraR():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_up = index_tip.y < index_pip.y - 0.03
            middle_up = middle_tip.y < middle_pip.y - 0.03
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            fingers_crossed = abs(index_tip.x - middle_tip.x) < 0.04
            return index_up and middle_up and ring_down and pinky_down and fingers_crossed

        def letraS():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            index_mcp = hand_landmarks.landmark[5]
            all_down = (
                index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y
            )
            thumb_over = abs(thumb_tip.x - index_mcp.x) < 0.12
            return all_down and thumb_over

        def letraT():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            index_mcp = hand_landmarks.landmark[5]
            all_down = (
                index_tip.y > index_pip.y and middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y
            )
            thumb_sticks_up = thumb_tip.y < index_mcp.y
            return all_down and thumb_sticks_up

        def letraU():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_up = index_tip.y < index_pip.y - 0.04
            middle_up = middle_tip.y < middle_pip.y - 0.04
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            fingers_together = abs(index_tip.x - middle_tip.x) < 0.05
            return index_up and middle_up and ring_down and pinky_down and fingers_together

        def letraV():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_up = index_tip.y < index_pip.y - 0.04
            middle_up = middle_tip.y < middle_pip.y - 0.04
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            fingers_spread = abs(index_tip.x - middle_tip.x) > 0.05
            return index_up and middle_up and ring_down and pinky_down and fingers_spread

        def letraW():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_up = index_tip.y < index_pip.y - 0.03
            middle_up = middle_tip.y < middle_pip.y - 0.03
            ring_up = ring_tip.y < ring_pip.y - 0.03
            pinky_down = pinky_tip.y > pinky_pip.y
            fingers_spread = abs(index_tip.x - ring_tip.x) > 0.08
            return index_up and middle_up and ring_up and pinky_down and fingers_spread

        def letraX():
            index_tip = hand_landmarks.landmark[8]
            index_dip = hand_landmarks.landmark[7]
            index_pip = hand_landmarks.landmark[6]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            index_hooked = (index_dip.y < index_pip.y) and (index_tip.y > index_dip.y + 0.02)
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            return index_hooked and middle_down and ring_down and pinky_down

        def letraY():
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            index_down = index_tip.y > index_pip.y
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_up = pinky_tip.y < pinky_pip.y - 0.03
            thumb_extended = abs(thumb_tip.x - thumb_mcp.x) > 0.07
            return index_down and middle_down and ring_down and pinky_up and thumb_extended

        def letraZ():
            # Posición: índice arriba, otros abajo, pulgar pegado
            # + movimiento horizontal amplio (traza una Z)
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            index_pip = hand_landmarks.landmark[6]
            index_mcp = hand_landmarks.landmark[5]
            middle_pip = hand_landmarks.landmark[10]
            ring_pip = hand_landmarks.landmark[14]
            pinky_pip = hand_landmarks.landmark[18]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            index_up = index_tip.y < index_pip.y - 0.04
            middle_down = middle_tip.y > middle_pip.y
            ring_down = ring_tip.y > ring_pip.y
            pinky_down = pinky_tip.y > pinky_pip.y
            thumb_close = abs(thumb_tip.x - thumb_mcp.x) < 0.07
            static_pos = index_up and middle_down and ring_down and pinky_down and thumb_close
            if not static_pos or len(self.landmark_history) < 10:
                return False
            # El índice se mueve horizontalmente de forma amplia (traza la Z)
            index_xs = [h['index'][0] for h in self.landmark_history[-15:]]
            x_range = max(index_xs) - min(index_xs)
            return x_range > 0.1

        # Detección en orden (más específicas primero)
        if letraA():
            self.CountGesture.set("Letra A")
            return "A"
        if letraB():
            self.CountGesture.set("Letra B")
            return "B"
        if letraC():
            self.CountGesture.set("Letra C")
            return "C"
        if letraD():
            self.CountGesture.set("Letra D")
            return "D"
        if letraE():
            self.CountGesture.set("Letra E")
            return "E"
        if letraF():
            self.CountGesture.set("Letra F")
            return "F"
        if letraG():
            self.CountGesture.set("Letra G")
            return "G"
        if letraH():
            self.CountGesture.set("Letra H")
            return "H"
        if letraJ():
            self.CountGesture.set("Letra J")
            return "J"
        if letraI():
            self.CountGesture.set("Letra I")
            return "I"
        if letraK():
            self.CountGesture.set("Letra K")
            return "K"
        if letraL():
            self.CountGesture.set("Letra L")
            return "L"
        if letraM():
            self.CountGesture.set("Letra M")
            return "M"
        if letraN_tilde():
            self.CountGesture.set("Letra Ñ")
            return "Ñ"
        if letraN():
            self.CountGesture.set("Letra N")
            return "N"
        if letraO():
            self.CountGesture.set("Letra O")
            return "O"
        if letraP():
            self.CountGesture.set("Letra P")
            return "P"
        if letraQ():
            self.CountGesture.set("Letra Q")
            return "Q"
        if letraR():
            self.CountGesture.set("Letra R")
            return "R"
        if letraS():
            self.CountGesture.set("Letra S")
            return "S"
        if letraT():
            self.CountGesture.set("Letra T")
            return "T"
        if letraU():
            self.CountGesture.set("Letra U")
            return "U"
        if letraV():
            self.CountGesture.set("Letra V")
            return "V"
        if letraW():
            self.CountGesture.set("Letra W")
            return "W"
        if letraX():
            self.CountGesture.set("Letra X")
            return "X"
        if letraY():
            self.CountGesture.set("Letra Y")
            return "Y"
        if letraZ():
            self.CountGesture.set("Letra Z")
            return "Z"

        self.CountGesture.set("")
        return None

    def get_current_gesture(self):
        return self.current_gesture

    def release(self):
        self.hands.close()

sign_lang_conv = TraductorLenguajeSeñas()
cap = cv2.VideoCapture(0)

# Video ocupa toda la pantalla
label1 = Label(win, bg='#000000')
label1.place(x=0, y=0, width=width, height=height)

# Etiquetas de gesto creadas una sola vez, flotando sobre el video
crrgesture_lbl = Label(win, text='Gesto:', font=('Calibri', 36, 'bold'),
                        bd=3, bg='#20262E', fg='#F5EAEA', relief=GROOVE)
crrgesture_lbl.place(x=0, y=height - 80)
status_lbl = Label(win, textvariable=sign_lang_conv.CountGesture,
                   font=('Georgia', 36, 'bold'), bd=3, bg='#20262E',
                   width=20, fg='#F5EAEA', relief=GROOVE)
status_lbl.place(x=200, y=height - 80)

def select_img():
    _, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    sign_lang_conv.detect_gesture(frame)
    gesture = sign_lang_conv.get_current_gesture()
    if gesture:
        cv2.putText(frame, gesture, (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 255, 0), 5, cv2.LINE_AA)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(framergb)
    finalImage = ImageTk.PhotoImage(img)
    label1.configure(image=finalImage)
    label1.image = finalImage
    win.after(1, select_img)

select_img()
win.mainloop()
