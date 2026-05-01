import streamlit as st
import cv2
import math
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands

st.set_page_config(page_title="Traductor LSC", layout="wide")
st.title("Traductor Lengua de Señas Colombiana")

class TraductorLenguajeSeñas:
    def __init__(self):
        self.current_gesture = None
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
                    'wrist': (lm.landmark[0].x, lm.landmark[0].y),
                })
                if len(self.landmark_history) > 25:
                    self.landmark_history.pop(0)
                self.current_gesture = self.get_gesture(lm)
                mp_drawing.draw_landmarks(image, lm, mp_hands.HAND_CONNECTIONS)
            else:
                self.landmark_history.clear()
                self.current_gesture = None
        return image

    def get_gesture(self, lm):
        thumb_tip  = lm.landmark[4]
        thumb_ip   = lm.landmark[3]
        thumb_mcp  = lm.landmark[2]

        idx_tip = lm.landmark[8]; idx_dip = lm.landmark[7]
        idx_pip = lm.landmark[6]; idx_mcp = lm.landmark[5]

        mid_tip = lm.landmark[12]; mid_dip = lm.landmark[11]
        mid_pip = lm.landmark[10]; mid_mcp = lm.landmark[9]

        rng_tip = lm.landmark[16]
        rng_pip = lm.landmark[14]; rng_mcp = lm.landmark[13]

        pnk_tip = lm.landmark[20]; pnk_dip = lm.landmark[19]
        pnk_pip = lm.landmark[18]; pnk_mcp = lm.landmark[17]

        wrist = lm.landmark[0]

        def dist(a, b):
            return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

        idx_up  = idx_tip.y < idx_pip.y - 0.03
        mid_up  = mid_tip.y < mid_pip.y - 0.03
        rng_up  = rng_tip.y < rng_pip.y - 0.03
        pnk_up  = pnk_tip.y < pnk_pip.y - 0.03

        idx_dn  = idx_tip.y > idx_pip.y + 0.01
        mid_dn  = mid_tip.y > mid_pip.y + 0.01
        rng_dn  = rng_tip.y > rng_pip.y + 0.01
        pnk_dn  = pnk_tip.y > pnk_pip.y + 0.01

        thumb_out = abs(thumb_tip.x - thumb_mcp.x) > 0.07

        # A: puño cerrado, pulgar al lado (no encima)
        def letraA():
            punio = idx_dn and mid_dn and rng_dn and pnk_dn
            pulgar_lateral = abs(thumb_tip.x - thumb_mcp.x) > 0.05 and thumb_tip.y > wrist.y
            pulgar_no_encima = thumb_tip.y > idx_mcp.y
            return punio and pulgar_lateral and pulgar_no_encima

        # B: 4 dedos juntos arriba, pulgar doblado dentro
        def letraB():
            cuatro_up = idx_up and mid_up and rng_up and pnk_up
            juntos = (abs(idx_tip.x - mid_tip.x) < 0.06 and
                      abs(mid_tip.x - rng_tip.x) < 0.06 and
                      abs(rng_tip.x - pnk_tip.x) < 0.06)
            pulgar_dentro = not thumb_out
            return cuatro_up and juntos and pulgar_dentro

        # C: mano curvada en C
        def letraC():
            semi = lambda tip, pip: abs(tip.y - pip.y) < 0.07
            todos_semi = semi(idx_tip, idx_pip) and semi(mid_tip, mid_pip) and semi(rng_tip, rng_pip) and semi(pnk_tip, pnk_pip)
            juntos = abs(idx_tip.x - mid_tip.x) < 0.07 and abs(mid_tip.x - rng_tip.x) < 0.07
            pulgar_separado = dist(thumb_tip, idx_tip) > 0.1
            return todos_semi and juntos and pulgar_separado

        # D: índice arriba, otros tocan pulgar
        def letraD():
            return idx_up and mid_dn and rng_dn and pnk_dn and dist(thumb_tip, mid_tip) < 0.1

        # E: todos los dedos doblados en gancho
        def letraE():
            gancho = idx_dn and mid_dn and rng_dn and pnk_dn
            tips_bajos = (abs(idx_tip.y - idx_mcp.y) < 0.12 and abs(mid_tip.y - mid_mcp.y) < 0.12)
            pulgar_plano = not thumb_out
            return gancho and tips_bajos and pulgar_plano

        # F: índice+pulgar hacen círculo, otros 3 arriba
        def letraF():
            circulo = dist(thumb_tip, idx_tip) < 0.07
            tres_up = mid_up and rng_up and pnk_up
            return circulo and tres_up and idx_dn

        # G: índice apunta horizontal
        def letraG():
            idx_horiz = abs(idx_tip.x - idx_mcp.x) > abs(idx_tip.y - idx_mcp.y) * 1.2
            return idx_horiz and mid_dn and rng_dn and pnk_dn

        # H: índice y medio horizontales juntos
        def letraH():
            idx_horiz = abs(idx_tip.x - idx_mcp.x) > abs(idx_tip.y - idx_mcp.y)
            mid_horiz = abs(mid_tip.x - mid_mcp.x) > abs(mid_tip.y - mid_mcp.y)
            juntos = abs(idx_tip.y - mid_tip.y) < 0.05
            return idx_horiz and mid_horiz and juntos and rng_dn and pnk_dn

        # I: solo meñique arriba
        def letraI():
            return pnk_up and idx_dn and mid_dn and rng_dn and not thumb_out

        # J: meñique arriba + traza J
        def letraJ():
            static_i = pnk_up and idx_dn and mid_dn and rng_dn
            if not static_i or len(self.landmark_history) < 10:
                return False
            pys = [h['pinky'][1] for h in self.landmark_history[-15:]]
            pxs = [h['pinky'][0] for h in self.landmark_history[-15:]]
            return (max(pys) - min(pys)) > 0.08 and (max(pxs) - min(pxs)) > 0.05

        # K: índice y medio en V, pulgar entre ellos
        def letraK():
            dos_up = idx_up and mid_up
            otros_dn = rng_dn and pnk_dn
            separados = abs(idx_tip.x - mid_tip.x) > 0.04
            pulgar_entre = (min(idx_mcp.x, mid_mcp.x) - 0.04 < thumb_tip.x < max(idx_mcp.x, mid_mcp.x) + 0.04)
            return dos_up and otros_dn and separados and pulgar_entre

        # L: índice arriba + pulgar extendido
        def letraL():
            return idx_up and mid_dn and rng_dn and pnk_dn and thumb_out

        # M: puño, 3 dedos sobre pulgar, meñique separado
        def letraM():
            todos_dn = idx_dn and mid_dn and rng_dn and pnk_dn
            tres_juntos = abs(idx_tip.x - mid_tip.x) < 0.06 and abs(mid_tip.x - rng_tip.x) < 0.06
            pnk_sep = abs(pnk_tip.x - rng_tip.x) > 0.03
            pulgar_dentro = thumb_tip.y > idx_mcp.y - 0.03
            return todos_dn and tres_juntos and pnk_sep and pulgar_dentro

        # N: puño, 2 dedos sobre pulgar
        def letraN():
            todos_dn = idx_dn and mid_dn and rng_dn and pnk_dn
            dos_juntos = abs(idx_tip.x - mid_tip.x) < 0.06
            rng_sep = abs(rng_tip.x - mid_tip.x) > 0.05
            pulgar_dentro = thumb_tip.y > idx_mcp.y - 0.03
            return todos_dn and dos_juntos and rng_sep and pulgar_dentro

        # Ñ: N + movimiento ondulante
        def letraN_tilde():
            if not letraN() or len(self.landmark_history) < 10:
                return False
            ixs = [h['index'][0] for h in self.landmark_history[-15:]]
            return (max(ixs) - min(ixs)) > 0.07

        # O: todos forman O con pulgar
        def letraO():
            semi = lambda tip, pip: abs(tip.y - pip.y) < 0.07
            todos_semi = semi(idx_tip, idx_pip) and semi(mid_tip, mid_pip) and semi(rng_tip, rng_pip)
            return todos_semi and dist(thumb_tip, idx_tip) < 0.08

        # P: índice y medio apuntan abajo
        def letraP():
            idx_abajo = idx_tip.y > idx_mcp.y + 0.04
            mid_abajo = mid_tip.y > mid_mcp.y
            return idx_abajo and mid_abajo and rng_dn and pnk_dn and abs(thumb_tip.x - thumb_ip.x) > 0.03

        # Q: índice apunta abajo, pulgar también
        def letraQ():
            idx_abajo = idx_tip.y > idx_mcp.y + 0.04
            pulgar_abajo = thumb_tip.y > thumb_mcp.y
            return idx_abajo and pulgar_abajo and mid_dn and rng_dn and pnk_dn

        # R: índice y medio cruzados arriba
        def letraR():
            dos_up = idx_up and mid_up
            cruzados = abs(idx_tip.x - mid_tip.x) < 0.035
            return dos_up and rng_dn and pnk_dn and cruzados

        # S: puño, pulgar encima de los dedos
        def letraS():
            todos_dn = idx_dn and mid_dn and rng_dn and pnk_dn
            pulgar_encima = thumb_tip.y < idx_mcp.y and abs(thumb_tip.x - idx_mcp.x) < 0.12
            return todos_dn and pulgar_encima

        # T: puño, pulgar entre índice y medio
        def letraT():
            todos_dn = idx_dn and mid_dn and rng_dn and pnk_dn
            pulgar_arriba = thumb_tip.y < idx_mcp.y
            pulgar_entre = (min(idx_mcp.x, mid_mcp.x) - 0.04 < thumb_tip.x < max(idx_mcp.x, mid_mcp.x) + 0.04)
            return todos_dn and pulgar_arriba and pulgar_entre

        # U: índice y medio juntos arriba
        def letraU():
            dos_up = idx_up and mid_up
            juntos = abs(idx_tip.x - mid_tip.x) < 0.04
            return dos_up and rng_dn and pnk_dn and juntos

        # V: índice y medio separados arriba
        def letraV():
            dos_up = idx_up and mid_up
            separados = abs(idx_tip.x - mid_tip.x) > 0.05
            return dos_up and rng_dn and pnk_dn and separados

        # W: índice, medio y anular arriba separados
        def letraW():
            tres_up = idx_up and mid_up and rng_up
            separados = abs(idx_tip.x - rng_tip.x) > 0.07
            return tres_up and pnk_dn and separados

        # X: índice en gancho
        def letraX():
            gancho = (idx_dip.y < idx_pip.y) and (idx_tip.y > idx_dip.y + 0.02)
            return gancho and mid_dn and rng_dn and pnk_dn

        # Y: meñique y pulgar extendidos
        def letraY():
            return pnk_up and idx_dn and mid_dn and rng_dn and thumb_out

        # Z: índice arriba + movimiento horizontal
        def letraZ():
            static = idx_up and mid_dn and rng_dn and pnk_dn and not thumb_out
            if not static or len(self.landmark_history) < 10:
                return False
            ixs = [h['index'][0] for h in self.landmark_history[-15:]]
            return (max(ixs) - min(ixs)) > 0.10

        # Orden: dinámicas primero, luego específicas, luego generales
        if letraJ():       return "J"
        if letraZ():       return "Z"
        if letraN_tilde(): return "Ñ"
        if letraF():       return "F"
        if letraT():       return "T"
        if letraS():       return "S"
        if letraK():       return "K"
        if letraR():       return "R"
        if letraU():       return "U"
        if letraV():       return "V"
        if letraW():       return "W"
        if letraX():       return "X"
        if letraY():       return "Y"
        if letraI():       return "I"
        if letraL():       return "L"
        if letraD():       return "D"
        if letraG():       return "G"
        if letraH():       return "H"
        if letraP():       return "P"
        if letraQ():       return "Q"
        if letraM():       return "M"
        if letraN():       return "N"
        if letraA():       return "A"
        if letraB():       return "B"
        if letraO():       return "O"
        if letraC():       return "C"
        if letraE():       return "E"
        return None

    def get_current_gesture(self):
        return self.current_gesture


# ── Estado ──
if "translator" not in st.session_state:
    st.session_state.translator = TraductorLenguajeSeñas()
if "word" not in st.session_state:
    st.session_state.word = ""
if "last_gesture" not in st.session_state:
    st.session_state.last_gesture = None
if "gesture_count" not in st.session_state:
    st.session_state.gesture_count = 0

translator = st.session_state.translator

# ── Layout ──
col1, col2 = st.columns([3, 1])

with col1:
    run = st.toggle("📷 Activar cámara")
    frame_placeholder = st.empty()

with col2:
    st.markdown("### Letra detectada")
    gesture_placeholder = st.empty()
    st.markdown("### Palabra formada")
    word_placeholder = st.empty()
    word_placeholder.markdown(f"## `{st.session_state.word}`")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⌫ Borrar letra"):
            st.session_state.word = st.session_state.word[:-1]
    with col_b:
        if st.button("🗑️ Limpiar"):
            st.session_state.word = ""

# ── Loop cámara ──
if run:
    cap = cv2.VideoCapture(0)
    CONFIRM_FRAMES = 15

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("No se puede acceder a la cámara.")
            break

        frame = translator.detect_gesture(frame)
        gesture = translator.get_current_gesture()

        if gesture == st.session_state.last_gesture:
            st.session_state.gesture_count += 1
        else:
            st.session_state.gesture_count = 0
            st.session_state.last_gesture = gesture

        if st.session_state.gesture_count == CONFIRM_FRAMES and gesture:
            st.session_state.word += gesture
            st.session_state.gesture_count = 0

        if gesture:
            gesture_placeholder.markdown(f"# {gesture}")
            cv2.putText(frame, gesture, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        else:
            gesture_placeholder.markdown("# —")

        word_placeholder.markdown(f"## `{st.session_state.word}`")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()