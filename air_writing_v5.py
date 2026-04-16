import cv2
import mediapipe as mp
import numpy as np
import time
import os

# =========================
# INITIAL SETUP
# =========================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================
# VARIABLES
# =========================
brush_thickness = 5
eraser_thickness = 35
draw_color = (255, 0, 255)   # Purple default
smoothening = 5

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

paused = False
eraser_mode = False
show_ui = True

last_save_time = 0
save_folder = "saved_drawings"
os.makedirs(save_folder, exist_ok=True)

# For dark background / false detection protection
hand_detected_frames = 0
min_detect_frames = 3
no_hand_frames = 0

# =========================
# BUTTONS
# =========================
buttons = {
    "purple": (20, 20, 120, 70),
    "blue":   (140, 20, 240, 70),
    "green":  (260, 20, 360, 70),
    "red":    (380, 20, 480, 70),
    "eraser": (500, 20, 620, 70),
    "clear":  (640, 20, 760, 70),
    "save":   (780, 20, 900, 70),
    "pause":  (920, 20, 1040, 70),
}

# =========================
# HELPER FUNCTIONS
# =========================
def fingers_up(hand_landmarks):
    lm = hand_landmarks.landmark
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if lm[tips_ids[0]].x < lm[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip_id in tips_ids[1:]:
        if lm[tip_id].y < lm[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def draw_buttons(frame):
    cv2.rectangle(frame, (20, 20), (120, 70), (255, 0, 255), -1)
    cv2.putText(frame, "Purple", (28, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (140, 20), (240, 70), (255, 0, 0), -1)
    cv2.putText(frame, "Blue", (162, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (260, 20), (360, 70), (0, 255, 0), -1)
    cv2.putText(frame, "Green", (275, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (380, 20), (480, 70), (0, 0, 255), -1)
    cv2.putText(frame, "Red", (408, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (500, 20), (620, 70), (60, 60, 60), -1)
    cv2.putText(frame, "Eraser", (525, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (640, 20), (760, 70), (40, 40, 160), -1)
    cv2.putText(frame, "Clear", (670, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (780, 20), (900, 70), (20, 120, 20), -1)
    cv2.putText(frame, "Save", (815, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.rectangle(frame, (920, 20), (1040, 70), (120, 80, 0), -1)
    cv2.putText(frame, "Pause", (947, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


def inside_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2


def save_canvas(canvas):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, f"air_drawing_{timestamp}.png")
    cv2.imwrite(filename, canvas)
    return filename


def is_dark_frame(frame, threshold=40):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold


# =========================
# START CAMERA
# =========================
ret, frame = cap.read()
if not ret:
    print("Camera not opening")
    cap.release()
    exit()

frame = cv2.flip(frame, 1)
h, w, c = frame.shape
canvas = np.zeros((h, w, 3), dtype=np.uint8)

print("STARTING AI AIR WRITING V5...")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if show_ui:
        draw_buttons(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    dark_scene = is_dark_frame(frame)
    hand_found = False

    if results.multi_hand_landmarks and not dark_scene:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_found = True
            hand_detected_frames += 1
            no_hand_frames = 0

            if hand_detected_frames < min_detect_frames:
                continue

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            x1, y1 = int(lm[8].x * w), int(lm[8].y * h)     # Index finger tip
            x2, y2 = int(lm[12].x * w), int(lm[12].y * h)   # Middle finger tip

            fingers = fingers_up(hand_landmarks)

            # =========================
            # SELECTION MODE
            # Index + Middle up
            # =========================
            if fingers[1] == 1 and fingers[2] == 1:
                prev_x, prev_y = 0, 0

                cv2.rectangle(frame, (x1, y1 - 20), (x2, y2 + 20), draw_color, cv2.FILLED)

                if inside_box(x1, y1, buttons["purple"]):
                    draw_color = (255, 0, 255)
                    eraser_mode = False

                elif inside_box(x1, y1, buttons["blue"]):
                    draw_color = (255, 0, 0)
                    eraser_mode = False

                elif inside_box(x1, y1, buttons["green"]):
                    draw_color = (0, 255, 0)
                    eraser_mode = False

                elif inside_box(x1, y1, buttons["red"]):
                    draw_color = (0, 0, 255)
                    eraser_mode = False

                elif inside_box(x1, y1, buttons["eraser"]):
                    eraser_mode = True

                elif inside_box(x1, y1, buttons["clear"]):
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)

                elif inside_box(x1, y1, buttons["save"]):
                    if time.time() - last_save_time > 1.5:
                        filename = save_canvas(canvas)
                        last_save_time = time.time()
                        print(f"Saved: {filename}")

                elif inside_box(x1, y1, buttons["pause"]):
                    paused = not paused
                    time.sleep(0.3)

            # =========================
            # DRAW MODE
            # Only index finger up
            # =========================
            elif fingers[1] == 1 and fingers[2] == 0 and not paused:
                pointer_color = draw_color if not eraser_mode else (200, 200, 200)
                cv2.circle(frame, (x1, y1), 10, pointer_color, cv2.FILLED)

                curr_x = prev_x + (x1 - prev_x) // smoothening if prev_x != 0 else x1
                curr_y = prev_y + (y1 - prev_y) // smoothening if prev_y != 0 else y1

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = curr_x, curr_y

                thickness = eraser_thickness if eraser_mode else brush_thickness
                color = (0, 0, 0) if eraser_mode else draw_color

                cv2.line(frame, (prev_x, prev_y), (curr_x, curr_y), color, thickness)
                cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), color, thickness)

                prev_x, prev_y = curr_x, curr_y

            else:
                prev_x, prev_y = 0, 0

    else:
        hand_detected_frames = 0
        no_hand_frames += 1
        prev_x, prev_y = 0, 0

    # =========================
    # MERGE CANVAS + FRAME
    # =========================
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # =========================
    # HUD
    # =========================
    mode_text = "PAUSED" if paused else ("ERASER" if eraser_mode else "DRAW")
    cv2.rectangle(frame, (20, h - 80), (330, h - 20), (25, 25, 25), -1)
    cv2.putText(frame, f"Mode: {mode_text}", (35, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Brush: {brush_thickness}", (360, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if dark_scene:
        cv2.putText(frame, "Low light detected", (580, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # =========================
    # SHOW WINDOW
    # =========================
    cv2.imshow("AI Air Writing System V5", frame)

    # =========================
    # KEYBOARD SHORTCUTS
    # =========================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('s'):
        filename = save_canvas(canvas)
        print(f"Saved: {filename}")
    elif key == ord('p'):
        paused = not paused
    elif key == ord('e'):
        eraser_mode = not eraser_mode
    elif key == ord('+') or key == ord('='):
        brush_thickness = min(30, brush_thickness + 1)
    elif key == ord('-'):
        brush_thickness = max(1, brush_thickness - 1)

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
