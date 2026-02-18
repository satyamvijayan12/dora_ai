import threading
import cv2
import mediapipe as mp
import time
import math
import pyttsx3

# ---------------- Utility ----------------
def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# ---------------- Reaction Engine ----------------
class ReactionEngine:
    def __init__(self, cooldown=3.0):
        self.last_expression = None
        self.last_reaction_time = 0
        self.cooldown = cooldown

    def react(self, expression):
        now = time.time()

        if expression != self.last_expression:
            self.last_expression = expression
            self.last_reaction_time = now
            return None

        if now - self.last_reaction_time < self.cooldown:
            return None

        reactions = {
            "Smile": "You look happy 🙂",
            "Curious": "You seem curious. Want to explore something?",
            "Confused": "You look unsure. Can I help clarify?",
            "Sad": "You seem a bit down. I'm here if you want to talk.",
            "Angry": "You look frustrated. Let’s slow things down.",
            "Neutral": "I'm here if you need me."
        }

        self.last_reaction_time = now
        return reactions.get(expression, None)

# ---------------- Voice Engine ----------------
class VoiceEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        voices = self.engine.getProperty("voices")
        self.engine.setProperty("voice", voices[1].id)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

# ---------------- MediaPipe Tasks ----------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv2.VideoCapture(1)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.6,
    min_face_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

reaction_engine = ReactionEngine(cooldown=3.0)
voice = VoiceEngine()
current_reaction = ""

prev_time = 0
timestamp = 0

class VoiceEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 170)
        self.lock = threading.Lock()

    def speak(self, text):
        def run():
            with self.lock:
                self.engine.say(text)
                self.engine.runAndWait()

        threading.Thread(target=run, daemon=True).start()
# ---------------- Main Loop ----------------
with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp)

        timestamp += 1

        expression = "Neutral"

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # --- Core measurements ---
            left_eye = dist(landmarks[159], landmarks[145])
            right_eye = dist(landmarks[386], landmarks[374])
            eye_open = (left_eye + right_eye) / 2

            mouth_open = dist(landmarks[13], landmarks[14])
            mouth_width = dist(landmarks[61], landmarks[291])
            smile_ratio = mouth_width / (mouth_open + 1e-6)

            # Eyebrows (distance to eye)
            left_brow = dist(landmarks[70], landmarks[159])
            right_brow = dist(landmarks[300], landmarks[386])
            brow_diff = abs(left_brow - right_brow)

            # --- Expression rules (ordered by priority) ---
            # Angry
            if eye_open < 0.018 and left_brow < 0.035 and right_brow < 0.035:
                expression = "Angry"

            # Sad
            elif mouth_open < 0.02 and smile_ratio < 1.8 and left_brow > 0.045:
                expression = "Sad"

            # Confused
            elif brow_diff > 0.015:
                expression = "Confused"

            # Curious
            elif mouth_open > 0.025 and eye_open > 0.03:
                expression = "Curious"

            # Smile
            elif smile_ratio > 2.0:
                expression = "Smile"

            # Mouth open (generic)
            elif mouth_open > 0.04:
                expression = "Mouth Open"

            # Blink
            elif eye_open < 0.015:
                expression = "Blink"

            # mp.tasks.vision.utils.draw_landmarks(
            #     image=frame,
            #     landmark_list=result.face_landmarks[0],
            #     connections=mp.tasks.vision.FACEMESH_TESSELATION
            # )

        reaction = reaction_engine.react(expression)
        if reaction:
            current_reaction = reaction
            voice.speak(reaction)

        # FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time else 0
        prev_time = curr_time

        # Overlay
        cv2.putText(frame, f"Expression: {expression}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Reaction: {current_reaction}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

        cv2.putText(frame, f"FPS: {fps}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Expression → Reaction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()