import cv2
import mediapipe as mp
import time

LOOK_AWAY_DURATION = 5 * 60
looking_away_start = None

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# landmark indices
left_eye_outer = 33
right_eye_outer = 263
nose_tip = 1
chin = 152
forehead = 10 # approx top of forehead

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # horizontal angle
            # get x-coordinates of key points
            x_left = landmarks[left_eye_outer].x
            x_right = landmarks[right_eye_outer].x
            x_nose = landmarks[nose_tip].x
            # compute nose position relative to eyes
            h_ratio = (x_nose - x_left) / (x_right - x_left)

            # vertical angle
            # get y-coordinates of key points
            y_chin = landmarks[chin].y
            y_forehead = landmarks[forehead].y
            y_nose = landmarks[nose_tip].y
            # compute nose position relative to chin & forehead
            v_ratio = (y_nose - y_forehead) / (y_chin - y_forehead)

            # ratio ~0.5 → facing forward, <0.35 or >0.65 → looking away
            if h_ratio < 0.35 or h_ratio > 0.65 or v_ratio < 0.35 or v_ratio > 0.65:
                if looking_away_start is None:
                    looking_away_start = time.time()
                else:
                    elapsed = time.time() - looking_away_start
                    if elapsed > LOOK_AWAY_DURATION:
                        print(f"⚠️ You've been looking away for {LOOK_AWAY_DURATION} seconds!")
            else:
                looking_away_start = None

        cv2.imshow("Face Angle Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()