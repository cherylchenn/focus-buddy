import cv2
import time
import torch
import numpy as np
import mediapipe as mp
import winsound # for Windows only
from models.headpose_mlp import HeadPoseMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/mlp.pth"
CLASSES = ["centre", "left", "right", "up", "down"]
YELL_THRESHOLD = 60.0 # seconds

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def load_model(input_dim, num_classes):
    model = HeadPoseMLP(input_dim, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def extract_landmarks(landmarks):
    lm = np.array([[p.x, p.y, p.z] for p in landmarks.landmark])
    lm = lm - lm[1]  # normalize to nose
    return lm.flatten()

def predict(model, landmarks):
    with torch.no_grad():
        x = torch.tensor(landmarks, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        pred_idx = model(x).argmax(dim=1).item()
    return CLASSES[pred_idx]

def draw_prediction(frame, landmarks, label):
    # get bbox from landmarks
    lm_pts = np.array([[int(p.x * frame.shape[1]), int(p.y * frame.shape[0])] for p in landmarks.landmark])
    x_min, y_min = lm_pts.min(axis=0)
    x_max, y_max = lm_pts.max(axis=0)
    
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
    cv2.putText(frame, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame

def run_webcam(model):
    cap = cv2.VideoCapture(0)
    print("Webcam opened! Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            landmarks = extract_landmarks(face_landmarks)
            label = predict(model, landmarks)
            frame = draw_prediction(frame, face_landmarks, label)

            if label == "centre":
                last_centre_time = time.time()
                yelled = False
            else:
                elapsed = time.time() - last_centre_time
                if elapsed > YELL_THRESHOLD and not yelled:
                    winsound.Beep(1000, 3000) # beep at 1000Hz for 3s
                    yelled = True

        cv2.imshow("Focus Buddy", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed!")

if __name__ == "__main__":
    model = load_model(input_dim=1404, num_classes=len(CLASSES))
    run_webcam(model)