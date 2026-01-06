import os
import cv2
import numpy as np
import mediapipe as mp

FACES_DIR = "BIWI"
OUTPUT_FILE = "biwi_landmarks.npz"

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

def rotation_matrix_to_euler(R):
    # convert 3x3 rotation matrix to yaw, pitch, roll in degrees
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    return np.degrees([y, x, z]) # yaw (left/right), pitch (up/down), roll (tilt)

def pose_to_class(yaw, pitch): # maps yaw / pitch to multi-class label
    if abs(yaw) < 15 and abs(pitch) < 15:
        return 0 # forward
    elif yaw <= -15:
        return 1 # left
    elif yaw >= 15:
        return 2 # right
    elif pitch <= -15:
        return 3 # up
    elif pitch >= 15:
        return 4 # down
    else:
        return None

X, y = [], []

for seq in sorted(os.listdir(FACES_DIR)):
    seq_path = os.path.join(FACES_DIR, seq)
    if not os.path.isdir(seq_path):
        continue
    
    print(f"Loading sequence: {seq}")

    # loop over all rgb frames
    for file in sorted(os.listdir(seq_path)):
        if not file.endswith("rgb.png"):
            continue

        img_path = os.path.join(seq_path, file)
        pose_path = img_path.replace("rgb.png", "pose.txt")
        if not os.path.exists(pose_path):
            continue
        
        # read rotation matrix from pose.txt file
        lines = [line.strip() for line in open(pose_path).readlines() if line.strip()]
        try:
            R = np.array([[float(x) for x in lines[i].split()] for i in range(3)])
            if R.shape != (3,3):
                print(f"Skipping {pose_path}, rotation matrix not 3x3")
                continue
        except:
            print(f"Skipping {pose_path}, failed to parse rotation matrix")
            continue
        
        yaw, pitch, roll = rotation_matrix_to_euler(R)
        label = pose_to_class(yaw, pitch)
        if label is None:
            continue

        # read image & extract landmarks
        img = cv2.imread(img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.multi_face_landmarks[0].landmark])
        landmarks = landmarks - landmarks[1] # normalize to nose
        X.append(landmarks.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)
np.savez(OUTPUT_FILE, X=X, y=y)
print(f"Saved {len(X)} samples to {OUTPUT_FILE}")