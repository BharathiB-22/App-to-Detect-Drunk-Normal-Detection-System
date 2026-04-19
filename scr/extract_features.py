import os
import cv2
import math
import numpy as np
import pandas as pd
import mediapipe as mp

INPUT_BASE = r"C:\Android_app\filtered_dataset"
OUTPUT_CSV = r"C:\Android_app\features.csv"

CLASSES = ["drunk", "normal"]
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_FACE_SIDE = 234
RIGHT_FACE_SIDE = 454
NOSE_TIP = 1


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = []
    for idx in eye_indices:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        pts.append((x, y))

    # EAR formula
    vertical_1 = euclidean(pts[1], pts[5])
    vertical_2 = euclidean(pts[2], pts[4])
    horizontal = euclidean(pts[0], pts[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return round(float(ear), 4)


def get_head_tilt(landmarks, w, h):
    left_point = (
        int(landmarks[LEFT_FACE_SIDE].x * w),
        int(landmarks[LEFT_FACE_SIDE].y * h)
    )
    right_point = (
        int(landmarks[RIGHT_FACE_SIDE].x * w),
        int(landmarks[RIGHT_FACE_SIDE].y * h)
    )

    dx = right_point[0] - left_point[0]
    dy = right_point[1] - left_point[1]

    angle = math.degrees(math.atan2(dy, dx))
    return round(float(angle), 2)


def estimate_smile_proxy(left_ear, right_ear):
    """
    Placeholder simple proxy.
    Since Face Mesh alone doesn't directly give smile probability,
    we keep this as a dummy feature for now.
    """
    avg_eye = (left_ear + right_ear) / 2.0
    smile_proxy = max(0.0, 1.0 - avg_eye)
    return round(float(smile_proxy), 4)


def extract_features_from_image(image_path, label):
    image = cv2.imread(image_path)

    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {
            "file_name": os.path.basename(image_path),
            "label": label,
            "face_detected": 0,
            "left_eye_ear": None,
            "right_eye_ear": None,
            "avg_eye_ear": None,
            "eye_diff": None,
            "head_tilt": None,
            "smile_proxy": None
        }

    face_landmarks = results.multi_face_landmarks[0].landmark

    left_ear = get_eye_aspect_ratio(face_landmarks, LEFT_EYE, w, h)
    right_ear = get_eye_aspect_ratio(face_landmarks, RIGHT_EYE, w, h)
    avg_ear = round((left_ear + right_ear) / 2.0, 4)
    eye_diff = round(abs(left_ear - right_ear), 4)
    head_tilt = get_head_tilt(face_landmarks, w, h)
    smile_proxy = estimate_smile_proxy(left_ear, right_ear)

    return {
        "file_name": os.path.basename(image_path),
        "label": label,
        "face_detected": 1,
        "left_eye_ear": left_ear,
        "right_eye_ear": right_ear,
        "avg_eye_ear": avg_ear,
        "eye_diff": eye_diff,
        "head_tilt": head_tilt,
        "smile_proxy": smile_proxy
    }


def main():
    rows = []

    for cls in CLASSES:
        folder = os.path.join(INPUT_BASE, cls)

        if not os.path.exists(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ])

        print(f"\nProcessing class: {cls}")
        print(f"Found {len(files)} images")

        for file_name in files:
            image_path = os.path.join(folder, file_name)
            result = extract_features_from_image(image_path, cls)

            if result is not None:
                rows.append(result)
                print(f"[OK] {file_name} -> face_detected={result['face_detected']} avg_eye_ear={result['avg_eye_ear']} head_tilt={result['head_tilt']}")
            else:
                print(f"[FAILED] {file_name}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n===== DONE =====")
    print(f"Saved features to: {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()