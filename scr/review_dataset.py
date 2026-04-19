import os
import cv2
import shutil

INPUT_BASE = r"C:\Android_app\processed_dataset"
OUTPUT_BASE = r"C:\Android_app\filtered_dataset"

CLASSES = ["drunk", "normal"]
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def ensure_output_folders():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_BASE, cls), exist_ok=True)


def review_images():
    ensure_output_folders()

    for cls in CLASSES:
        input_folder = os.path.join(INPUT_BASE, cls)
        output_folder = os.path.join(OUTPUT_BASE, cls)

        if not os.path.exists(input_folder):
            print(f"[WARNING] Folder not found: {input_folder}")
            continue

        image_files = sorted([
            f for f in os.listdir(input_folder)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ])

        print(f"\nReviewing class: {cls}")
        print(f"Found {len(image_files)} images")

        for filename in image_files:
            input_path = os.path.join(input_folder, filename)

            # Skip if already reviewed and copied
            output_path = os.path.join(output_folder, filename)
            if os.path.exists(output_path):
                print(f"[SKIP] Already kept: {filename}")
                continue

            image = cv2.imread(input_path)
            if image is None:
                print(f"[FAILED] Could not read: {filename}")
                continue

            display = image.copy()
            display = cv2.resize(display, (700, 700))

            info_text1 = f"Class: {cls} | File: {filename}"
            info_text2 = "Press K = Keep | D = Discard | Q = Quit"

            cv2.putText(display, info_text1, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, info_text2, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Dataset Review", display)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('k'):
                shutil.copy2(input_path, output_path)
                print(f"[KEEP] {filename}")
            elif key == ord('d'):
                print(f"[DISCARD] {filename}")
            elif key == ord('q'):
                print("\n[INFO] Review stopped by user.")
                cv2.destroyAllWindows()
                return
            else:
                print(f"[SKIP] Invalid key for {filename}")

    cv2.destroyAllWindows()
    print("\n[INFO] Review completed.")


if __name__ == "__main__":
    review_images()
