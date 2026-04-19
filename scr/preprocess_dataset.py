from PIL import Image
import os

# Input folders
INPUT_BASE = r"C:\Android_app\DATASET"
OUTPUT_BASE = r"C:\Android_app\processed_dataset"

CLASSES = {
    "Drunk": "drunk",
    "Normal": "normal"
}

TARGET_SIZE = (512, 512)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def process_image(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img = img.resize(TARGET_SIZE)
            img.save(output_path, quality=95)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    total_processed = 0
    total_failed = 0

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for input_class, output_class in CLASSES.items():
        input_folder = os.path.join(INPUT_BASE, input_class)
        output_folder = os.path.join(OUTPUT_BASE, output_class)

        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print(f"[WARNING] Input folder not found: {input_folder}")
            continue

        files = os.listdir(input_folder)
        image_files = [f for f in files if f.lower().endswith(SUPPORTED_EXTENSIONS)]

        print(f"\nProcessing class: {input_class}")
        print(f"Found {len(image_files)} images")

        for idx, filename in enumerate(image_files, start=1):
            input_path = os.path.join(input_folder, filename)
            new_filename = f"{output_class}_{idx:03d}.jpg"
            output_path = os.path.join(output_folder, new_filename)

            success, error = process_image(input_path, output_path)

            if success:
                total_processed += 1
                print(f"[OK] {filename} -> {new_filename}")
            else:
                total_failed += 1
                print(f"[FAILED] {filename} -> {error}")

    print("\n===== SUMMARY =====")
    print(f"Processed: {total_processed}")
    print(f"Failed: {total_failed}")
    print(f"Output folder: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()