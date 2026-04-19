import pandas as pd

INPUT_CSV = r"C:\Android_App\features.csv"


def classify(row):
    score = 0
    reasons = []

    # Rule 1: Eye openness
    if row["avg_eye_ear"] is not None:
        if row["avg_eye_ear"] < 0.20:
            score += 40
            reasons.append("eyes mostly closed")
        elif row["avg_eye_ear"] < 0.30:
            score += 25
            reasons.append("eyes partially closed")

    # Rule 2: Head tilt
    if row["head_tilt"] is not None:
        if abs(row["head_tilt"]) > 10:
            score += 30
            reasons.append("head tilted strongly")
        elif abs(row["head_tilt"]) > 5:
            score += 15
            reasons.append("head slightly tilted")

    # Rule 3: Eye asymmetry
    if row["eye_diff"] is not None and row["eye_diff"] > 0.10:
        score += 10
        reasons.append("uneven eye openness")

    # Final decision
    if score >= 40:
        label = "Drunk"
    else:
        label = "Normal"

    confidence = min(score / 100, 1.0)

    return label, confidence, reasons


def main():
    df = pd.read_csv(INPUT_CSV)

    results = []

    print("\n===== CLASSIFICATION RESULTS =====\n")

    for _, row in df.iterrows():
        if row["face_detected"] == 0:
            print(f"{row['file_name']} → No face detected")
            continue

        label, confidence, reasons = classify(row)

        print(f"{row['file_name']} → {label} ({confidence*100:.1f}%)")
        print(f"   Reasons: {', '.join(reasons)}\n")

        results.append({
            "file_name": row["file_name"],
            "actual": row["label"],
            "predicted": label,
            "confidence": confidence
        })

    result_df = pd.DataFrame(results)

    accuracy = (result_df["actual"].str.lower() == result_df["predicted"].str.lower()).mean()

    print("===== SUMMARY =====")
    print(f"Accuracy: {accuracy*100:.2f}%")

    result_df.to_csv(r"C:\Android_App\results.csv", index=False)
    print("Saved results to results.csv")


if __name__ == "__main__":
    main()