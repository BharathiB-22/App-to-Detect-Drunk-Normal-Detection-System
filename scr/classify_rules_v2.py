import pandas as pd

INPUT_CSV = r"C:\Android_App\features.csv"


def classify(row):
    score = 0
    reasons = []

    avg_eye = row["avg_eye_ear"]
    head_tilt = abs(row["head_tilt"]) if pd.notna(row["head_tilt"]) else 0
    eye_diff = row["eye_diff"] if pd.notna(row["eye_diff"]) else 0

    # Eye openness rules
    if pd.notna(avg_eye):
        if avg_eye < 0.18:
            score += 45
            reasons.append("eyes mostly closed")
        elif avg_eye < 0.26:
            score += 20
            reasons.append("eyes partially closed")

    # Head tilt rules
    if head_tilt > 12:
        score += 20
        reasons.append("head tilted strongly")
    elif head_tilt > 7:
        score += 10
        reasons.append("head slightly tilted")

    # Eye asymmetry
    if eye_diff > 0.12:
        score += 10
        reasons.append("uneven eye openness")

    # Strong combo bonus
    if pd.notna(avg_eye) and avg_eye < 0.26 and head_tilt > 7:
        score += 15
        reasons.append("combined drowsy + tilt pattern")

    # Final label
    if score >= 40:
        label = "Drunk"
    else:
        label = "Normal"

    confidence = min(score / 100, 1.0)
    return label, confidence, reasons


def main():
    df = pd.read_csv(INPUT_CSV)
    results = []

    print("\n===== CLASSIFICATION RESULTS V2 =====\n")

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

    result_df.to_csv(r"C:\Android_App\results_v2.csv", index=False)
    print("Saved results to results_v2.csv")


if __name__ == "__main__":
    main()