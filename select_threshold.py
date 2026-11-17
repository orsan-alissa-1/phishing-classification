# Script to select the cutoff that catches the most phishing records within a 5% validation false-positive limit.

import csv
from sklearn.metrics import confusion_matrix
from split_data import load_data, make_split
from train_models import DEFAULT_THRESHOLD, fit_logistic

MAX_FALSE_POSITIVE_PERCENT = 5

# Function to compare decision thresholds on validation predictions.
def main():
    features, labels = load_data()
    parts = make_split(features, labels)
    train = parts["Train"]
    validation = parts["Validation"]
    actual = labels[validation]

    # One fitted model supplies the same scores for every cutoff in the sweep.
    model = fit_logistic(features[train], labels[train])
    scores = model.predict_proba(features[validation])[:, 1]

    legitimate = int((actual == 0).sum())
    phishing = int((actual == 1).sum())

    # The 5% validation limit rounds down to a whole number of false alarms.
    max_false_positives = legitimate * MAX_FALSE_POSITIVE_PERCENT // 100

    results = []
    for percent in range(1, 100):
        threshold = percent / 100
        tn, fp, fn, tp = confusion_matrix(
            actual, scores >= threshold, labels=[0, 1]
        ).ravel()
        results.append({
            "threshold": threshold,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "recall": tp / phishing,
            "false_positive_rate": fp / legitimate,
        })

    with open("validation_thresholds.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    default = next(row for row in results if row["threshold"] == DEFAULT_THRESHOLD)
    eligible = [row for row in results if row["false_positive"] <= max_false_positives]
    if not eligible:
        raise ValueError("No cutoff meets the validation false-positive limit.")

    # The most detections win. Ties favour fewer false alarms, then a higher cutoff.
    selected = max(eligible, key=lambda row: (
        row["true_positive"], -row["false_positive"], row["threshold"]
    ))

    print(f"Validation false-positive limit: {max_false_positives} of {legitimate}")
    for name, row in [("Default", default), ("Selected", selected)]:
        print(
            f"{name}: threshold={row['threshold']:.2f} | "
            f"TP={row['true_positive']} | FP={row['false_positive']} | "
            f"Recall={row['recall']:.2%} | FPR={row['false_positive_rate']:.2%}"
        )

if __name__ == "__main__":
    main()