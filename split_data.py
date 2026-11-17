# Script to load the UCI data and split it while keeping identical 13-feature patterns together.

from pathlib import Path
import numpy as np
from scipy.io import arff
from sklearn.model_selection import StratifiedGroupKFold

DATA_FILE = Path(__file__).resolve().parent / "data" / "Training Dataset.arff"

URL_FEATURES = (
    "having_IP_Address",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "HTTPS_token",
)

HTML_FEATURES = (
    "Favicon",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Iframe",
)

# Function to load the selected features and recode the original labels.
def load_data():
    records, _ = arff.loadarff(DATA_FILE)
    if set(records["Result"]) != {b"-1", b"1"}:
        raise ValueError("Expected the original UCI labels -1 and 1.")

    # URL columns come first so both models can use the same feature array.
    features = np.column_stack([
        records[name].astype(int) for name in URL_FEATURES + HTML_FEATURES
    ])

    # Phishing is the positive class so recall measures how much phishing we catch.
    labels = (records["Result"] == b"-1").astype(int)
    return features, labels

# Function to split the records while keeping complete input patterns together.
def make_split(features, labels):
    # Records with identical inputs share a group, even when their labels differ.
    _, groups = np.unique(features, axis=0, return_inverse=True)
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    folds = [
        held_out for _, held_out in splitter.split(features, labels, groups)
    ]

    # Five folds for fitting, one for threshold selection, and one for final testing.
    parts = {
        "Train": np.sort(np.concatenate(folds[2:])),
        "Validation": folds[0],
        "Test": folds[1],
    }

    all_rows = np.sort(np.concatenate(list(parts.values())))
    if not np.array_equal(all_rows, np.arange(len(labels))):
        raise ValueError("Every row must appear in exactly one partition.")

    # Row separation alone would still allow identical inputs in different partitions.
    pattern_sets = [set(groups[rows]) for rows in parts.values()]
    if sum(map(len, pattern_sets)) != len(set.union(*pattern_sets)):
        raise ValueError("A feature pattern crosses partitions.")

    return parts

def main():
    features, labels = load_data()
    parts = make_split(features, labels)

    for name, rows in parts.items():
        phishing = int(labels[rows].sum())
        print(
            f"{name}: {len(rows)} rows | {phishing} phishing | "
            f"{len(rows) - phishing} legitimate | {phishing / len(rows):.2%} phishing"
        )

    print("Checks passed: all rows assigned once, with zero shared 13-feature patterns.")

if __name__ == "__main__":
    main()