# Script to train the majority baseline and compare both logistic regression models on validation data at a 0.50 cutoff.

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from split_data import URL_FEATURES, load_data, make_split

DEFAULT_THRESHOLD = 0.50

# Function to fit preprocessing and logistic regression on the training rows.
def fit_logistic(features, labels):
    # The integer codes are categories. One-hot encoding gives each its own weight.
    # Fitting the pipeline here learns the categories from training data only.
    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=1000),
    )
    return model.fit(features, labels)

# Function to calculate the report metrics from the confusion-matrix counts.
def print_metrics(name, actual, predicted):
    # Fixed label order gives TN, FP, FN, and TP when the matrix is flattened.
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / len(actual)
    recall = tp / (tp + fn)
    false_positive_rate = fp / (fp + tn)

    # A model that never flags a page has no predictions to use for precision.
    precision = f"{tp / (tp + fp):.2%}" if tp + fp else "undefined"

    print(f"\n{name}")
    print(f"TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"Accuracy={accuracy:.2%} | Precision={precision}")
    print(f"Recall={recall:.2%} | False-positive rate={false_positive_rate:.2%}")

def main():
    features, labels = load_data()
    parts = make_split(features, labels)
    train = parts["Train"]
    validation = parts["Validation"]

    x_train, y_train = features[train], labels[train]
    x_val, y_val = features[validation], labels[validation]

    # The baseline shows what the training class counts achieve on their own.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    print_metrics("Majority baseline", y_val, baseline.predict(x_val))

    url_count = len(URL_FEATURES)
    feature_sets = [
        ("URL only", slice(0, url_count)),
        ("URL + HTML", slice(None)),
    ]

    # The same records and cutoff keep the comparison focused on the added fields.
    for name, columns in feature_sets:
        model = fit_logistic(x_train[:, columns], y_train)
        scores = model.predict_proba(x_val[:, columns])[:, 1]
        print_metrics(
            f"{name} at {DEFAULT_THRESHOLD:.2f}",
            y_val,
            scores >= DEFAULT_THRESHOLD,
        )

if __name__ == "__main__":
    main()    