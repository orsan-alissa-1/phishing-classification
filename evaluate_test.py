# Script to evaluate the baseline and both models on held-out data using cutoffs fixed before testing.

from sklearn.dummy import DummyClassifier
from split_data import URL_FEATURES, load_data, make_split
from train_models import DEFAULT_THRESHOLD, fit_logistic, print_metrics

# Cutoff selected on validation and fixed before looking at test predictions.
VALIDATION_THRESHOLD = 0.44

# Function to evaluate both feature sets and the fixed cutoffs on held-out rows.
def main():
    features, labels = load_data()
    parts = make_split(features, labels)
    train, test = parts["Train"], parts["Test"]

    x_train, y_train = features[train], labels[train]
    x_test, y_test = features[test], labels[test]

    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    print_metrics("Test | Majority baseline", y_test, baseline.predict(x_test))

    # The shared cutoff lets us compare the feature sets on the same test rows.
    url_count = len(URL_FEATURES)
    url_model = fit_logistic(x_train[:, :url_count], y_train)
    url_scores = url_model.predict_proba(x_test[:, :url_count])[:, 1]
    print_metrics(
        f"Test | URL only at {DEFAULT_THRESHOLD:.2f}",
        y_test,
        url_scores >= DEFAULT_THRESHOLD,
    )

    # The same scores are used twice so the cutoff is the only change here.
    html_model = fit_logistic(x_train, y_train)
    html_scores = html_model.predict_proba(x_test)[:, 1]
    for threshold in (DEFAULT_THRESHOLD, VALIDATION_THRESHOLD):
        print_metrics(
            f"Test | URL + HTML at {threshold:.2f}",
            y_test,
            html_scores >= threshold,
        )

if __name__ == "__main__":
    main()