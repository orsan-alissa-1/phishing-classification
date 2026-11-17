# Script to count which test records each model got right or wrong at the shared 0.50 cutoff.

from split_data import URL_FEATURES, load_data, make_split
from train_models import DEFAULT_THRESHOLD, fit_logistic

# Function to count which errors changed when the HTML features were added.
def main():
    features, labels = load_data()
    parts = make_split(features, labels)
    train, test = parts["Train"], parts["Test"]

    x_train, y_train = features[train], labels[train]
    x_test, actual = features[test], labels[test]
    url_count = len(URL_FEATURES)

    url_model = fit_logistic(x_train[:, :url_count], y_train)
    html_model = fit_logistic(x_train, y_train)

    url_scores = url_model.predict_proba(x_test[:, :url_count])[:, 1]
    html_scores = html_model.predict_proba(x_test)[:, 1]
    url_predictions = url_scores >= DEFAULT_THRESHOLD
    html_predictions = html_scores >= DEFAULT_THRESHOLD

    # Each pair of values refers to the same record in the two models.
    url_correct = url_predictions == actual
    html_correct = html_predictions == actual

    # A changed prediction means something different for each actual class.
    for label, name in [(0, "Legitimate"), (1, "Phishing")]:
        class_mask = actual == label
        url = url_correct[class_mask]
        html = html_correct[class_mask]

        print(f"\n{name} ({len(url)} records)")
        print(f"Both correct: {(url & html).sum()}")
        print(f"URL only correct: {(url & ~html).sum()}")
        print(f"URL + HTML only correct: {(~url & html).sum()}")
        print(f"Both wrong: {(~url & ~html).sum()}")

if __name__ == "__main__":
    main()