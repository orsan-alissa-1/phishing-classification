# Script to calculate one validation prediction from the fitted weights and check it against predict_proba.

import numpy as np
from split_data import HTML_FEATURES, URL_FEATURES, load_data, make_split
from train_models import fit_logistic

# Function to reconstruct one validation prediction from the fitted weights.
def main():
    features, labels = load_data()
    parts = make_split(features, labels)
    train = parts["Train"]

    # The first validation record makes the worked example repeatable.
    validation_row = parts["Validation"][:1]
    row = features[validation_row]

    model = fit_logistic(features[train], labels[train])
    encoder = model.named_steps["onehotencoder"]
    classifier = model.named_steps["logisticregression"]

    encoded = encoder.transform(row).toarray()[0]
    weights = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    feature_names = encoder.get_feature_names_out(URL_FEATURES + HTML_FEATURES)

    # Active categories contribute their weight to the dot product.
    weighted_sum = float(encoded @ weights)
    score = weighted_sum + intercept

    # The sigmoid converts the linear score into an estimated phishing probability.
    manual_probability = 1 / (1 + np.exp(-score))
    library_probability = model.predict_proba(row)[0, 1]

    print("Active categories and their contributions:")
    for index in np.flatnonzero(encoded):
        print(f"  {feature_names[index]}: {weights[index]:+.8f}")

    print(f"Weighted sum:        {weighted_sum:+.8f}")
    print(f"Intercept:           {intercept:+.8f}")
    print(f"Score:               {score:+.8f}")
    print(f"Manual probability:  {manual_probability:.10f}")
    print(f"Library probability: {library_probability:.10f}")

    if not np.isclose(manual_probability, library_probability, rtol=0, atol=1e-10):
        raise ValueError("The manual calculation differs from predict_proba.")

    print("Check passed: the manual calculation matches predict_proba.")

if __name__ == "__main__":
    main()