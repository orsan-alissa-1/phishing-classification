# Phishing Website Classification

An independent experiment comparing URL-only features with URL and HTML features for phishing website classification. Both models use one-hot encoding and logistic regression, with the same training and evaluation records.

The study examines how adding HTML features changes false alarms and missed phishing records. A second experiment selects a decision threshold on validation data and checks whether its false-positive limit holds on a separate test partition.

## Method

- Use the UCI Phishing Websites dataset: 11,055 records with precomputed categorical features.
- Select seven URL features and six HTML-derived features.
- Group records by their complete 13-feature pattern before splitting, so identical complete inputs cannot cross partitions.
- Use `StratifiedGroupKFold` with seven folds and `random_state=42`: five folds for training, one for validation, and one for testing.
- Fit a majority-class baseline and two logistic regression models. The one-hot encoder is fitted on training data only.
- Compare both feature sets at a threshold of 0.50.
- Sweep thresholds from 0.01 to 0.99 on validation data. Select the highest-recall threshold satisfying a 5% validation false-positive limit, with the tie-breaking rule defined in `select_threshold.py`.
- Apply the selected threshold of 0.44 to the held-out test set without refitting on validation data.
- Compare paired errors and reconstruct one prediction from the model weights, intercept, and sigmoid.

| Partition | Records | Phishing | Legitimate |
| --- | ---: | ---: | ---: |
| Training | 7,903 | 3,502 | 4,401 |
| Validation | 1,578 | 699 | 879 |
| Test | 1,574 | 697 | 877 |

## Held-out results

Phishing is the positive class. False-positive rate measures the fraction of legitimate records incorrectly flagged.

| Model | Threshold | Accuracy | Phishing recall | False-positive rate |
| --- | ---: | ---: | ---: | ---: |
| Majority baseline | — | 55.72% | 0.00% | 0.00% |
| URL only | 0.50 | 65.88% | 84.51% | 48.92% |
| URL + HTML | 0.50 | 86.09% | 80.20% | 9.24% |
| URL + HTML | 0.44 | 86.21% | 81.64% | 10.15% |

At the shared 0.50 threshold, HTML features reduced false positives from 429 to 81, but phishing detections fell from 589 to 559. The improvement in accuracy therefore came with lower phishing recall.

The selected threshold of 0.44 met the validation false-positive limit with a rate of 4.78%. Its test rate reached 10.15%. The validation constraint did not carry over to this test partition.

The [full report](Phishing_Classification_Report.pdf) covers the feature audit, mathematical example, threshold selection, paired error analysis, and limitations. The report and experiment source are preserved unchanged.

## Reproduce the experiment

The original environment used Python 3.12 and the dependency versions in `requirements.txt`. All six scripts were also rerun during repository preparation using Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and scikit-learn 1.8.0. The test and paired-error outputs matched the supplied result files.

From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\split_data.py
.\.venv\Scripts\python.exe .\train_models.py
.\.venv\Scripts\python.exe .\check_probability.py
.\.venv\Scripts\python.exe .\select_threshold.py
.\.venv\Scripts\python.exe .\evaluate_test.py
.\.venv\Scripts\python.exe .\compare_errors.py
```

The dataset is included at `data/Training Dataset.arff`. Its SHA-256 is:

```text
3bcf93b1612306537ee2737751f1bac185b2a77886ab376644c3bf0c7f7cb3e6
```

Expected checks: 7,903/1,578/1,574 rows in training/validation/test; zero shared complete 13-feature patterns; a selected threshold of 0.44; and 559 true positives with 81 false positives for the combined test model at 0.50.

Each script fits its own model when needed; no saved model file is required. `select_threshold.py` writes `validation_thresholds.csv` in the current directory. The test and error scripts print to the terminal; saved outputs are included as supporting evidence.

## Files

| File | Purpose |
| --- | --- |
| `split_data.py` | Loads selected features, recodes labels, and checks the grouped split |
| `train_models.py` | Fits the baseline and models and reports validation results |
| `check_probability.py` | Verifies one probability using the fitted coefficients |
| `select_threshold.py` | Sweeps validation thresholds and selects the cutoff |
| `evaluate_test.py` | Evaluates the fixed models and thresholds on test data |
| `compare_errors.py` | Counts paired prediction outcomes |
| `validation_thresholds.csv` | Saved validation sweep |
| `test_results.txt`, `error_comparison.txt` | Saved test outputs, normalised to UTF-8 for readability |
| `Phishing_Classification_Report.pdf` | Full project report |
| `data/` | Dataset, original feature guide, and attribution |

## Limitations

- This is a historical dataset experiment using precomputed features. It does not collect live URLs, retrieve HTML, or implement a deployable phishing detector.
- Grouping prevents overlap of complete 13-feature patterns. Shorter URL-only patterns can recur across partitions, and domain-level or chronological separation is not established.
- Repeated rows and conflicting labels are retained. Common patterns have more weight in the per-record metrics.
- Results come from one grouped split. They do not establish performance across other splits, new domains, or current phishing campaigns.
- The 5% limit was a validation selection rule, not a guaranteed operating false-positive rate.

## Dataset attribution

Mohammad, R. & McCluskey, L. (2012). *Phishing Websites* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C51W2X](https://doi.org/10.24432/C51W2X).

UCI distributes the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [dataset attribution](data/ATTRIBUTION.md) for source and reuse details. The underlying dataset and feature definitions are the dataset authors' work.

## Author

Orsan Al Issa

