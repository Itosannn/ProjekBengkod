# Review of the original notebook

## Current result

The notebook reports Random Forest as the strongest tested model:

- Accuracy: 0.8447
- Recall for churn: 0.5968
- F1-score for churn: 0.5321

This makes Random Forest the current candidate, not yet a fully validated final model.

## Problems that must be corrected

1. `X_train_processed`, `X_test_processed`, `y_train`, and `y_test` are used but their creation cells are missing.
2. Missing-value treatment and outlier treatment modify the full dataset before a valid train and test pipeline is shown.
3. SMOTE is executed before `GridSearchCV`. Synthetic observations can cross validation-fold boundaries and cause leakage.
4. The notebook drops all categorical and date variables during the baseline model. This discards useful information.
5. `customer_id` is used as a numeric predictor in the baseline. It is an identifier and should not drive churn predictions.
6. Accuracy alone is misleading because only about 15.32 percent of customers churn.
7. The Random Forest grid returns the default parameters, so the search has not produced a meaningful improvement.
8. The decision threshold remains 0.50 even though threshold selection can improve F1 or recall.

## Corrections in this project

- Preprocessing, oversampling, and modeling run inside one pipeline.
- Oversampling occurs separately inside each training fold.
- Train and test splitting uses stratification.
- Model selection uses mean cross-validation F1.
- Recall, precision, ROC-AUC, and PR-AUC remain visible.
- The probability threshold is selected from out-of-fold training predictions.
- The test set is used only for final evaluation.
- Dates become tenure and purchase-recency features.
- The final fitted pipeline and threshold are saved together for Streamlit.
