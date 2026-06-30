# Sales Marketing Customer Churn Prediction

This repository trains several classification models, selects the best model with stratified cross-validation, tunes its probability threshold, and serves predictions through Streamlit.

## Why the original notebook needs one more step

The original notebook currently identifies Random Forest as the best tested candidate with an F1-score of 0.5321. However, it applies SMOTE before `GridSearchCV` and does not preserve the full preprocessing pipeline. This project corrects those issues before deployment.

## Project structure

```text
sales_marketing_churn_app/
├── app.py
├── features.py
├── train_model.py
├── requirements.txt
├── README.md
├── NOTEBOOK_REVIEW.md
├── data/
│   └── Sales Marketing.csv       # local only, excluded from Git
├── model/
│   ├── churn_model.joblib        # generated after training
│   ├── model_comparison.csv      # generated after training
│   ├── test_metrics.json         # generated after training
│   └── feature_importance.csv    # generated when supported
└── notebooks/
    └── Projek sales marketing.ipynb
```

## 1. Prepare the project

Copy your dataset into:

```text
data/Sales Marketing.csv
```

The filename must match exactly unless you pass another path to `train_model.py`.

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train and select the best model

```bash
python train_model.py --data "data/Sales Marketing.csv"
```

The script performs these steps:

1. Stratified train and test split.
2. Leakage-safe feature engineering.
3. Missing-value imputation and categorical encoding.
4. Oversampling inside each cross-validation training fold.
5. Comparison of Logistic Regression, Random Forest, and Extra Trees.
6. Hyperparameter search based on mean cross-validation F1.
7. Probability-threshold tuning from out-of-fold predictions.
8. Final evaluation on the untouched test set.
9. Saving the complete model artifact to `model/churn_model.joblib`.

For a faster initial test:

```bash
python train_model.py --data "data/Sales Marketing.csv" --cv 3 --n-iter 5
```

## 4. Run Streamlit locally

```bash
streamlit run app.py
```

The application supports:

- One-customer prediction
- CSV batch prediction
- Prediction download
- Model and evaluation details

## 5. Upload to GitHub

Create an empty GitHub repository, for example `sales-marketing-churn`, then run:

```bash
git init
git add .
git commit -m "Build customer churn prediction app"
git branch -M main
git remote add origin https://github.com/USERNAME/sales-marketing-churn.git
git push -u origin main
```

Replace `USERNAME` with your GitHub username.

Before pushing, confirm that this generated file exists and is staged:

```text
model/churn_model.joblib
```

Do not upload confidential customer data. The `.gitignore` file excludes CSV files in `data/`.

## 6. Deploy to Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Click **Create app**.
3. Select the GitHub repository and the `main` branch.
4. Set the entrypoint file to `app.py`.
5. Deploy the app.

Streamlit installs the packages listed in `requirements.txt`. Keep that file in the repository root next to `app.py`.

## Model-selection rule

The script selects the model with the highest mean cross-validation F1-score. This is more appropriate than accuracy for this dataset because the churn class is substantially smaller than the stay class.

A real retention program may prefer recall over F1 when missing a customer who will churn costs more than contacting a customer who would stay. In that case, change the selection rule and threshold objective after defining the business cost.
