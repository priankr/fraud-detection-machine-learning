# Credit Card Fraud Detection with Machine Learning

Machine learning analysis of 284,807 credit card transactions to detect the 0.17% that are fraudulent. This project compares supervised classification (Logistic Regression, Random Forest, XGBoost) and unsupervised anomaly detection (Isolation Forest) on highly imbalanced data, using SMOTE oversampling and precision-recall focused evaluation.

> This is an updated 2025 analysis. An [earlier version](#previous-analysis-2021) using a different methodology is included for reference.

**[View the Interactive Dashboard](https://priankr.github.io/fraud-detection-machine-learning/)**

## Dataset

The dataset was published by the [Machine Learning Group at ULB](http://mlg.ulb.ac.be) (Universit&eacute; Libre de Bruxelles) and is available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

> **Note:** The dataset (~150MB) is not included in this repository. Download it directly from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- **284,807 transactions** made by European cardholders over two days in September 2013
- **492 fraudulent** (0.173%) — extreme class imbalance
- **28 PCA-transformed features** (`V1`–`V28`) anonymized for privacy
- **2 original features**: `Time` (seconds since first transaction) and `Amount`
- **Target**: `Class` (1 = fraud, 0 = legitimate)

## Approach

### 1. Exploratory Analysis

Before building models, we explored how fraudulent transactions differ from legitimate ones:

- **Amount**: Fraudulent transactions tend to be smaller (median $9.25 vs. $22.00 for legitimate). Fraudsters often test with small amounts to check if a stolen card works.
- **Timing**: The fraud rate spikes during off-peak hours, peaking at 2 AM (1.7%) with elevated rates through 3–5 AM — suggesting fraudsters prefer hours with reduced monitoring.
- **Feature correlations**: V17, V14, V12, and V10 have the strongest correlations with fraud, though no single feature reliably indicates fraud on its own.

### 2. Handling Class Imbalance with SMOTE

A model that labels every transaction as legitimate achieves 99.8% accuracy while catching zero fraud. To address this, we used **SMOTE** (Synthetic Minority Oversampling Technique) to generate synthetic fraud examples in the training data, balancing the classes for model training while leaving the test set untouched to reflect real-world conditions.

### 3. Feature Engineering and Preprocessing

- Converted raw `Time` (seconds) to `Hour` of day to capture daily patterns
- Applied `StandardScaler` to `Amount` and `Hour` (fitted only on training data to prevent leakage)
- Stratified 80/20 train-test split preserving the fraud ratio in both sets
- `random_state=42` across all operations for full reproducibility

### 4. Models

| Model | Type | How It Works |
|---|---|---|
| **Logistic Regression** | Supervised | Linear decision boundary — fast, interpretable baseline |
| **Random Forest** | Supervised | Ensemble of decision trees with majority voting — handles non-linear patterns |
| **XGBoost** | Supervised | Sequential gradient boosting — state-of-the-art for tabular data |
| **Isolation Forest** | Unsupervised | Learns "normal" patterns and flags anomalies — no fraud labels needed |

## Results

Evaluated using **AUPRC** (Area Under the Precision-Recall Curve) as the primary metric, which is more informative than accuracy or ROC-AUC for imbalanced data.

| Model | ROC-AUC | AUPRC | Fraud Caught | False Alarms | Precision | Recall |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.9779** | **0.8553** | 79/98 (80.6%) | 14 | 84.95% | 80.61% |
| XGBoost | 0.9761 | 0.8477 | 86/98 (87.8%) | 142 | 37.72% | 87.76% |
| Logistic Regression | 0.9706 | 0.7281 | 90/98 (91.8%) | 1,534 | 5.54% | 91.84% |
| Isolation Forest | N/A | N/A | 34/98 (34.7%) | 70 | 32.69% | 34.69% |

**Random Forest** achieved the best precision-recall balance: when it flags a transaction as fraud, it is correct 85% of the time, with only 14 false alarms. Threshold optimization at 0.72 further improves this to 95% precision with 77.6% recall.

### Key Takeaways

- **Class imbalance demands careful handling.** SMOTE oversampling on training data allows models to learn fraud patterns without discarding 99.8% of the data.
- **Accuracy is misleading.** AUPRC and precision-recall analysis provide an honest picture of performance on imbalanced data.
- **Ensemble methods excel.** Random Forest and XGBoost outperform Logistic Regression by capturing non-linear patterns in the PCA features.
- **Anomaly detection works without labels.** Isolation Forest catches ~35% of fraud without any labeled examples — valuable when labeled data is scarce.
- **Threshold tuning matters.** The default 0.5 threshold is suboptimal. At 0.72, Random Forest achieves 95% precision with 77.6% recall.
- **A small number of features drive the signal.** V14, V10, V4, V17, and V12 consistently emerge as most important across models.

## Project Structure

```
├── credit-card-fraud-detection-machine-learning-2025.ipynb   # Main analysis notebook
├── credit-card-fraud-detection-machine-learning-2021.ipynb   # Previous analysis (reference)
├── docs/
│   └── index.html                                            # Interactive dashboard (GitHub Pages)
└── README.md
```

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- imbalanced-learn (for SMOTE)
- xgboost

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## Previous Analysis (2021)

The earlier notebook (`credit-card-fraud-detection-machine-learning-2021.ipynb`) used a different methodology with Decision Trees, Random Forest, K Nearest Neighbours, and K-Means Clustering. Key differences from the current analysis:

- **Sampled subset**: Used only 500 transactions (250 fraud + 250 legitimate) instead of the full dataset
- **Accuracy as metric**: Reported 96% accuracy, but precision on fraud was only 4% — 96 of every 100 fraud flags were wrong
- **Data leakage**: Models were evaluated on data that included training samples
- **No feature scaling**: `Amount` and `Time` were unscaled, impacting distance-based models
- **No reproducibility**: No `random_state` set; used deprecated pandas APIs

The 2025 analysis addresses all of these issues. See the [notebook's final section](credit-card-fraud-detection-machine-learning-2025.ipynb) for a detailed comparison.

## Links

- [Interactive Dashboard](https://priankr.github.io/fraud-detection-machine-learning/)
- [Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Notebook on Kaggle](https://www.kaggle.com/priankravichandar/credit-card-fraud-detection-machine-learning)
- [ULB Machine Learning Group](http://mlg.ulb.ac.be)
