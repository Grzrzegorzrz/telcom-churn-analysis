# Customer Churn Prediction

A machine learning system designed to predict customer churn in telecommunications with 76% accuracy and a 0.833 ROC-AUC score, enabling proactive retention campaigns with an estimated 257% ROI.

**Built with:** Python 3.12+, scikit-learn, XGBoost

## Table of Contents
*   [Problem Statement](#problem-statement)
*   [Business Impact](#business-impact)
*   [Key Insights](#key-insights)
*   [Model Performance](#model-performance)
*   [Technical Implementation](#technical-implementation)
*   [Try It Live](#try-it-live)
*   [Installation](#installation)

## Problem Statement
Customer churn is a critical business challenge in subscription-based industries. Acquiring a new customer costs 5-25 times more than retaining an existing one. In telecommunications, the average annual churn rate is 20-30%. This project builds a predictive model to identify customers at risk of churning before they cancel, enabling targeted retention interventions.

## Business Impact

### Model Performance Summary
| Metric | Value | Business Meaning |
| :--- | :--- | :--- |
| Accuracy | 76.0% | Correctly classifies 3 out of 4 customers |
| Precision | 54% | When we flag a customer, we're right 54% of the time |
| Recall | 68% | We catch 68% of actual churners |
| ROC-AUC | 0.833 | Strong ability to rank customers by churn risk[cite: 1] |

### Return on Investment
Based on testing (1,409 customers) with a campaign cost of $75 per customer and an estimated customer lifetime value of $2,000[cite: 1]:
*   **Campaign targets:** 470 high-risk customers[cite: 1]
*   **Campaign cost:** $35,250[cite: 1]
*   **Revenue saved:** $126,000[cite: 1]
*   **Net benefit:** $90,750[cite: 1]
*   **ROI:** 257%[cite: 1]

## Key Insights

1.  **Contract Type:** Month-to-month contracts drive the highest churn rate (42.7%)[cite: 1]. One-year and two-year contracts have significantly lower churn rates (11.3% and 2.8%, respectively)[cite: 1].
    *   *Action:* Incentivize annual contracts with discounts[cite: 1].
2.  **First Year is Critical:** Nearly half of churn (47.4%) happens within the first 12 months[cite: 1].
    *   *Action:* Assign dedicated customer success managers to new customers during their first year[cite: 1].
3.  **Tech Support:** Customers without tech support have a 41.6% churn rate, compared to 15.2% for those with it[cite: 1].
    *   *Action:* Offer free tech support trials[cite: 1].
4.  **Payment Method:** Customers using electronic checks have a 45.3% churn rate[cite: 1]. Automatic payment methods see churn rates of 15.2% - 16.7%[cite: 1].
    *   *Action:* Incentivize automatic payment methods[cite: 1].
5.  **Customer Spending Patterns:** Churned customers have higher monthly charges ($74.44 vs $61.27) but lower total charges ($1,531 vs $2,555) than retained customers[cite: 1].

## Model Performance

Three models were evaluated: Logistic Regression, Random Forest, and XGBoost[cite: 1].

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 73.7% | 50.3% | 79.1% | 0.615 | 0.842 |
| Random Forest | 76.0% | 53.8% | 67.7% | 0.600 | 0.833 |
| XGBoost | 76.4% | 54.8% | 63.9% | 0.590 | 0.821 |

**Selected Model:** Random Forest with SMOTE oversampling[cite: 1].
**Rationale:** Provides the best balance of precision and recall, offers strong ROC-AUC, and provides feature importance for business insights[cite: 1].

## Technical Implementation

### Data Preprocessing & Feature Engineering
*   **Dataset:** IBM Telco Customer Churn (7,043 customers, 21 features)[cite: 1].
*   **Missing Values:** Handled missing total charges by imputing based on monthly charges[cite: 1].
*   **Feature Engineering:** Created 6 new features, including tenure groups, average monthly spend, and services count[cite: 1].
*   **Encoding:** One-hot encoding for categorical variables and binary encoding for Yes/No features (38 final features)[cite: 1].
*   **Class Imbalance:** Handled using SMOTE (Synthetic Minority Over-sampling) on training data[cite: 1].
*   **Scaling:** Applied StandardScaler to numerical features[cite: 1].

### Training Process
*   Data split: 80% train / 20% test, stratified by churn label, `random_state=42` (5,634 train / 1,409 test).
*   SMOTE applied only on the training set (sampling strategy 1.0, k-neighbors 5).
*   Final model: Random Forest (150 trees, `class_weight="balanced"`, `max_features="log2"`, `min_samples_leaf=2`).
*   Campaign targeting: the 470 customers with the highest predicted churn probability on the holdout set (decision threshold 0.454).
*   Optimization objective: model selection weighted F1-score and ROC-AUC; the final configuration balances predictive strength against the retention-campaign economics (targeting the top 470 customers, 25% retention assumption).

## Try It Live

Run the Gradio interface locally to enter customer details and get instant churn probability predictions and personalized retention recommendations[cite: 1].

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python scripts/run_all.py
```

`run_all.py` runs the full pipeline in order: key insights → final training → model comparison → ROI report. Equivalent individual steps:

```bash
.venv/bin/python scripts/insights_report.py     # data-driven key insights
.venv/bin/python scripts/train_final.py         # fit the final model, write metrics/predictions
.venv/bin/python scripts/compare_models.py      # LR / RF / XGBoost comparison table
.venv/bin/python scripts/roi_report.py          # retention-campaign economics
```
# Clone repository
git clone [https://github.com/KuldeepChoksi/customer-churn-prediction.git](https://github.com/KuldeepChoksi/customer-churn-prediction.git)
cd customer-churn-prediction

All model hyper-parameters are predefined in `src/telco_churn/config.py` (`BEST_RF_CONFIG`, `SMOTE_CONFIG`, `LR_CONFIG`, `XGB_CONFIG`) rather than discovered at runtime, so results are deterministic. The dataset ships in `data/`; no download step is required.

### Generated artifacts (`outputs/`)

| File | Contents |
| :--- | :--- |
| `final_metrics.json` | Final model metrics, confusion matrix, and hyper-parameters |
| `roi.json` | Full retention-campaign economics |
| `model_comparison.json` | LR / RF / XGBoost comparison table and settings |
| `key_insights.json` | Key-insight statistics from the raw data |
| `predictions.csv` | Per-customer holdout predictions and churn probabilities |
| `rf_smote_model.joblib`, `scaler.joblib` | Fitted model and scaler for reuse |
| `feature_names.json` | Final 33-feature list |
