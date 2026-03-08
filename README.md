# 🏦 Credit Risk XAI

An end-to-end machine learning project that predicts borrower default risk using the **Give Me Some Credit** Kaggle dataset, with a strong emphasis on **Explainable AI (XAI)** — making every prediction transparent, interpretable, and regulator-friendly.

🔗 **[Live Demo →]https://credit-risk-xai.streamlit.app/

---

## 📌 What Problem Does This Solve?

Any loan officer can tell you that missed payments are a red flag. The real challenge is **quantifying exactly how risky a borrower is, across dozens of variables, consistently, at scale.**

A bank processing 10,000 loan applications a month can't rely on human judgement alone. This model:
- Predicts default probability instantly for any borrower profile
- Explains *why* a borrower is flagged as risky in plain English
- Provides the audit trail regulators require when a loan is declined
- Enables risk-based pricing (not just approve/reject, but *at what interest rate*)

---

## 📊 Dataset

**Source:** [Give Me Some Credit – Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

- 150,000 borrower records
- 10 financial features (credit utilization, age, debt ratio, income, delinquency history, etc.)
- Binary target: `SeriousDlqin2yrs` — whether a borrower defaulted within 2 years
- Class imbalance: ~6.7% defaults vs ~93.3% non-defaults

---

## 🤖 Models

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|
| Logistic Regression | 0.835 | 0.336 | 0.167 |
| **Random Forest** | **0.858** | **0.390** | **0.067** |

Random Forest outperforms across all metrics and is better calibrated — making it the preferred model for risk-based decisions.

---

## 🔧 Feature Engineering

**Original features (10)**
- Winsorization of outliers at 1st/99th percentile
- `Income_per_Dependent` — financial strain relative to family size
- `MonthlyDebtProxy` — estimated monthly debt in dollar terms
- `Total_Delinquencies` — aggregated late payment count across 30/60/90-day windows

**New engineered features (7)**
- `DebtBurden` — combined pressure from debt ratio × credit utilization
- `DelinquencyRate` — late payments relative to number of open credit lines
- `LoansPerDependent` — credit load relative to family obligations
- `AgeBucket` — non-linear age risk bands (under 25, 25–35, 35–50, 50–65, 65+)
- `HighUtilization` — binary flag for utilization above 75%
- `HasSevereDelinquency` — binary flag for any 90+ day late payment
- `YoungBorrower` — binary flag for borrowers under 30

---

## 🔍 Explainability Techniques

| Technique | Purpose |
|---|---|
| **Permutation Importance** | Ranks which features influence predictions the most |
| **Partial Dependence Plots (PDP)** | Shows how risk changes as a single feature varies |
| **Individual Conditional Expectation (ICE)** | Reveals prediction differences across individual borrowers |
| **Logistic Regression Coefficients & Odds Ratios** | Direct linear model interpretability |
| **Calibration Curves** | Validates predicted probabilities against actual default rates |

### Key Findings
- **Credit utilization, delinquency history, and age** are the top drivers of default risk
- Short-term delinquencies increase default odds by **3.7×** per standard deviation
- 90-day late payments increase odds by **2.7×**
- Older age reduces default risk by ~**23%** per standard deviation
- High utilization + high debt ratio together (DebtBurden) is a stronger signal than either alone

---

## 🚀 Interactive Dashboard

Built with **Streamlit** — enter any borrower profile and get:
- A default probability score (0–100%)
- Low / Medium / High risk classification
- Plain-English explanation of the top risk factors driving the prediction
- Global feature importance chart
- Input validation to prevent unrealistic entries

### Running Locally

```bash
# Using Anaconda Prompt (recommended)
git clone https://github.com/yourusername/credit-risk-xai
cd credit-risk-xai
pip install -r requirements.txt
streamlit run credit_xai_app.py
```

> The app will automatically train the model on first launch using `cs-training.csv`. This takes 1–2 minutes. Subsequent loads are instant.

---

## 📁 Repository Structure

```
credit-risk-xai/
├── credit_xai_app.py                            # Streamlit dashboard
├── 25754730_XAI_for_Credit_Risk_Assessment.ipynb  # Full analysis notebook
├── cs-training.csv                              # Dataset
├── feature_names.pkl                            # Saved feature column names
├── requirements.txt                             # Python dependencies
└── README.md
```

---

## 🧰 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
joblib>=1.3.0
```

---

## 💼 Business Implications

- **Automated screening** — flags high-risk applicants instantly at scale
- **Regulatory compliance** — XAI outputs provide explainable, auditable decisions
- **Risk-based pricing** — calibrated probabilities support tiered interest rate decisions
- **Consistent decisions** — eliminates human variability across thousands of applications

### Where This Model Has Limitations
- Uses 10 base features — real credit models use hundreds (spending patterns, employment history, etc.)
- Trained on 2011 data — financial behaviour patterns have evolved
- Does not account for macroeconomic conditions
- Age as a feature raises potential fairness concerns under anti-discrimination laws in some jurisdictions

---

## 👤 Author

**Tarun Singh** — Student ID: 25754730

---

## 📄 License

This project is for educational and portfolio purposes. Dataset sourced from Kaggle under their competition terms.
