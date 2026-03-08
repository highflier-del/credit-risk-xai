import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Assessor", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    .stApp { background: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
    .metric-card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 24px; text-align: center; }
    .risk-high { background: linear-gradient(135deg, #3d0000, #1a0000); border: 1px solid #ff4444; border-radius: 16px; padding: 32px; text-align: center; }
    .risk-medium { background: linear-gradient(135deg, #2d1f00, #1a1200); border: 1px solid #ffaa00; border-radius: 16px; padding: 32px; text-align: center; }
    .risk-low { background: linear-gradient(135deg, #001a0d, #000d07); border: 1px solid #00cc66; border-radius: 16px; padding: 32px; text-align: center; }
    .factor-card { background: #161b22; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; }
    .stButton > button { background: linear-gradient(135deg, #1f6feb, #388bfd); color: white; border: none; border-radius: 8px; padding: 12px 32px; font-family: 'Syne', sans-serif; font-weight: 600; font-size: 16px; width: 100%; }
    .disclaimer { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; font-size: 12px; color: #8b949e; margin-top: 24px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET = "SeriousDlqin2yrs"
BASE_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]
FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
    "age": "Age", "NumberOfTime30-59DaysPastDueNotWorse": "30–59 Days Late",
    "DebtRatio": "Debt Ratio", "MonthlyIncome": "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberOfTimes90DaysLate": "90+ Days Late",
    "NumberRealEstateLoansOrLines": "Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60–89 Days Late",
    "NumberOfDependents": "Dependents",
    "Income_per_Dependent": "Income per Dependent",
    "MonthlyDebtProxy": "Monthly Debt Proxy",
    "Total_Delinquencies": "Total Delinquencies",
    "DebtBurden": "Debt Burden Index",
    "DelinquencyRate": "Delinquency Rate",
    "LoansPerDependent": "Loans per Dependent",
    "AgeBucket": "Age Group",
    "HighUtilization": "High Utilization Flag",
    "HasSevereDelinquency": "Severe Delinquency Flag",
    "YoungBorrower": "Young Borrower Flag"
}
MODEL_PATH = Path("credit_risk_model.pkl")
FEATURE_NAMES_PATH = Path("feature_names.pkl")


# ── Feature engineering ─────────────────────────────────────────────────────────
def engineer_features(df):
    X = df.copy()
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
        if col in X.columns:
            X[col] = X[col].clip(lower=0)
    X["Income_per_Dependent"] = np.where(X["NumberOfDependents"] > 0, X["MonthlyIncome"] / X["NumberOfDependents"], np.nan)
    X["MonthlyDebtProxy"] = X["DebtRatio"] * X["MonthlyIncome"]
    X["Total_Delinquencies"] = (X["NumberOfTime30-59DaysPastDueNotWorse"] + X["NumberOfTime60-89DaysPastDueNotWorse"] + X["NumberOfTimes90DaysLate"])
    X["DebtBurden"] = X["DebtRatio"] * X["RevolvingUtilizationOfUnsecuredLines"]
    X["DelinquencyRate"] = X["Total_Delinquencies"] / (X["NumberOfOpenCreditLinesAndLoans"] + 1)
    X["LoansPerDependent"] = X["NumberOfOpenCreditLinesAndLoans"] / (X["NumberOfDependents"] + 1)
    X["AgeBucket"] = pd.cut(X["age"], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    X["HighUtilization"] = (X["RevolvingUtilizationOfUnsecuredLines"] > 0.75).astype(int)
    X["HasSevereDelinquency"] = (X["NumberOfTimes90DaysLate"] > 0).astype(int)
    X["YoungBorrower"] = (X["age"] < 30).astype(int)
    return X


# ── Model ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH) if FEATURE_NAMES_PATH.exists() else None
        return model, feature_names
    data_path = Path("cs-training.csv")
    if not data_path.exists():
        return None, None
    df = pd.read_csv(data_path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
    y = df[TARGET].astype(int)
    X_eng = engineer_features(df[BASE_FEATURES])
    X_train, _, y_train, _ = train_test_split(X_eng, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    return model, feature_names


@st.cache_data(show_spinner=False)
def get_feature_importances(_model, _feature_names):
    return pd.Series(_model.named_steps["clf"].feature_importances_, index=_feature_names).sort_values(ascending=False)


def get_risk_level(prob):
    if prob >= 0.5:   return "HIGH",   "#ff4444", "risk-high",   "⚠️"
    elif prob >= 0.2: return "MEDIUM", "#ffaa00", "risk-medium", "⚡"
    else:             return "LOW",    "#00cc66", "risk-low",    "✅"


def validate_inputs(age, income, util, debt_ratio, late_30, late_60, late_90):
    errors = []
    if age < 18:              errors.append("Age must be at least 18.")
    if age > 100:             errors.append("Please enter a realistic age (under 100).")
    if income <= 0:           errors.append("Monthly income must be greater than $0.")
    if income > 200000:       errors.append("Monthly income seems unusually high — please double-check.")
    if not (0 <= util <= 1):  errors.append("Credit utilization must be between 0 and 1.")
    if debt_ratio < 0:        errors.append("Debt ratio cannot be negative.")
    if debt_ratio > 10:       errors.append("Debt ratio seems unusually high — please double-check.")
    if late_30 + late_60 + late_90 > 50: errors.append("Total late payments seem unusually high — please double-check.")
    return errors


def explain_prediction(base_input, engineered_row):
    explanations = []
    util      = base_input["RevolvingUtilizationOfUnsecuredLines"]
    age       = base_input["age"]
    late_90   = base_input["NumberOfTimes90DaysLate"]
    late_30   = base_input["NumberOfTime30-59DaysPastDueNotWorse"]
    late_60   = base_input["NumberOfTime60-89DaysPastDueNotWorse"]
    income    = base_input["MonthlyIncome"]
    debt_burden      = engineered_row.get("DebtBurden", 0)
    delinquency_rate = engineered_row.get("DelinquencyRate", 0)

    if util > 0.75:
        explanations.append(("Very high credit utilization", f"{util:.0%} of available credit used — lenders prefer below 30%.", "negative"))
    elif util > 0.4:
        explanations.append(("Moderate credit utilization", f"{util:.0%} utilization — slightly elevated.", "neutral"))
    else:
        explanations.append(("Healthy credit utilization", f"Only {util:.0%} of available credit is used.", "positive"))

    if late_90 > 0:
        explanations.append(("Severe delinquency history", f"{int(late_90)} instance(s) of 90+ day late payments — strongest default predictor.", "negative"))

    total_late = late_30 + late_60
    if total_late > 2:
        explanations.append(("Multiple late payments", f"{int(total_late)} instances of 30–89 day late payments.", "negative"))
    elif total_late == 0 and late_90 == 0:
        explanations.append(("Clean payment history", "No late payments recorded — a strong positive signal.", "positive"))

    if age < 30:
        explanations.append(("Young borrower", "Under 30 — statistically higher risk due to shorter credit history.", "negative"))
    elif age > 50:
        explanations.append(("Established credit history", f"Age {int(age)} — longer track record lowers risk.", "positive"))

    if debt_burden > 0.3:
        explanations.append(("High combined debt burden", "High debt ratio AND high utilization together signal financial stress.", "negative"))

    if delinquency_rate > 0.2:
        explanations.append(("High delinquency rate", "Late payments are high relative to number of credit lines.", "negative"))

    if income < 3000:
        explanations.append(("Low monthly income", f"${income:,.0f}/month may limit repayment capacity.", "negative"))
    elif income > 8000:
        explanations.append(("Strong monthly income", f"${income:,.0f}/month suggests good repayment capacity.", "positive"))

    return explanations


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk Assessor")
    st.markdown("*Powered by Explainable AI*")
    st.markdown("---")
    st.markdown("### Borrower Information")

    age        = st.number_input("Age", min_value=18, max_value=100, value=40)
    income     = st.number_input("Monthly Income ($)", min_value=0, max_value=200000, value=5000, step=100)
    util       = st.slider("Credit Utilization Rate", 0.0, 1.0, 0.3, 0.01, help="0 = no credit used, 1 = fully maxed out")
    debt_ratio = st.slider("Debt Ratio", 0.0, 5.0, 0.3, 0.01, help="Monthly debt payments ÷ gross monthly income")

    st.markdown("**Payment History**")
    late_30 = st.number_input("Times 30–59 Days Late", min_value=0, max_value=30, value=0)
    late_60 = st.number_input("Times 60–89 Days Late", min_value=0, max_value=30, value=0)
    late_90 = st.number_input("Times 90+ Days Late",   min_value=0, max_value=30, value=0)

    st.markdown("**Credit Profile**")
    open_lines  = st.number_input("Open Credit Lines & Loans", min_value=0, max_value=50, value=5)
    real_estate = st.number_input("Real Estate Loans",         min_value=0, max_value=20, value=1)
    dependents  = st.number_input("Number of Dependents",      min_value=0, max_value=20, value=0)

    assess_btn = st.button("🔍 Assess Credit Risk")

# ── Main ────────────────────────────────────────────────────────────────────────
st.markdown("# Credit Risk Assessment")
st.markdown("*Enter borrower details in the sidebar, then click **Assess Credit Risk**.*")
st.markdown("---")

with st.spinner("Loading model..."):
    model, feature_names = load_or_train_model()

if model is None:
    st.error("⚠️ Model could not be loaded.")
    st.info("Place `cs-training.csv` or `credit_risk_model.pkl` + `feature_names.pkl` in the same folder as this app.")
    st.stop()

if assess_btn:
    errors = validate_inputs(age, income, util, debt_ratio, late_30, late_60, late_90)
    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
        st.stop()

    base_input = {
        "RevolvingUtilizationOfUnsecuredLines":  util,
        "age":                                   float(age),
        "NumberOfTime30-59DaysPastDueNotWorse":  float(late_30),
        "DebtRatio":                             debt_ratio,
        "MonthlyIncome":                         float(income),
        "NumberOfOpenCreditLinesAndLoans":        float(open_lines),
        "NumberOfTimes90DaysLate":               float(late_90),
        "NumberRealEstateLoansOrLines":          float(real_estate),
        "NumberOfTime60-89DaysPastDueNotWorse":  float(late_60),
        "NumberOfDependents":                    float(dependents)
    }

    input_df = engineer_features(pd.DataFrame([base_input]))
    if feature_names:
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[feature_names]

    prob = model.predict_proba(input_df)[0][1]
    risk_level, risk_color, risk_class, risk_icon = get_risk_level(prob)

    # Result banner
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
        <div class="{risk_class}">
            <div style="font-size:48px">{risk_icon}</div>
            <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:800;color:{risk_color};margin:8px 0">{risk_level} RISK</div>
            <div style="font-size:52px;font-weight:700;color:{risk_color}">{prob:.1%}</div>
            <div style="color:#8b949e;font-size:14px;margin-top:4px">Estimated probability of default within 2 years</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    cl, cr = st.columns(2)

    # Gauge
    with cl:
        st.markdown("### 📊 Risk Gauge")
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.barh(0, 1,   color="#21262d", height=0.5, zorder=1)
        ax.barh(0, 0.2, color="#003d1f", height=0.5, zorder=2)
        ax.barh(0, 0.3, color="#3d2800", height=0.5, left=0.2, zorder=2)
        ax.barh(0, 0.5, color="#3d0000", height=0.5, left=0.5, zorder=2)
        ax.axvline(prob, color=risk_color, linewidth=4, zorder=5)
        ax.scatter([prob], [0], color=risk_color, s=200, zorder=6)
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.8); ax.set_yticks([])
        ax.set_xticks([0, 0.2, 0.5, 1.0])
        ax.set_xticklabels(["0%", "20%\n(Low/Med)", "50%\n(Med/High)", "100%"], color="#8b949e", fontsize=9)
        ax.spines[:].set_visible(False)
        ax.text(prob, 0.4, f"{prob:.1%}", ha="center", va="bottom", color=risk_color, fontsize=16, fontweight="bold")
        ax.legend(handles=[
            mpatches.Patch(color="#00cc66", label="Low (<20%)"),
            mpatches.Patch(color="#ffaa00", label="Medium (20–50%)"),
            mpatches.Patch(color="#ff4444", label="High (>50%)")
        ], loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Explanations
    with cr:
        st.markdown("### 🔍 Why This Score?")
        eng_row = input_df.iloc[0].to_dict()
        explanations = explain_prediction(base_input, eng_row)
        for title, detail, sentiment in explanations:
            color = {"negative": "#ff4444", "positive": "#00cc66", "neutral": "#ffaa00"}[sentiment]
            icon  = {"negative": "🔴",       "positive": "🟢",       "neutral": "🟡"}[sentiment]
            st.markdown(f"""
            <div class="factor-card" style="border-left:3px solid {color}">
                <strong style="color:{color}">{icon} {title}</strong><br>
                <span style="color:#8b949e;font-size:13px">{detail}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Feature importance
    st.markdown("### 📈 What Matters Most (Global Feature Importance)")
    fn = feature_names or input_df.columns.tolist()
    importances = get_feature_importances(model, tuple(fn))
    top = importances.head(12)
    labels = [FEATURE_LABELS.get(f, f) for f in top.index]
    fig2, ax2 = plt.subplots(figsize=(10, 4.5), facecolor="#0d1117")
    ax2.set_facecolor("#0d1117")
    colors2 = ["#388bfd" if i < 3 else "#1f6feb" for i in range(len(top))]
    bars2 = ax2.barh(labels[::-1], top.values[::-1], color=colors2[::-1], height=0.6)
    ax2.set_xlabel("Feature Importance", color="#8b949e")
    ax2.tick_params(colors="#8b949e"); ax2.spines[:].set_visible(False)
    for bar, val in zip(bars2, top.values[::-1]):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", color="#8b949e", fontsize=9)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Summary cards
    st.markdown("### 📋 Borrower Summary")
    cols5 = st.columns(5)
    for col, (label, val) in zip(cols5, [
        ("Credit Utilization", f"{util:.0%}"), ("Age", str(age)),
        ("Monthly Income", f"${income:,}"), ("Debt Ratio", f"{debt_ratio:.2f}"),
        ("Total Late Payments", str(late_30 + late_60 + late_90))
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#8b949e;font-size:12px">{label}</div>
                <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:#e6edf3">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This tool is for educational and portfolio demonstration purposes only. Not intended as an actual credit decision-making system.</div>""", unsafe_allow_html=True)

else:
    c1, c2, c3 = st.columns(3)
    for col, (icon, title, desc) in zip([c1, c2, c3], [
        ("🤖", "Random Forest Model",  "Trained on 150,000 borrower records — ROC-AUC of 0.858"),
        ("🔍", "Explainable AI",       "Every prediction comes with plain-English explanation of risk factors"),
        ("📊", "17 Features",          "10 original + 7 engineered features for richer financial behaviour patterns")
    ]):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:32px">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;margin:8px 0">{title}</div>
                <div style="color:#8b949e;font-size:13px">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 👈 Enter borrower details in the sidebar and click **Assess Credit Risk** to get started.")
    st.markdown("""<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This tool is for educational and portfolio demonstration purposes only.</div>""", unsafe_allow_html=True)
