import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
    }

    /* Section headers */
    .section-title {
        color: #a78bfa;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    /* Big metric */
    .metric-big {
        font-size: 56px;
        font-weight: 800;
        line-height: 1;
    }
    .metric-label {
        font-size: 13px;
        color: rgba(255,255,255,0.5);
        margin-top: 4px;
    }

    /* Risk badge */
    .risk-high   { color: #f87171; }
    .risk-medium { color: #fbbf24; }
    .risk-low    { color: #34d399; }

    /* Progress bar override */
    div[data-testid="stMetricValue"] { color: white; }

    /* White labels */
    label, .stSelectbox label, .stSlider label, .stNumberInput label { color: rgba(255,255,255,0.8) !important; }
    .stSelectbox > div > div { background: rgba(255,255,255,0.08) !important; color: white !important; }

    /* Predict button */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 700;
        padding: 14px 32px;
        width: 100%;
        letter-spacing: 1px;
        transition: all 0.2s;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(124,58,237,0.5);
    }

    h1, h2, h3, p, span { color: white; }
    .stMarkdown p { color: rgba(255,255,255,0.85); }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load("Logistic.pkl")

model = load_model()

FEATURES = list(model.feature_names_in_)

# ── Helper: build feature vector ─────────────────────────────────────────────
def build_input(vals: dict) -> pd.DataFrame:
    row = {f: 0 for f in FEATURES}

    # Numeric
    row["SeniorCitizen"]  = vals["senior"]
    row["tenure"]         = vals["tenure"]
    row["MonthlyCharges"] = vals["monthly"]
    row["TotalCharges"]   = vals["total"]

    # One-hot helpers
    def oh(prefix, value):
        key = f"{prefix}_{value}"
        if key in row:
            row[key] = 1

    oh("gender",         vals["gender"])
    oh("Partner",        vals["partner"])
    oh("Dependents",     vals["dependents"])
    oh("PhoneService",   vals["phone"])
    oh("MultipleLines",  vals["multilines"])
    oh("InternetService",vals["internet"])
    oh("OnlineSecurity", vals["security"])
    oh("OnlineBackup",   vals["backup"])
    oh("DeviceProtection",vals["device"])
    oh("TechSupport",    vals["techsupport"])
    oh("StreamingTV",    vals["tv"])
    oh("StreamingMovies",vals["movies"])
    oh("Contract",       vals["contract"])
    oh("PaperlessBilling",vals["paperless"])
    oh("PaymentMethod",  vals["payment"])

    return pd.DataFrame([row])

# ── Sidebar: inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Customer Profile")
    st.markdown("---")

    st.markdown('<p class="section-title">👤 Demographics</p>', unsafe_allow_html=True)
    gender    = st.selectbox("Gender",  ["Male", "Female"])
    senior    = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner   = st.selectbox("Has Partner",    ["No", "Yes"])
    dependents= st.selectbox("Has Dependents", ["No", "Yes"])

    st.markdown("---")
    st.markdown('<p class="section-title">📋 Account Info</p>', unsafe_allow_html=True)
    tenure    = st.slider("Tenure (months)", 0, 72, 12)
    contract  = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment   = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ])
    monthly   = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total     = st.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly * tenure), step=10.0)

    st.markdown("---")
    st.markdown('<p class="section-title">📞 Services</p>', unsafe_allow_html=True)
    phone      = st.selectbox("Phone Service",    ["Yes", "No"])
    multilines = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
    internet   = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security   = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])
    backup     = st.selectbox("Online Backup",    ["No", "Yes", "No internet service"])
    device     = st.selectbox("Device Protection",["No", "Yes", "No internet service"])
    techsupport= st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
    tv         = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
    movies     = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("---")
    predict_btn = st.button("🔮  PREDICT CHURN")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 📡 Customer Churn Predictor")
st.markdown("Predict whether a telecom customer is likely to churn using a Logistic Regression model.")
st.markdown("---")

if predict_btn:
    vals = dict(
        gender=gender, senior=1 if senior == "Yes" else 0,
        partner=partner, dependents=dependents,
        tenure=tenure, monthly=monthly, total=total,
        phone=phone, multilines=multilines, internet=internet,
        security=security, backup=backup, device=device,
        techsupport=techsupport, tv=tv, movies=movies,
        contract=contract, paperless=paperless, payment=payment,
    )

    X = build_input(vals)
    prob = model.predict_proba(X)[0][1]  # probability of churn
    pred = int(prob >= 0.5)
    pct  = round(prob * 100, 1)

    # Risk tier
    if pct >= 70:
        risk_label, risk_class, risk_icon = "HIGH RISK", "risk-high", "🔴"
    elif pct >= 40:
        risk_label, risk_class, risk_icon = "MEDIUM RISK", "risk-medium", "🟡"
    else:
        risk_label, risk_class, risk_icon = "LOW RISK", "risk-low", "🟢"

    # ── Top result row ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <p class="section-title">Churn Probability</p>
            <div class="metric-big {risk_class}">{pct}%</div>
            <div class="metric-label">{risk_icon} {risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        verdict = "Will Churn" if pred else "Will Stay"
        icon    = "⚠️" if pred else "✅"
        vcolor  = "#f87171" if pred else "#34d399"
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <p class="section-title">Prediction</p>
            <div class="metric-big" style="color:{vcolor}; font-size:42px;">{icon}</div>
            <div class="metric-label" style="font-size:18px; color:{vcolor};">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        stay_pct = round(100 - pct, 1)
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <p class="section-title">Retention Probability</p>
            <div class="metric-big" style="color:#34d399;">{stay_pct}%</div>
            <div class="metric-label">Chance of staying</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability bar ─────────────────────────────────────────────────────
    st.markdown("### Risk Gauge")
    bar_color = "#f87171" if pct >= 70 else ("#fbbf24" if pct >= 40 else "#34d399")
    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="color:#34d399; font-size:12px;">LOW RISK</span>
            <span style="color:#fbbf24; font-size:12px;">MEDIUM RISK</span>
            <span style="color:#f87171; font-size:12px;">HIGH RISK</span>
        </div>
        <div style="background:rgba(255,255,255,0.1); border-radius:999px; height:14px; position:relative;">
            <div style="
                width:{pct}%;
                background: linear-gradient(90deg, #34d399 0%, #fbbf24 50%, #f87171 100%);
                height:14px; border-radius:999px;
                box-shadow: 0 0 12px {bar_color}88;
                transition: width 1s ease;
            "></div>
            <!-- Marker -->
            <div style="
                position:absolute;
                left: calc({pct}% - 8px);
                top: -4px;
                width:22px; height:22px;
                border-radius:50%;
                background:{bar_color};
                border: 3px solid white;
                box-shadow: 0 0 10px {bar_color};
            "></div>
        </div>
        <div style="text-align:center; margin-top:12px; font-size:15px; color:white;">
            Churn risk: <strong style="color:{bar_color}">{pct}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Risk Factors ────────────────────────────────────────────────────
    st.markdown("### 🔍 Top Risk Factors")
    coef   = model.coef_[0]
    x_vals = X.values[0]
    contrib= coef * x_vals
    factor_df = pd.DataFrame({
        "Feature": FEATURES,
        "Contribution": contrib
    }).sort_values("Contribution", key=abs, ascending=False).head(8)

    col_a, col_b = st.columns(2)
    pos_factors = factor_df[factor_df["Contribution"] > 0].head(4)
    neg_factors = factor_df[factor_df["Contribution"] < 0].head(4)

    with col_a:
        st.markdown('<p class="section-title" style="color:#f87171;">↑ Increasing Churn Risk</p>', unsafe_allow_html=True)
        for _, row in pos_factors.iterrows():
            width = min(int(abs(row["Contribution"]) / factor_df["Contribution"].abs().max() * 100), 100)
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; color:rgba(255,255,255,0.85); margin-bottom:4px;">
                    <span>{row['Feature']}</span>
                    <span style="color:#f87171;">+{row['Contribution']:.3f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.1); border-radius:4px; height:8px;">
                    <div style="width:{width}%; background:#f87171; border-radius:4px; height:8px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<p class="section-title" style="color:#34d399;">↓ Reducing Churn Risk</p>', unsafe_allow_html=True)
        for _, row in neg_factors.iterrows():
            width = min(int(abs(row["Contribution"]) / factor_df["Contribution"].abs().max() * 100), 100)
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; color:rgba(255,255,255,0.85); margin-bottom:4px;">
                    <span>{row['Feature']}</span>
                    <span style="color:#34d399;">{row['Contribution']:.3f}</span>
                </div>
                <div style="background:rgba(255,255,255,0.1); border-radius:4px; height:8px;">
                    <div style="width:{width}%; background:#34d399; border-radius:4px; height:8px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Recommendations ──────────────────────────────────────────────────────
    st.markdown("### 💡 Retention Recommendations")
    recs = []
    if contract == "Month-to-month":
        recs.append(("📋 Upgrade Contract", "Offer a discounted annual or 2-year contract to reduce churn risk."))
    if internet == "Fiber optic" and security == "No":
        recs.append(("🔒 Add Online Security", "Fiber optic customers without security are high-risk — bundle it in."))
    if tenure < 12:
        recs.append(("🎁 Early Loyalty Reward", "New customers (< 1 year) are more likely to leave. Offer a loyalty bonus."))
    if payment == "Electronic check":
        recs.append(("💳 Switch Payment Method", "Electronic check users churn more — encourage auto-pay via card/bank."))
    if not recs:
        recs.append(("✅ Good Standing", "This customer has a healthy profile. Continue standard engagement."))

    rec_cols = st.columns(len(recs))
    for i, (title, desc) in enumerate(recs):
        with rec_cols[i]:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:18px; margin-bottom:8px;">{title}</div>
                <div style="font-size:13px; color:rgba(255,255,255,0.65);">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Placeholder ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="card" style="text-align:center; padding: 60px 24px;">
        <div style="font-size:64px;">📡</div>
        <h2 style="margin: 16px 0 8px;">Ready to Predict</h2>
        <p style="color:rgba(255,255,255,0.5); max-width:400px; margin:0 auto;">
            Fill in the customer profile on the left sidebar, then hit
            <strong style="color:#a78bfa;">PREDICT CHURN</strong> to get
            an instant churn probability, risk factors, and retention advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature overview
    st.markdown("### 📊 Model Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card" style="text-align:center;">
            <div style="font-size:36px; font-weight:800; color:#a78bfa;">45</div>
            <div style="color:rgba(255,255,255,0.5); font-size:13px;">Input Features</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card" style="text-align:center;">
            <div style="font-size:36px; font-weight:800; color:#60a5fa;">2</div>
            <div style="color:rgba(255,255,255,0.5); font-size:13px;">Output Classes</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card" style="text-align:center;">
            <div style="font-size:24px; font-weight:800; color:#34d399;">Logistic</div>
            <div style="color:rgba(255,255,255,0.5); font-size:13px;">Regression Model</div>
        </div>""", unsafe_allow_html=True)
