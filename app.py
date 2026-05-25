import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI | Customer Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Logistic.pkl")

model = load_model()

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Root Variables */
:root {
    --bg-primary: #0A0E1A;
    --bg-secondary: #111827;
    --bg-card: #161D2E;
    --accent-blue: #3B82F6;
    --accent-cyan: #06B6D4;
    --accent-green: #10B981;
    --accent-red: #EF4444;
    --accent-amber: #F59E0B;
    --text-primary: #F1F5F9;
    --text-muted: #64748B;
    --border: rgba(59, 130, 246, 0.15);
}

/* Global Reset */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
}

.stApp {
    background: linear-gradient(135deg, #0A0E1A 0%, #0F172A 50%, #0A0E1A 100%);
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1421 0%, #111827 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Section Headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    color: var(--accent-blue) !important;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding: 8px 0 4px;
    border-bottom: 1px solid var(--border);
}

/* Metric Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0D1B3E 0%, #1a1040 50%, #0D2340 100%);
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '⬡';
    position: absolute;
    right: -20px; top: -30px;
    font-size: 200px;
    color: rgba(59,130,246,0.04);
    line-height: 1;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, #F1F5F9, #93C5FD, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
}
.hero-sub {
    color: #94A3B8;
    font-size: 15px;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Risk Badge */
.risk-high {
    background: linear-gradient(135deg, #450A0A, #7F1D1D);
    border: 1px solid #DC2626;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #052E16, #14532D);
    border: 1px solid #16A34A;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.risk-medium {
    background: linear-gradient(135deg, #451A03, #78350F);
    border: 1px solid #D97706;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

/* Gauge container */
.gauge-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
}

/* Action Button */
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #0891B2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 40px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.4) !important;
}

/* Input fields */
[data-testid="stSelectbox"], [data-testid="stNumberInput"] {
    background: var(--bg-secondary) !important;
}

/* Slider */
.stSlider > div {
    padding: 4px 0;
}

/* Info box */
.info-box {
    background: rgba(59,130,246,0.08);
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 13px;
    color: #94A3B8;
}

/* Feature importance bars */
.feat-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    gap: 10px;
}
.feat-label {
    font-size: 12px;
    color: #94A3B8;
    width: 160px;
    text-align: right;
    flex-shrink: 0;
}
.feat-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    height: 8px;
}
.feat-val {
    font-size: 11px;
    color: #64748B;
    width: 40px;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    box-shadow: 0 0 8px var(--accent-green);
    animation: pulse 2s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 13px;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper: Build Feature Vector ────────────────────────────────────────────
def build_features(data: dict) -> pd.DataFrame:
    features = {
        'SeniorCitizen': [data['senior']],
        'tenure': [data['tenure']],
        'MonthlyCharges': [data['monthly']],
        'TotalCharges': [data['monthly'] * data['tenure']],
        'gender_Female': [1 if data['gender'] == 'Female' else 0],
        'gender_Male': [1 if data['gender'] == 'Male' else 0],
        'Partner_No': [1 if data['partner'] == 'No' else 0],
        'Partner_Yes': [1 if data['partner'] == 'Yes' else 0],
        'Dependents_No': [1 if data['dependents'] == 'No' else 0],
        'Dependents_Yes': [1 if data['dependents'] == 'Yes' else 0],
        'PhoneService_No': [1 if data['phone'] == 'No' else 0],
        'PhoneService_Yes': [1 if data['phone'] == 'Yes' else 0],
        'MultipleLines_No': [1 if data['multilines'] == 'No' else 0],
        'MultipleLines_No phone service': [1 if data['multilines'] == 'No phone service' else 0],
        'MultipleLines_Yes': [1 if data['multilines'] == 'Yes' else 0],
        'InternetService_DSL': [1 if data['internet'] == 'DSL' else 0],
        'InternetService_Fiber optic': [1 if data['internet'] == 'Fiber optic' else 0],
        'InternetService_No': [1 if data['internet'] == 'No' else 0],
        'OnlineSecurity_No': [1 if data['security'] == 'No' else 0],
        'OnlineSecurity_No internet service': [1 if data['security'] == 'No internet service' else 0],
        'OnlineSecurity_Yes': [1 if data['security'] == 'Yes' else 0],
        'OnlineBackup_No': [1 if data['backup'] == 'No' else 0],
        'OnlineBackup_No internet service': [1 if data['backup'] == 'No internet service' else 0],
        'OnlineBackup_Yes': [1 if data['backup'] == 'Yes' else 0],
        'DeviceProtection_No': [1 if data['device'] == 'No' else 0],
        'DeviceProtection_No internet service': [1 if data['device'] == 'No internet service' else 0],
        'DeviceProtection_Yes': [1 if data['device'] == 'Yes' else 0],
        'TechSupport_No': [1 if data['techsupport'] == 'No' else 0],
        'TechSupport_No internet service': [1 if data['techsupport'] == 'No internet service' else 0],
        'TechSupport_Yes': [1 if data['techsupport'] == 'Yes' else 0],
        'StreamingTV_No': [1 if data['tv'] == 'No' else 0],
        'StreamingTV_No internet service': [1 if data['tv'] == 'No internet service' else 0],
        'StreamingTV_Yes': [1 if data['tv'] == 'Yes' else 0],
        'StreamingMovies_No': [1 if data['movies'] == 'No' else 0],
        'StreamingMovies_No internet service': [1 if data['movies'] == 'No internet service' else 0],
        'StreamingMovies_Yes': [1 if data['movies'] == 'Yes' else 0],
        'Contract_Month-to-month': [1 if data['contract'] == 'Month-to-month' else 0],
        'Contract_One year': [1 if data['contract'] == 'One year' else 0],
        'Contract_Two year': [1 if data['contract'] == 'Two year' else 0],
        'PaperlessBilling_No': [1 if data['paperless'] == 'No' else 0],
        'PaperlessBilling_Yes': [1 if data['paperless'] == 'Yes' else 0],
        'PaymentMethod_Bank transfer': [1 if data['payment'] == 'Bank transfer' else 0],
        'PaymentMethod_Credit card': [1 if data['payment'] == 'Credit card' else 0],
        'PaymentMethod_Electronic check': [1 if data['payment'] == 'Electronic check' else 0],
        'PaymentMethod_Mailed check': [1 if data['payment'] == 'Mailed check' else 0],
    }
    return pd.DataFrame(features)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 10px; text-align:center;'>
        <div style='font-family:Syne;font-size:22px;font-weight:800;
            background:linear-gradient(135deg,#3B82F6,#06B6D4);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            🛡️ ChurnGuard AI
        </div>
        <div style='font-size:11px;color:#475569;letter-spacing:2px;margin-top:4px;'>
            CUSTOMER INTELLIGENCE
        </div>
        <div style='margin-top:10px;font-size:12px;color:#334155;'>
            <span class='status-dot'></span>Model Active · v2.1
        </div>
    </div>
    <div class='custom-divider'></div>
    """, unsafe_allow_html=True)

    # ── DEMOGRAPHICS ──
    st.markdown("<div class='section-label'>👤 Demographics</div>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ── ACCOUNT INFO ──
    st.markdown("<div class='section-label'>💼 Account Info</div>", unsafe_allow_html=True)
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ])
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ── SERVICES ──
    st.markdown("<div class='section-label'>📡 Services</div>", unsafe_allow_html=True)
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multilines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    net_opts = ["No", "No internet service", "Yes"]
    security = st.selectbox("Online Security", net_opts)
    backup = st.selectbox("Online Backup", net_opts)
    device = st.selectbox("Device Protection", net_opts)
    techsupport = st.selectbox("Tech Support", net_opts)
    tv = st.selectbox("Streaming TV", net_opts)
    movies = st.selectbox("Streaming Movies", net_opts)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ RUN CHURN ANALYSIS")


# ─── Main Layout ─────────────────────────────────────────────────────────────
# Hero Banner
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>Customer Churn Intelligence Platform</div>
    <div class='hero-sub'>
        AI-powered retention analytics · Logistic Regression Engine · Real-time Risk Scoring
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯  Risk Analysis", "📊  Feature Insights", "📋  Model Info"])

# ─── TAB 1: Risk Analysis ─────────────────────────────────────────────────────
with tab1:
    if not predict_btn:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;'>
            <div style='font-size:64px;margin-bottom:16px;'>🛡️</div>
            <div style='font-family:Syne;font-size:22px;font-weight:700;color:#F1F5F9;margin-bottom:8px;'>
                Ready to Analyze
            </div>
            <div style='color:#64748B;font-size:14px;max-width:400px;margin:0 auto;'>
                Configure customer profile in the sidebar and click
                <strong style='color:#3B82F6;'>Run Churn Analysis</strong> to get AI-powered risk assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Build features & predict
        inp = {
            'senior': 1 if senior == 'Yes' else 0,
            'tenure': tenure,
            'monthly': monthly,
            'gender': gender,
            'partner': partner,
            'dependents': dependents,
            'phone': phone,
            'multilines': multilines,
            'internet': internet,
            'security': security,
            'backup': backup,
            'device': device,
            'techsupport': techsupport,
            'tv': tv,
            'movies': movies,
            'contract': contract,
            'paperless': paperless,
            'payment': payment,
        }

        df = build_features(inp)
        prob = model.predict_proba(df)[0][1]
        pred = model.predict(df)[0]
        pct = prob * 100
        total_charges = monthly * tenure

        # Risk level
        if pct >= 70:
            risk_label, risk_color, risk_emoji, risk_class = "HIGH RISK", "#EF4444", "🔴", "risk-high"
            action = "Immediate intervention required. Offer premium retention package."
        elif pct >= 40:
            risk_label, risk_color, risk_emoji, risk_class = "MEDIUM RISK", "#F59E0B", "🟡", "risk-medium"
            action = "Monitor closely. Consider proactive engagement campaign."
        else:
            risk_label, risk_color, risk_emoji, risk_class = "LOW RISK", "#10B981", "🟢", "risk-low"
            action = "Customer is stable. Opportunity for upsell or loyalty rewards."

        # Row 1: KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:11px;letter-spacing:2px;color:#64748B;font-family:Syne;'>CHURN PROBABILITY</div>
                <div style='font-size:42px;font-weight:800;font-family:Syne;color:{risk_color};margin:8px 0;'>{pct:.1f}%</div>
                <div style='font-size:12px;color:#475569;'>Model Confidence</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:11px;letter-spacing:2px;color:#64748B;font-family:Syne;'>RISK LEVEL</div>
                <div style='font-size:28px;font-weight:800;font-family:Syne;color:{risk_color};margin:8px 0;'>{risk_emoji} {risk_label}</div>
                <div style='font-size:12px;color:#475569;'>Classification</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:11px;letter-spacing:2px;color:#64748B;font-family:Syne;'>TENURE</div>
                <div style='font-size:42px;font-weight:800;font-family:Syne;color:#3B82F6;margin:8px 0;'>{tenure}</div>
                <div style='font-size:12px;color:#475569;'>Months Active</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:11px;letter-spacing:2px;color:#64748B;font-family:Syne;'>LIFETIME VALUE</div>
                <div style='font-size:32px;font-weight:800;font-family:Syne;color:#06B6D4;margin:8px 0;'>${total_charges:,.0f}</div>
                <div style='font-size:12px;color:#475569;'>Total Charges</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)

        # Row 2: Gauge + Recommendation
        col_g, col_r = st.columns([1.2, 1])

        with col_g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={'suffix': "%", 'font': {'size': 52, 'color': risk_color, 'family': 'Syne'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155",
                             'tickfont': {'color': '#64748B', 'size': 11}},
                    'bar': {'color': risk_color, 'thickness': 0.25},
                    'bgcolor': "#161D2E",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(16,185,129,0.15)'},
                        {'range': [40, 70], 'color': 'rgba(245,158,11,0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239,68,68,0.15)'},
                    ],
                    'threshold': {
                        'line': {'color': risk_color, 'width': 3},
                        'thickness': 0.8,
                        'value': pct
                    }
                },
                title={'text': "CHURN RISK SCORE", 'font': {'size': 13, 'color': '#64748B', 'family': 'Syne'}}
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=280,
                margin=dict(t=40, b=10, l=20, r=20)
            )
            st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key="gauge")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown(f"""
            <div class='{risk_class}' style='height:100%;min-height:250px;display:flex;flex-direction:column;justify-content:center;'>
                <div style='font-family:Syne;font-size:13px;letter-spacing:3px;color:rgba(255,255,255,0.5);margin-bottom:10px;'>
                    AI RECOMMENDATION
                </div>
                <div style='font-size:32px;margin-bottom:12px;'>{risk_emoji}</div>
                <div style='font-family:Syne;font-size:18px;font-weight:700;color:white;margin-bottom:12px;'>
                    {"Will Churn" if pred == 1 else "Will Stay"}
                </div>
                <div style='font-size:13px;color:rgba(255,255,255,0.7);line-height:1.6;'>
                    {action}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin:24px 0;'></div>", unsafe_allow_html=True)

        # Row 3: Probability Bar Chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=["Will Stay", "Will Churn"],
            x=[(1 - prob) * 100, pct],
            orientation='h',
            marker=dict(
                color=["rgba(16,185,129,0.8)", f"rgba({','.join(['239','68','68'] if pct >= 70 else ['245','158','11'] if pct >= 40 else ['16','185','129'])},0.85)"],
                line=dict(color='rgba(255,255,255,0.1)', width=1),
            ),
            text=[f"{(1-prob)*100:.1f}%", f"{pct:.1f}%"],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='white', size=14, family='Syne'),
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=[0, 100], showgrid=False, color='#475569',
                       ticksuffix='%', title='Probability', title_font=dict(color='#64748B')),
            yaxis=dict(showgrid=False, color='#94A3B8'),
            height=160,
            margin=dict(t=10, b=10, l=10, r=10),
            title=dict(text='Prediction Breakdown', font=dict(family='Syne', size=14, color='#94A3B8'), x=0.5),
            bargap=0.3,
        )
        st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True, key="prob_bar")
        st.markdown("</div>", unsafe_allow_html=True)

        # Customer Summary Box
        st.markdown(f"""
        <div class='info-box'>
            <strong style='color:#3B82F6;font-family:Syne;'>📋 Customer Profile Summary</strong><br>
            <span style='color:#94A3B8;'>
            {gender} · {'Senior' if senior=='Yes' else 'Non-Senior'} · {contract} Contract · ${monthly}/mo ·
            {tenure} months tenure · {internet} internet · {payment} payment ·
            {'Has Partner' if partner=='Yes' else 'No Partner'} · {'Has Dependents' if dependents=='Yes' else 'No Dependents'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin:28px 0 8px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:Syne;font-size:16px;font-weight:700;color:#F1F5F9;margin-bottom:16px;
            padding-bottom:10px;border-bottom:1px solid rgba(59,130,246,0.15);'>
            📈 Deep Dive Analytics
        </div>
        """, unsafe_allow_html=True)

        g1, g2 = st.columns(2)

        # ── GRAPH 1: Monthly Charge Risk Simulation ──
        with g1:
            st.markdown("<div style='font-size:13px;font-weight:600;color:#94A3B8;margin-bottom:8px;font-family:Syne;letter-spacing:1px;'>💸 MONTHLY CHARGE vs CHURN RISK</div>", unsafe_allow_html=True)
            charge_range = list(range(20, 121, 5))
            charge_probs = []
            for ch in charge_range:
                temp_c = inp.copy()
                temp_c['monthly'] = ch
                charge_probs.append(model.predict_proba(build_features(temp_c))[0][1] * 100)

            fig_c = go.Figure()
            fig_c.add_hrect(y0=0, y1=40, fillcolor="rgba(16,185,129,0.06)", line_width=0)
            fig_c.add_hrect(y0=40, y1=70, fillcolor="rgba(245,158,11,0.06)", line_width=0)
            fig_c.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.06)", line_width=0)
            fig_c.add_trace(go.Scatter(
                x=charge_range, y=charge_probs,
                mode='lines+markers',
                line=dict(color='#06B6D4', width=2.5, shape='spline'),
                marker=dict(size=5, color='#06B6D4', line=dict(color='white', width=1)),
                fill='tozeroy',
                fillcolor='rgba(6,182,212,0.07)',
                name='Churn %',
                hovertemplate='$%{x}/mo → %{y:.1f}% risk<extra></extra>'
            ))
            fig_c.add_trace(go.Scatter(
                x=[monthly], y=[pct],
                mode='markers',
                marker=dict(size=14, color=risk_color, symbol='diamond',
                            line=dict(color='white', width=2)),
                name='Current Customer',
                hovertemplate=f'Current: ${monthly}/mo → {pct:.1f}%<extra></extra>'
            ))
            fig_c.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.4)",
                            annotation_text="High Risk Threshold", annotation_font_color='#EF4444',
                            annotation_font_size=10)
            fig_c.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title='Monthly Charges ($)', color='#64748B', showgrid=True,
                           gridcolor='rgba(255,255,255,0.04)', title_font=dict(color='#64748B'),
                           tickprefix='$'),
                yaxis=dict(title='Churn Probability (%)', color='#64748B', showgrid=False,
                           title_font=dict(color='#64748B'), range=[0, 105]),
                height=280, margin=dict(t=15, b=15, l=10, r=10),
                showlegend=True,
                legend=dict(font=dict(color='#64748B', size=10), bgcolor='rgba(0,0,0,0)',
                            orientation='h', y=-0.2),
            )
            st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_c, use_container_width=True, key="charge_sim")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── GRAPH 2: Service Usage Radar ──
        with g2:
            st.markdown("<div style='font-size:13px;font-weight:600;color:#94A3B8;margin-bottom:8px;font-family:Syne;letter-spacing:1px;'>🕸️ SERVICE USAGE RADAR</div>", unsafe_allow_html=True)

            radar_categories = [
                'Phone', 'Online Security', 'Online Backup',
                'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies'
            ]
            def svc_score(val): return 1 if val == 'Yes' else (0.5 if val == 'No' else 0)

            customer_scores = [
                svc_score(phone), svc_score(security), svc_score(backup),
                svc_score(device), svc_score(techsupport), svc_score(tv), svc_score(movies)
            ]
            radar_cats_closed = radar_categories + [radar_categories[0]]
            scores_closed = customer_scores + [customer_scores[0]]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=radar_cats_closed,
                fill='toself',
                fillcolor=f"rgba(239,68,68,0.15)" if pct >= 70 else f"rgba(245,158,11,0.15)" if pct >= 40 else f"rgba(16,185,129,0.15)",
                line=dict(color=risk_color, width=2),
                marker=dict(size=7, color=risk_color),
                name='Customer',
                hovertemplate='%{theta}: %{r}<extra></extra>'
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=[1]*8,
                theta=radar_cats_closed,
                fill='toself',
                fillcolor='rgba(59,130,246,0.05)',
                line=dict(color='rgba(59,130,246,0.3)', width=1, dash='dot'),
                name='Full Services',
                hoverinfo='skip'
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                                   gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.1)'),
                    angularaxis=dict(color='#64748B', gridcolor='rgba(255,255,255,0.08)',
                                    linecolor='rgba(255,255,255,0.1)', tickfont=dict(size=10, color='#94A3B8'))
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                height=280, margin=dict(t=20, b=20, l=30, r=30),
                showlegend=True,
                legend=dict(font=dict(color='#64748B', size=10), bgcolor='rgba(0,0,0,0)',
                            orientation='h', y=-0.15),
            )
            st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
            st.plotly_chart(fig_r, use_container_width=True, key="radar")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── GRAPH 3: Contract Type Risk Comparison ──
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;font-weight:600;color:#94A3B8;margin-bottom:8px;font-family:Syne;letter-spacing:1px;'>📋 CONTRACT TYPE RISK COMPARISON</div>", unsafe_allow_html=True)

        contracts_list = ['Month-to-month', 'One year', 'Two year']
        contract_probs = []
        for ct in contracts_list:
            temp_ct = inp.copy()
            temp_ct['contract'] = ct
            contract_probs.append(model.predict_proba(build_features(temp_ct))[0][1] * 100)

        bar_colors = [risk_color if ct == contract else 'rgba(99,130,191,0.4)' for ct in contracts_list]

        fig_con = go.Figure()
        fig_con.add_trace(go.Bar(
            x=contracts_list,
            y=contract_probs,
            marker=dict(color=bar_colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
            text=[f"{v:.1f}%" for v in contract_probs],
            textposition='outside',
            textfont=dict(color='#94A3B8', size=13, family='Syne'),
            hovertemplate='%{x}: %{y:.1f}% churn risk<extra></extra>',
            width=0.4,
        ))
        cur_idx = contracts_list.index(contract)
        fig_con.add_annotation(
            x=contract, y=contract_probs[cur_idx] + 5,
            text="◀ Current", showarrow=False,
            font=dict(color=risk_color, size=11, family='Syne'),
        )
        fig_con.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.35)",
                          annotation_text="High Risk Zone", annotation_font_color='#EF4444',
                          annotation_font_size=10)
        fig_con.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#94A3B8', showgrid=False),
            yaxis=dict(title='Churn Probability (%)', color='#64748B', showgrid=True,
                       gridcolor='rgba(255,255,255,0.04)', title_font=dict(color='#64748B'),
                       range=[0, max(contract_probs) + 20]),
            height=240, margin=dict(t=20, b=10, l=10, r=10),
            bargap=0.5, showlegend=False,
        )
        st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
        st.plotly_chart(fig_con, use_container_width=True, key="contract_cmp")
        st.markdown("</div>", unsafe_allow_html=True)


# ─── TAB 2: Feature Insights ─────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style='font-family:Syne;font-size:18px;font-weight:700;color:#F1F5F9;margin-bottom:6px;'>
        Feature Impact Analysis
    </div>
    <div style='color:#64748B;font-size:13px;margin-bottom:24px;'>
        Model coefficient weights showing each feature's influence on churn prediction
    </div>
    """, unsafe_allow_html=True)

    # Feature importance from coefficients
    feature_names = list(model.feature_names_in_)
    coef = model.coef_[0]

    feat_df = pd.DataFrame({'feature': feature_names, 'coef': coef})
    feat_df['abs_coef'] = feat_df['coef'].abs()
    feat_df = feat_df.sort_values('abs_coef', ascending=True).tail(20)

    fig3 = go.Figure()
    colors = ['#EF4444' if c > 0 else '#10B981' for c in feat_df['coef']]
    fig3.add_trace(go.Bar(
        y=feat_df['feature'],
        x=feat_df['coef'],
        orientation='h',
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color='rgba(255,255,255,0.05)', width=1)
        ),
        text=[f"{v:.3f}" for v in feat_df['coef']],
        textposition='outside',
        textfont=dict(color='#94A3B8', size=10),
    ))
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='Coefficient Value', showgrid=True, gridcolor='rgba(255,255,255,0.05)',
            color='#64748B', zeroline=True, zerolinecolor='rgba(255,255,255,0.2)', zerolinewidth=1.5,
            title_font=dict(color='#64748B')
        ),
        yaxis=dict(showgrid=False, color='#94A3B8', tickfont=dict(size=11)),
        height=600,
        margin=dict(t=20, b=20, l=180, r=80),
        bargap=0.2,
    )

    st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig3, use_container_width=True, key="feat_imp")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
            border-radius:16px;padding:20px;margin-top:16px;'>
            <div style='font-family:Syne;font-size:13px;font-weight:700;color:#EF4444;letter-spacing:1px;margin-bottom:12px;'>
                🔴 HIGH CHURN RISK FACTORS
            </div>
            <div style='font-size:13px;color:#CBD5E1;line-height:2;'>
                • Month-to-month contract<br>
                • Fiber optic internet service<br>
                • Electronic check payment<br>
                • No online security<br>
                • Paperless billing enabled
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
            border-radius:16px;padding:20px;margin-top:16px;'>
            <div style='font-family:Syne;font-size:13px;font-weight:700;color:#10B981;letter-spacing:1px;margin-bottom:12px;'>
                🟢 RETENTION POSITIVE FACTORS
            </div>
            <div style='font-size:13px;color:#CBD5E1;line-height:2;'>
                • Two-year contract<br>
                • Tech support enabled<br>
                • Long tenure (24+ months)<br>
                • Online backup active<br>
                • Mailed check payment
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tenure vs Churn risk simulation
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne;font-size:15px;font-weight:700;color:#F1F5F9;margin-bottom:4px;'>
        Tenure vs. Churn Risk Simulation
    </div>
    <div style='color:#64748B;font-size:12px;margin-bottom:16px;'>How churn probability changes with tenure (current profile)</div>
    """, unsafe_allow_html=True)

    tenure_range = list(range(1, 73))
    probs_tenure = []
    for t in tenure_range:
        temp = inp.copy() if predict_btn else {
            'senior': 0, 'tenure': t, 'monthly': 65, 'gender': 'Male', 'partner': 'No',
            'dependents': 'No', 'phone': 'Yes', 'multilines': 'No', 'internet': 'DSL',
            'security': 'No', 'backup': 'No', 'device': 'No', 'techsupport': 'No',
            'tv': 'No', 'movies': 'No', 'contract': 'Month-to-month',
            'paperless': 'Yes', 'payment': 'Electronic check'
        }
        temp['tenure'] = t
        temp['monthly'] = monthly if predict_btn else 65
        pdf = build_features(temp)
        probs_tenure.append(model.predict_proba(pdf)[0][1] * 100)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=tenure_range, y=probs_tenure,
        mode='lines',
        line=dict(color='#3B82F6', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(59,130,246,0.08)',
        name='Churn Probability'
    ))
    fig4.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.5)",
                   annotation_text="High Risk", annotation_font_color='#EF4444')
    fig4.add_hline(y=40, line_dash="dot", line_color="rgba(245,158,11,0.5)",
                   annotation_text="Medium Risk", annotation_font_color='#F59E0B')
    fig4.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Tenure (Months)', color='#64748B', showgrid=True,
                   gridcolor='rgba(255,255,255,0.04)', title_font=dict(color='#64748B')),
        yaxis=dict(title='Churn Probability (%)', color='#64748B', showgrid=True,
                   gridcolor='rgba(255,255,255,0.04)', title_font=dict(color='#64748B')),
        height=250, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False
    )
    st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig4, use_container_width=True, key="tenure_sim")
    st.markdown("</div>", unsafe_allow_html=True)


# ─── TAB 3: Model Info ────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style='font-family:Syne;font-size:18px;font-weight:700;color:#F1F5F9;margin-bottom:24px;'>
        Model Metadata & Technical Details
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    specs = [
        ("Algorithm", "Logistic Regression", "#3B82F6"),
        ("Regularization", "L2 (Ridge)", "#06B6D4"),
        ("Solver", "LBFGS", "#10B981"),
        ("C Parameter", "1.0", "#F59E0B"),
        ("Max Iterations", "100", "#8B5CF6"),
        ("Features Used", "45", "#EC4899"),
        ("Output Classes", "2 (Stay / Churn)", "#EF4444"),
        ("Fit Intercept", "Yes", "#10B981"),
        ("Warm Start", "No", "#64748B"),
    ]
    for i, (label, val, color) in enumerate(specs):
        col = [m1, m2, m3][i % 3]
        with col:
            st.markdown(f"""
            <div style='background:var(--bg-card);border:1px solid var(--border);
                border-radius:12px;padding:16px;margin-bottom:12px;'>
                <div style='font-size:10px;letter-spacing:2px;color:#475569;font-family:Syne;margin-bottom:6px;'>
                    {label.upper()}
                </div>
                <div style='font-family:Syne;font-size:16px;font-weight:700;color:{color};'>
                    {val}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.15);
        border-radius:16px;padding:24px;margin-top:12px;'>
        <div style='font-family:Syne;font-size:14px;font-weight:700;color:#3B82F6;margin-bottom:14px;letter-spacing:1px;'>
            📖 ABOUT THIS MODEL
        </div>
        <div style='color:#94A3B8;font-size:13px;line-height:1.9;'>
            This Logistic Regression model is trained on a Telecom Customer Churn dataset containing 
            demographic information, account history, and subscribed services. It uses <strong style='color:#CBD5E1;'>45 binary/encoded features</strong>
            derived from one-hot encoding of categorical variables.<br><br>
            The model predicts the probability that a customer will churn (leave the service), 
            enabling proactive retention strategies. An L2 regularization penalty prevents overfitting,
            and the LBFGS solver is well-suited for multi-class logistic regression problems.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature list
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    with st.expander("📋 View All 45 Feature Names"):
        feat_cols = st.columns(3)
        for i, feat in enumerate(model.feature_names_in_):
            with feat_cols[i % 3]:
                st.markdown(f"<div style='font-size:12px;color:#64748B;padding:4px 0;'>• {feat}</div>",
                            unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:40px 0 20px;'>
    <div style='font-family:Syne;font-size:13px;color:#1E293B;letter-spacing:2px;'>
        CHURNGUARD AI · BUILT ON LOGISTIC REGRESSION · TELECOM ANALYTICS
    </div>
</div>
""", unsafe_allow_html=True)