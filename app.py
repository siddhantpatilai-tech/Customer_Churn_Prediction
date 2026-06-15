"""
Customer Churn Prediction — Full Streamlit App
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import io
import os
warnings.filterwarnings('ignore')

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }
    .block-container { padding: 1.5rem 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #12141f 100%);
        border-right: 1px solid #2d3250;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2138 0%, #252844 100%);
        border: 1px solid #3d4275;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] { color: #8892b0 !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: #ccd6f6 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }

    /* Headers */
    h1 { color: #64ffda !important; font-family: 'Courier New', monospace; }
    h2, h3 { color: #ccd6f6 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d2e;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8892b0;
        border-radius: 8px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2d3250, #424769) !important;
        color: #64ffda !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #64ffda, #00b4d8);
        color: #0f1117;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.5rem 2rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(100, 255, 218, 0.4);
    }

    /* Selectbox / Slider */
    .stSelectbox > div > div { background: #1e2138; border-color: #3d4275; color: #ccd6f6; }
    .stSlider > div > div { color: #64ffda; }

    /* Churn badge */
    .churn-yes {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; border-radius: 10px; padding: 1.2rem 1.5rem;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(255,107,107,0.4);
    }
    .churn-no {
        background: linear-gradient(135deg, #64ffda, #26de81);
        color: #0f1117; border-radius: 10px; padding: 1.2rem 1.5rem;
        text-align: center; font-size: 1.4rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(100,255,218,0.4);
    }
    .section-title {
        background: linear-gradient(90deg, #1e2138, transparent);
        border-left: 3px solid #64ffda;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #ccd6f6;
        font-weight: 600;
    }
    .info-card {
        background: #1e2138;
        border: 1px solid #3d4275;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ARTIFACTS + DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, 'artifacts.pkl'), 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_raw_data():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, 'telecom_churn.csv'))
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df

artifacts = load_artifacts()
raw_df    = load_raw_data()

models        = artifacts['models']
scaler        = artifacts['scaler']
results       = artifacts['results']
feature_names = artifacts['feature_names']
best_name     = artifacts['best_model_name']
feat_imp      = artifacts['feat_importances']

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 ChurnGuard AI")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Overview Dashboard",
        "🔍 EDA & Insights",
        "🤖 Model Evaluation",
        "👤 Single Prediction",
        "📂 Bulk Prediction"
    ])

    st.markdown("---")
    st.markdown(f"""
    <div class='info-card'>
        <b style='color:#64ffda'>Best Model</b><br>
        <span style='color:#ccd6f6'>{best_name}</span><br><br>
        <b style='color:#64ffda'>AUC Score</b><br>
        <span style='color:#ccd6f6; font-size:1.3rem'>{results[best_name]['roc_auc']}%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-card' style='margin-top:0.5rem'>
        <b style='color:#64ffda'>Dataset</b><br>
        <span style='color:#8892b0'>{len(raw_df):,} customers</span><br>
        <span style='color:#8892b0'>21 features</span><br>
        <span style='color:#8892b0'>Churn rate: {(raw_df['Churn']=='Yes').mean()*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER: PREPROCESS SINGLE ROW
# ─────────────────────────────────────────────
def preprocess_input(inp: dict) -> np.ndarray:
    df = pd.DataFrame([inp])
    df['TotalCharges']      = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['AvgMonthlySpend']   = df['TotalCharges'] / (df['tenure'] + 1)
    df['NumAddonServices']  = (
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup']   == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport']    == 'Yes').astype(int) +
        (df['StreamingTV']    == 'Yes').astype(int) +
        (df['StreamingMovies']== 'Yes').astype(int)
    )
    df['IsLongTermContract'] = (df['Contract'] != 'Month-to-month').astype(int)
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,72],
                               labels=['0-12m','13-24m','25-48m','49-72m'], include_lowest=True)

    binary_map = {'Yes':1,'No':0,'Male':1,'Female':0}
    for col in ['gender','Partner','Dependents','PhoneService','PaperlessBilling']:
        df[col] = df[col].map(binary_map)

    multi_cat_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                      'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                      'Contract','PaymentMethod','TenureGroup']
    df = pd.get_dummies(df, columns=multi_cat_cols)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Align to training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    df = df.fillna(0)

    scaled = scaler.transform(df)
    return scaled

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    'primary':   '#64ffda',
    'secondary': '#00b4d8',
    'danger':    '#ff6b6b',
    'warning':   '#ffd166',
    'bg':        '#1e2138',
    'card':      '#252844',
    'text':      '#ccd6f6',
}
PLOTLY_TEMPLATE = dict(
    plot_bgcolor ='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#ccd6f6', family='Inter, sans-serif'),
    xaxis=dict(gridcolor='#2d3250', linecolor='#3d4275'),
    yaxis=dict(gridcolor='#2d3250', linecolor='#3d4275'),
)


# ═══════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════
if page == "🏠 Overview Dashboard":
    st.markdown("# 📡 ChurnGuard AI")
    st.markdown("#### End-to-End Customer Churn Prediction System · Telecom Dataset · 6 ML Models")
    st.markdown("---")

    # KPI ROW
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total Customers",   f"{len(raw_df):,}")
    with c2: st.metric("Churned",           f"{(raw_df['Churn']=='Yes').sum():,}", delta=f"{(raw_df['Churn']=='Yes').mean()*100:.1f}%")
    with c3: st.metric("Retained",          f"{(raw_df['Churn']=='No').sum():,}")
    with c4: st.metric("Best Model AUC",    f"{results[best_name]['roc_auc']}%", delta=best_name)
    with c5: st.metric("Best Model F1",     f"{results[best_name]['f1']}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>📊 Churn Distribution</div>", unsafe_allow_html=True)
        cnt = raw_df['Churn'].value_counts()
        fig = go.Figure(go.Pie(
            labels=['Retained', 'Churned'],
            values=[cnt['No'], cnt['Yes']],
            hole=0.55,
            marker_colors=[COLORS['primary'], COLORS['danger']],
            textfont_size=14,
        ))
        fig.update_layout(**PLOTLY_TEMPLATE, height=300,
                          annotations=[dict(text='<b>Churn</b>', font_size=16, showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>📈 Model Comparison — ROC AUC</div>", unsafe_allow_html=True)
        model_names = list(results.keys())
        aucs = [results[m]['roc_auc'] for m in model_names]
        colors = [COLORS['primary'] if m == best_name else COLORS['secondary'] for m in model_names]
        fig = go.Figure(go.Bar(
            x=aucs, y=model_names,
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0)')),
            text=[f"{v}%" for v in aucs],
            textposition='outside',
        ))
        fig.update_layout(**PLOTLY_TEMPLATE, height=300, xaxis_range=[60, 80])
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-title'>💰 Monthly Charges Distribution</div>", unsafe_allow_html=True)
        fig = go.Figure()
        for label, color in [('No', COLORS['primary']), ('Yes', COLORS['danger'])]:
            fig.add_trace(go.Histogram(
                x=raw_df[raw_df['Churn'] == label]['MonthlyCharges'],
                name=f'Churn={label}',
                marker_color=color,
                opacity=0.7,
                nbinsx=30,
            ))
        fig.update_layout(**PLOTLY_TEMPLATE, height=280, barmode='overlay',
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>📅 Tenure vs Churn</div>", unsafe_allow_html=True)
        raw_df['TenureGroup'] = pd.cut(raw_df['tenure'], bins=[0,12,24,48,72],
                                       labels=['0-12m','13-24m','25-48m','49-72m'], include_lowest=True)
        tg_churn = raw_df.groupby(['TenureGroup','Churn']).size().unstack(fill_value=0)
        tg_churn['rate'] = tg_churn['Yes'] / (tg_churn['Yes'] + tg_churn['No']) * 100
        fig = go.Figure(go.Bar(
            x=tg_churn.index.astype(str),
            y=tg_churn['rate'],
            marker_color=COLORS['warning'],
            text=[f"{v:.1f}%" for v in tg_churn['rate']],
            textposition='outside',
        ))
        fig.update_layout(**PLOTLY_TEMPLATE, height=280, yaxis_title='Churn Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("<div class='section-title'>🔑 Top 15 Feature Importances (Random Forest)</div>", unsafe_allow_html=True)
    fi_df = pd.DataFrame({'Feature': list(feat_imp.keys()), 'Importance': list(feat_imp.values())}).sort_values('Importance')
    fig = go.Figure(go.Bar(
        x=fi_df['Importance'], y=fi_df['Feature'],
        orientation='h',
        marker=dict(
            color=fi_df['Importance'],
            colorscale=[[0,'#1e2138'],[0.5,'#00b4d8'],[1,'#64ffda']],
            showscale=False,
        ),
    ))
    fig.update_layout(**PLOTLY_TEMPLATE, height=400)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════
elif page == "🔍 EDA & Insights":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset Preview", "🔢 Distributions", "🔗 Correlations", "📊 Service Analysis"])

    with tab1:
        st.subheader("Raw Dataset (first 50 rows)")
        st.dataframe(raw_df.head(50), use_container_width=True, height=400)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Shape**")
            st.code(f"{raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
        with c2:
            st.markdown("**Missing values**")
            st.code(f"TotalCharges: {raw_df['TotalCharges'].isna().sum()} rows (imputed)")
        with c3:
            st.markdown("**Numeric Stats**")
        st.dataframe(raw_df.describe().round(2), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Gender vs Churn</div>", unsafe_allow_html=True)
            gv = raw_df.groupby(['gender','Churn']).size().reset_index(name='count')
            fig = px.bar(gv, x='gender', y='count', color='Churn',
                         color_discrete_map={'No': COLORS['primary'], 'Yes': COLORS['danger']},
                         barmode='group', template='plotly_dark')
            fig.update_layout(**PLOTLY_TEMPLATE, height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("<div class='section-title'>Senior Citizen vs Churn</div>", unsafe_allow_html=True)
            sv = raw_df.groupby(['SeniorCitizen','Churn']).size().reset_index(name='count')
            sv['SeniorCitizen'] = sv['SeniorCitizen'].map({0:'Non-Senior',1:'Senior'})
            fig = px.bar(sv, x='SeniorCitizen', y='count', color='Churn',
                         color_discrete_map={'No': COLORS['primary'], 'Yes': COLORS['danger']},
                         barmode='group', template='plotly_dark')
            fig.update_layout(**PLOTLY_TEMPLATE, height=300)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("<div class='section-title'>Contract Type vs Churn</div>", unsafe_allow_html=True)
            cv = raw_df.groupby(['Contract','Churn']).size().reset_index(name='count')
            fig = px.bar(cv, x='Contract', y='count', color='Churn',
                         color_discrete_map={'No': COLORS['primary'], 'Yes': COLORS['danger']},
                         barmode='stack', template='plotly_dark')
            fig.update_layout(**PLOTLY_TEMPLATE, height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown("<div class='section-title'>Payment Method vs Churn</div>", unsafe_allow_html=True)
            pv = raw_df.groupby(['PaymentMethod','Churn']).size().reset_index(name='count')
            fig = px.bar(pv, x='PaymentMethod', y='count', color='Churn',
                         color_discrete_map={'No': COLORS['primary'], 'Yes': COLORS['danger']},
                         barmode='stack', template='plotly_dark')
            fig.update_layout(**PLOTLY_TEMPLATE, height=300, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>Monthly Charges vs Total Charges (colored by Churn)</div>", unsafe_allow_html=True)
        fig = px.scatter(raw_df, x='MonthlyCharges', y='TotalCharges', color='Churn',
                         color_discrete_map={'No': COLORS['primary'], 'Yes': COLORS['danger']},
                         opacity=0.5, template='plotly_dark')
        fig.update_layout(**PLOTLY_TEMPLATE, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-title'>Correlation Heatmap (Numeric Features)</div>", unsafe_allow_html=True)
        num_df = raw_df[['tenure','MonthlyCharges','TotalCharges','SeniorCitizen']].copy()
        num_df['Churn_bin'] = (raw_df['Churn'] == 'Yes').astype(int)
        corr = num_df.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale=[[0,'#ff6b6b'],[0.5,'#1e2138'],[1,'#64ffda']],
            text=corr.round(2).values,
            texttemplate='%{text}',
            showscale=True,
        ))
        fig.update_layout(**PLOTLY_TEMPLATE, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        service_cols = ['PhoneService','MultipleLines','InternetService',
                        'OnlineSecurity','TechSupport','StreamingTV','StreamingMovies']
        st.markdown("<div class='section-title'>Churn Rate by Service Subscription</div>", unsafe_allow_html=True)
        rates = []
        for col in service_cols:
            for val in raw_df[col].unique():
                sub = raw_df[raw_df[col] == val]
                rate = (sub['Churn'] == 'Yes').mean() * 100
                rates.append({'Service': col, 'Value': str(val), 'ChurnRate': round(rate,1)})
        rates_df = pd.DataFrame(rates)
        fig = px.bar(rates_df, x='Service', y='ChurnRate', color='Value',
                     barmode='group', template='plotly_dark',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(**PLOTLY_TEMPLATE, height=400, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 3 — MODEL EVALUATION
# ═══════════════════════════════════════════════
elif page == "🤖 Model Evaluation":
    st.markdown("# 🤖 Model Evaluation & Comparison")
    st.markdown("---")

    # Metrics table
    st.markdown("<div class='section-title'>📊 All Models — Metrics Summary</div>", unsafe_allow_html=True)
    metrics_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy (%)': r['accuracy'],
            'Precision (%)': r['precision'],
            'Recall (%)': r['recall'],
            'F1 Score (%)': r['f1'],
            'ROC-AUC (%)': r['roc_auc'],
        }
        for name, r in results.items()
    ]).sort_values('ROC-AUC (%)', ascending=False).reset_index(drop=True)

    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color: rgba(100,255,218,0.15); color:#64ffda; font-weight:bold'
                if v else '' for v in is_max]

    st.dataframe(
        metrics_df.style.apply(highlight_best, subset=['Accuracy (%)','Precision (%)','Recall (%)','F1 Score (%)','ROC-AUC (%)']),
        use_container_width=True, height=280
    )

    # Radar chart
    st.markdown("<div class='section-title'>🕸️ Model Radar Chart</div>", unsafe_allow_html=True)
    categories = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
    fig = go.Figure()
    palette = [COLORS['primary'], COLORS['secondary'], COLORS['warning'],
               COLORS['danger'], '#b07efb', '#ff9f43']
    for i, (name, r) in enumerate(results.items()):
        vals = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['roc_auc']]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill='toself',
            name=name,
            line_color=palette[i % len(palette)],
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(30,33,56,0.7)',
            radialaxis=dict(visible=True, range=[0,100], gridcolor='#3d4275', color='#8892b0'),
            angularaxis=dict(gridcolor='#3d4275', color='#ccd6f6'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccd6f6'),
        height=450,
        legend=dict(bgcolor='rgba(30,33,56,0.7)'),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ROC curves
    st.markdown("<div class='section-title'>📈 ROC Curves — All Models</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                             line=dict(dash='dash', color='#8892b0'), name='Random (AUC=50%)'))
    for i, (name, r) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=r['fpr'], y=r['tpr'],
            mode='lines',
            name=f"{name} ({r['roc_auc']}%)",
            line=dict(color=palette[i % len(palette)], width=2 if name != best_name else 3),
        ))
    fig.update_layout(**PLOTLY_TEMPLATE, height=420,
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      legend=dict(bgcolor='rgba(30,33,56,0.7)'))
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    st.markdown("<div class='section-title'>🔲 Confusion Matrices</div>", unsafe_allow_html=True)
    model_list = list(results.keys())
    cols_cm = st.columns(3)
    for idx, name in enumerate(model_list):
        cm = np.array(results[name]['cm'])
        with cols_cm[idx % 3]:
            st.markdown(f"**{name}**")
            fig = go.Figure(go.Heatmap(
                z=cm, x=['Pred: No','Pred: Yes'], y=['True: No','True: Yes'],
                colorscale=[[0,'#1e2138'],[1,'#64ffda']],
                text=cm, texttemplate='<b>%{text}</b>',
                showscale=False,
            ))
            fig.update_layout(**PLOTLY_TEMPLATE, height=220, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4 — SINGLE PREDICTION
# ═══════════════════════════════════════════════
elif page == "👤 Single Prediction":
    st.markdown("# 👤 Single Customer Churn Prediction")
    st.markdown("---")

    selected_model = st.selectbox("🤖 Select Model", list(models.keys()),
                                  index=list(models.keys()).index(best_name))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>👤 Customer Demographics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: gender     = st.selectbox("Gender",        ['Male','Female'])
    with c2: senior     = st.selectbox("Senior Citizen",['No','Yes'])
    with c3: partner    = st.selectbox("Partner",       ['Yes','No'])
    with c4: dependents = st.selectbox("Dependents",    ['Yes','No'])
    tenure = st.slider("Tenure (months)", 0, 72, 24)

    st.markdown("<div class='section-title'>📞 Phone & Internet</div>", unsafe_allow_html=True)
    c5, c6, c7 = st.columns(3)
    with c5: phone_svc   = st.selectbox("Phone Service",   ['Yes','No'])
    with c6: multi_lines = st.selectbox("Multiple Lines",  ['No','Yes','No phone service'])
    with c7: internet    = st.selectbox("Internet Service",['DSL','Fiber optic','No'])

    st.markdown("<div class='section-title'>🛡️ Add-on Services</div>", unsafe_allow_html=True)
    c8, c9, c10 = st.columns(3)
    with c8:
        online_sec = st.selectbox("Online Security",    ['No','Yes','No internet service'])
        online_bk  = st.selectbox("Online Backup",      ['No','Yes','No internet service'])
    with c9:
        device_prot = st.selectbox("Device Protection", ['No','Yes','No internet service'])
        tech_sup    = st.selectbox("Tech Support",       ['No','Yes','No internet service'])
    with c10:
        stream_tv   = st.selectbox("Streaming TV",       ['No','Yes','No internet service'])
        stream_mv   = st.selectbox("Streaming Movies",   ['No','Yes','No internet service'])

    st.markdown("<div class='section-title'>💳 Billing</div>", unsafe_allow_html=True)
    c11, c12, c13 = st.columns(3)
    with c11: contract = st.selectbox("Contract", ['Month-to-month','One year','Two year'])
    with c12: paperless = st.selectbox("Paperless Billing", ['Yes','No'])
    with c13: payment   = st.selectbox("Payment Method",
                            ['Electronic check','Mailed check',
                             'Bank transfer (automatic)','Credit card (automatic)'])
    c14, c15 = st.columns(2)
    with c14: monthly_charges = st.slider("Monthly Charges ($)", 18.0, 119.0, 65.0, step=0.5)
    with c15: total_charges   = st.slider("Total Charges ($)",   0.0, 8700.0, float(monthly_charges*tenure), step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

    if predict_btn:
        inp = {
            'gender': gender, 'SeniorCitizen': 1 if senior=='Yes' else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone_svc, 'MultipleLines': multi_lines,
            'InternetService': internet, 'OnlineSecurity': online_sec,
            'OnlineBackup': online_bk, 'DeviceProtection': device_prot,
            'TechSupport': tech_sup, 'StreamingTV': stream_tv,
            'StreamingMovies': stream_mv, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
        }
        X_inp = preprocess_input(inp)
        model = models[selected_model]
        pred  = model.predict(X_inp)[0]
        prob  = model.predict_proba(X_inp)[0]

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns([1,2,1])
        with col_r2:
            if pred == 1:
                st.markdown(f"""
                <div class='churn-yes'>
                    ⚠️ LIKELY TO CHURN<br>
                    <span style='font-size:0.9rem'>Churn Probability: {prob[1]*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='churn-no'>
                    ✅ LIKELY TO STAY<br>
                    <span style='font-size:0.85rem'>Retention Probability: {prob[0]*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=prob[1]*100,
                title=dict(text='Churn Probability', font=dict(color='#ccd6f6')),
                number=dict(suffix='%', font=dict(color='#ccd6f6', size=36)),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor='#8892b0'),
                    bar=dict(color=COLORS['danger'] if prob[1]>0.5 else COLORS['primary']),
                    bgcolor='#1e2138',
                    borderwidth=0,
                    steps=[
                        dict(range=[0,40],  color='rgba(100,255,218,0.15)'),
                        dict(range=[40,65], color='rgba(255,209,102,0.15)'),
                        dict(range=[65,100],color='rgba(255,107,107,0.15)'),
                    ],
                    threshold=dict(line=dict(color='white',width=3), value=50),
                ),
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ccd6f6', height=280)
            st.plotly_chart(fig, use_container_width=True)

        with c_g2:
            fig = go.Figure(go.Bar(
                x=['Retain','Churn'],
                y=[prob[0]*100, prob[1]*100],
                marker_color=[COLORS['primary'], COLORS['danger']],
                text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
                textposition='outside',
            ))
            fig.update_layout(**PLOTLY_TEMPLATE, height=280, yaxis_range=[0,110],
                              title='Class Probabilities')
            st.plotly_chart(fig, use_container_width=True)

        # Retention tips
        st.markdown("<div class='section-title'>💡 Retention Recommendations</div>", unsafe_allow_html=True)
        tips = []
        if contract == 'Month-to-month':    tips.append("📄 Offer a discounted **1-year or 2-year contract** to improve retention.")
        if internet == 'Fiber optic':       tips.append("🌐 Fiber customers churn more — consider a **loyalty discount or service upgrade**.")
        if online_sec == 'No':              tips.append("🔐 Offer **Online Security** add-on — it significantly reduces churn probability.")
        if tech_sup == 'No':                tips.append("🛠️ Offer **Tech Support** subscription — customers with it churn less.")
        if payment == 'Electronic check':   tips.append("💳 Encourage **auto-payment** via bank transfer or credit card.")
        if tenure < 12:                     tips.append("🎁 Customer is new (<1 year) — provide a **welcome loyalty reward**.")
        if not tips:
            tips.append("✅ Customer profile is low-risk. Keep offering consistent service quality.")
        for tip in tips:
            st.markdown(f"- {tip}")


# ═══════════════════════════════════════════════
# PAGE 5 — BULK PREDICTION
# ═══════════════════════════════════════════════
elif page == "📂 Bulk Prediction":
    st.markdown("# 📂 Bulk Churn Prediction via File Upload")
    st.markdown("---")

    selected_model = st.selectbox("🤖 Select Model", list(models.keys()),
                                  index=list(models.keys()).index(best_name))

    st.markdown("""
    <div class='info-card'>
        <b style='color:#64ffda'>📌 Instructions</b><br>
        Upload a CSV or Excel file with the same columns as the telecom dataset.<br>
        The app will predict churn for each customer and provide a downloadable result file.
    </div>
    """, unsafe_allow_html=True)

    # Download sample
    sample = raw_df.drop(columns=['Churn']).head(10)
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button("⬇️ Download Sample CSV Template", buf.getvalue(),
                       file_name='sample_customers.csv', mime='text/csv')

    uploaded = st.file_uploader("📤 Upload CSV / Excel", type=['csv','xlsx'])

    if uploaded:
        try:
            if uploaded.name.endswith('.xlsx'):
                up_df = pd.read_excel(uploaded)
            else:
                up_df = pd.read_csv(uploaded)

            st.markdown(f"**Uploaded:** {len(up_df)} rows × {len(up_df.columns)} columns")
            st.dataframe(up_df.head(10), use_container_width=True)

            if st.button("🚀 Run Bulk Prediction", use_container_width=True):
                with st.spinner("Running predictions..."):
                    preds, probs_yes = [], []
                    for _, row in up_df.iterrows():
                        try:
                            inp = row.to_dict()
                            X_row = preprocess_input(inp)
                            model = models[selected_model]
                            p = model.predict(X_row)[0]
                            prob = model.predict_proba(X_row)[0][1]
                            preds.append('Yes' if p == 1 else 'No')
                            probs_yes.append(round(prob*100, 2))
                        except Exception:
                            preds.append('Error')
                            probs_yes.append(0.0)

                result_df = up_df.copy()
                result_df['Predicted_Churn']       = preds
                result_df['Churn_Probability (%)'] = probs_yes

                # Risk bucket
                def risk(p):
                    if p >= 65:   return '🔴 High'
                    elif p >= 40: return '🟡 Medium'
                    else:         return '🟢 Low'
                result_df['Risk_Level'] = result_df['Churn_Probability (%)'].apply(risk)

                st.markdown("---")
                st.subheader("📋 Prediction Results")
                st.dataframe(result_df, use_container_width=True, height=400)

                # Summary stats
                total = len(result_df)
                churned = (result_df['Predicted_Churn'] == 'Yes').sum()
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Total Customers",  total)
                with c2: st.metric("Predicted Churn",  churned, delta=f"{churned/total*100:.1f}%")
                with c3: st.metric("High Risk",        (result_df['Risk_Level']=='🔴 High').sum())
                with c4: st.metric("Low Risk",         (result_df['Risk_Level']=='🟢 Low').sum())

                # Pie
                col_p, col_b = st.columns(2)
                with col_p:
                    cnt = result_df['Predicted_Churn'].value_counts()
                    fig = go.Figure(go.Pie(
                        labels=['Retain','Churn'],
                        values=[cnt.get('No',0), cnt.get('Yes',0)],
                        hole=0.5,
                        marker_colors=[COLORS['primary'], COLORS['danger']],
                    ))
                    fig.update_layout(**PLOTLY_TEMPLATE, height=280,
                                      title='Churn Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    risk_cnt = result_df['Risk_Level'].value_counts()
                    fig = go.Figure(go.Bar(
                        x=risk_cnt.index.tolist(),
                        y=risk_cnt.values.tolist(),
                        marker_color=[COLORS['danger'], COLORS['warning'], COLORS['primary']],
                        text=risk_cnt.values.tolist(),
                        textposition='outside',
                    ))
                    fig.update_layout(**PLOTLY_TEMPLATE, height=280, title='Risk Level Breakdown')
                    st.plotly_chart(fig, use_container_width=True)

                # Download
                csv_buf = io.StringIO()
                result_df.to_csv(csv_buf, index=False)
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    csv_buf.getvalue(),
                    file_name='churn_predictions.csv',
                    mime='text/csv',
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("""
<br><br>
<div style='text-align:center; color:#3d4275; font-size:0.8rem; border-top:1px solid #2d3250; padding-top:1rem'>
    ChurnGuard AI · Built with Streamlit · Scikit-learn · Plotly · Telecom Churn Dataset
</div>
""", unsafe_allow_html=True)
