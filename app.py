import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Groundwater Depletion Risk Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #0a0f1e; }
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); }

    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #00ff9f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #8899aa;
        font-size: 1rem;
        letter-spacing: 0.08em;
        margin-bottom: 2rem;
    }

    .risk-card-high {
        background: linear-gradient(135deg, #3d0000, #7a0000);
        border: 1px solid #ff4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(255,68,68,0.3);
    }
    .risk-card-medium {
        background: linear-gradient(135deg, #2d2000, #5a4000);
        border: 1px solid #ffaa00;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(255,170,0,0.3);
    }
    .risk-card-low {
        background: linear-gradient(135deg, #002d1a, #005a34);
        border: 1px solid #00cc66;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(0,204,102,0.3);
    }
    .risk-label {
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-box h3 {
        color: #00d4ff;
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        margin: 0;
    }
    .metric-box p {
        color: #8899aa;
        font-size: 0.8rem;
        margin: 0.2rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .section-header {
        font-family: 'Space Mono', monospace;
        color: #00d4ff;
        font-size: 1.1rem;
        border-bottom: 1px solid rgba(0,212,255,0.3);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .stSidebar { background: #07111d !important; }
    div[data-testid="stSidebarContent"] { background: #07111d; }
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #00ff9f);
        color: #0a0f1e;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
        font-size: 1rem;
        cursor: pointer;
    }
    .stButton>button:hover { opacity: 0.85; }
    label, .stSelectbox label, .stSlider label { color: #aabbcc !important; font-size: 0.85rem !important; }
    .stSelectbox > div > div { background: #0d1b2a; border-color: rgba(0,212,255,0.3); color: #eee; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=7000, seed=42):
    np.random.seed(seed)
    states = ['Tamil Nadu','Karnataka','Maharashtra','Punjab','Gujarat',
              'Rajasthan','Uttar Pradesh','Andhra Pradesh','Madhya Pradesh','Haryana']
    districts = {
        'Tamil Nadu':['Chennai','Coimbatore','Madurai','Salem'],
        'Karnataka':['Mysore','Bengaluru','Hubli','Mangalore'],
        'Maharashtra':['Pune','Nagpur','Nashik','Aurangabad'],
        'Punjab':['Ludhiana','Amritsar','Jalandhar','Patiala'],
        'Gujarat':['Ahmedabad','Surat','Vadodara','Rajkot'],
        'Rajasthan':['Jaipur','Jodhpur','Udaipur','Kota'],
        'Uttar Pradesh':['Lucknow','Kanpur','Agra','Varanasi'],
        'Andhra Pradesh':['Hyderabad','Vijayawada','Visakhapatnam','Guntur'],
        'Madhya Pradesh':['Bhopal','Indore','Gwalior','Jabalpur'],
        'Haryana':['Gurugram','Faridabad','Ambala','Rohtak']
    }
    irrigation_types = ['Borewell','Canal','Drip','Tube well','Sprinkler']
    crop_types = ['Rice','Wheat','Sugarcane','Cotton','Bajra','Maize','Soybean']

    rows = []
    for _ in range(n):
        state = np.random.choice(states)
        district = np.random.choice(districts[state])
        rainfall = np.random.uniform(200, 1400)
        irr_type = np.random.choice(irrigation_types)
        crop = np.random.choice(crop_types)
        temp = np.random.uniform(22, 42)
        soil_moisture = np.random.uniform(0.2, 0.7)

        # Depth influenced by features
        base_depth = 8
        if rainfall < 500: base_depth -= 4
        elif rainfall > 1000: base_depth += 3
        if irr_type in ['Borewell','Tube well']: base_depth -= 3
        elif irr_type == 'Drip': base_depth += 2
        if crop in ['Rice','Sugarcane']: base_depth -= 2
        depth = max(0.5, base_depth + np.random.normal(0, 2.5))

        # Label
        if depth < 5: risk = 'High'
        elif depth < 10: risk = 'Medium'
        else: risk = 'Low'

        rows.append([state, district, rainfall, irr_type, crop,
                     round(depth,2), round(temp,1), round(soil_moisture,3), risk])

    df = pd.DataFrame(rows, columns=['State','District','Rainfall_mm','Irrigation_Type',
                                      'Crop_Type','Groundwater_Depth_m','Temperature_C',
                                      'Soil_Moisture','Risk'])
    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    le_irr  = LabelEncoder()
    le_crop = LabelEncoder()
    le_risk = LabelEncoder()

    df2 = df.copy()
    df2['Irrigation_enc'] = le_irr.fit_transform(df2['Irrigation_Type'])
    df2['Crop_enc']       = le_crop.fit_transform(df2['Crop_Type'])
    df2['Risk_enc']       = le_risk.fit_transform(df2['Risk'])

    features = ['Rainfall_mm','Irrigation_enc','Crop_enc',
                'Groundwater_Depth_m','Temperature_C','Soil_Moisture']
    X = df2[features]
    y = df2['Risk_enc']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Random Forest':    RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting':GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree':    DecisionTreeClassifier(max_depth=8, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'KNN':              KNeighborsClassifier(n_neighbors=7)
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        preds = m.predict(X_test_s)
        results[name] = {
            'model':    m,
            'accuracy': accuracy_score(y_test, preds),
            'preds':    preds,
            'y_test':   y_test
        }

    best_name = max(results, key=lambda k: results[k]['accuracy'])
    return results, best_name, scaler, le_irr, le_crop, le_risk, features, X_test_s, y_test


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_risk(model, scaler, le_irr, le_crop, le_risk, features,
                 rainfall, irr_type, crop_type, depth, temp, soil_moisture):
    irr_enc  = le_irr.transform([irr_type])[0]
    crop_enc = le_crop.transform([crop_type])[0]
    inp = pd.DataFrame([[rainfall, irr_enc, crop_enc, depth, temp, soil_moisture]],
                       columns=features)
    inp_s = scaler.transform(inp)
    pred  = model.predict(inp_s)[0]
    proba = model.predict_proba(inp_s)[0]
    label = le_risk.inverse_transform([pred])[0]
    prob_dict = {le_risk.classes_[i]: proba[i] for i in range(len(proba))}
    return label, prob_dict


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # Hero
    st.markdown('<div class="hero-title">💧 Groundwater Depletion Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AIML Mini Project · PES University · SDG 2 & SDG 15</div>', unsafe_allow_html=True)

    # Load data & models
    with st.spinner("Initialising models..."):
        df = generate_dataset()
        results, best_name, scaler, le_irr, le_crop, le_risk, features, X_test_s, y_test = train_models(df)

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-header">🔧 Input Parameters</div>', unsafe_allow_html=True)

        model_choice = st.selectbox("Select Model", list(results.keys()), index=list(results.keys()).index(best_name))
        st.markdown("---")

        rainfall     = st.slider("Rainfall (mm)", 200, 1400, 750, step=10)
        depth        = st.slider("Groundwater Depth (m)", 0.5, 20.0, 7.0, step=0.5)
        temp         = st.slider("Temperature (°C)", 22, 42, 31)
        soil_moisture= st.slider("Soil Moisture", 0.20, 0.70, 0.45, step=0.01)
        irr_type     = st.selectbox("Irrigation Type", sorted(le_irr.classes_))
        crop_type    = st.selectbox("Crop Type", sorted(le_crop.classes_))

        st.markdown("---")
        predict_btn = st.button("⚡ Predict Risk")

    # ── TABS ─────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Model Performance", "🗂️ Dataset", "📈 EDA"])

    # ────────── TAB 1: PREDICTION ────────────
    with tab1:
        if predict_btn:
            model  = results[model_choice]['model']
            label, prob_dict = predict_risk(model, scaler, le_irr, le_crop, le_risk, features,
                                            rainfall, irr_type, crop_type, depth, temp, soil_moisture)

            emoji  = {"High":"🔴","Medium":"🟠","Low":"🟢"}[label]
            css_cls= {"High":"risk-card-high","Medium":"risk-card-medium","Low":"risk-card-low"}[label]
            color  = {"High":"#ff4444","Medium":"#ffaa00","Low":"#00cc66"}[label]

            st.markdown(f"""
            <div class="{css_cls}">
                <div style="font-size:1rem; color:#aaa; font-family:'Space Mono',monospace; margin-bottom:0.3rem;">DEPLETION RISK LEVEL</div>
                <div class="risk-label" style="color:{color};">{emoji} {label.upper()} RISK</div>
                <div style="color:#aaa; margin-top:0.5rem; font-size:0.9rem;">
                    {"⚠️ Immediate action needed – aquifer critically low" if label=='High'
                     else "⚡ Monitor closely – moderate stress detected" if label=='Medium'
                     else "✅ Stable groundwater levels – continue sustainable practices"}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge chart
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="section-header">Risk Probability Breakdown</div>', unsafe_allow_html=True)
                labels = list(prob_dict.keys())
                values = list(prob_dict.values())
                colors_map = {'High':'#ff4444','Medium':'#ffaa00','Low':'#00cc66'}
                bar_colors = [colors_map.get(l,'#888') for l in labels]

                fig_bar = go.Figure(go.Bar(
                    x=[f"{v*100:.1f}%" for v in values],
                    y=labels,
                    orientation='h',
                    marker_color=bar_colors,
                    text=[f"{v*100:.1f}%" for v in values],
                    textposition='inside',
                ))
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#aabbcc'),
                    height=220,
                    margin=dict(l=10,r=10,t=10,b=10),
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
                summary = {
                    "Rainfall": f"{rainfall} mm",
                    "Groundwater Depth": f"{depth} m",
                    "Temperature": f"{temp} °C",
                    "Soil Moisture": f"{soil_moisture:.2f}",
                    "Irrigation Type": irr_type,
                    "Crop Type": crop_type,
                    "Model Used": model_choice
                }
                for k, v in summary.items():
                    st.markdown(f"<span style='color:#aaa;font-size:0.85rem;'>{k}:</span> "
                                f"<span style='color:#00d4ff;font-weight:600;'>{v}</span><br>",
                                unsafe_allow_html=True)

            # Feature importance (if RF or GB)
            if hasattr(model, 'feature_importances_'):
                st.markdown('<div class="section-header" style="margin-top:1rem;">Feature Importance</div>', unsafe_allow_html=True)
                feat_names = ['Rainfall','Irrigation','Crop Type','GW Depth','Temperature','Soil Moisture']
                imp = model.feature_importances_
                fi_df = pd.DataFrame({'Feature':feat_names,'Importance':imp}).sort_values('Importance', ascending=True)
                fig_fi = go.Figure(go.Bar(
                    x=fi_df['Importance'], y=fi_df['Feature'],
                    orientation='h',
                    marker=dict(color=fi_df['Importance'], colorscale='Teal'),
                    text=[f"{v:.3f}" for v in fi_df['Importance']],
                    textposition='outside'
                ))
                fig_fi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#aabbcc'), height=280,
                    margin=dict(l=10,r=10,t=10,b=10),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False),
                    yaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("👈 Set parameters in the sidebar and click **⚡ Predict Risk** to get started.")

    # ────────── TAB 2: MODEL PERFORMANCE ─────
    with tab2:
        st.markdown('<div class="section-header">Model Accuracy Comparison</div>', unsafe_allow_html=True)

        model_names = list(results.keys())
        accuracies  = [results[m]['accuracy']*100 for m in model_names]

        col1, col2, col3, col4, col5 = st.columns(5)
        for col, name in zip([col1,col2,col3,col4,col5], model_names):
            acc = results[name]['accuracy']*100
            with col:
                st.markdown(f"""<div class="metric-box">
                    <h3>{acc:.1f}%</h3><p>{name}</p></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_acc = go.Figure(go.Bar(
            x=model_names, y=accuracies,
            marker=dict(color=accuracies, colorscale='Teal', showscale=False),
            text=[f"{a:.1f}%" for a in accuracies], textposition='outside'
        ))
        fig_acc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aabbcc'), height=350,
            yaxis=dict(range=[0,105], gridcolor='rgba(255,255,255,0.05)', title='Accuracy (%)'),
            xaxis=dict(showgrid=False),
            margin=dict(l=10,r=10,t=20,b=10)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # Confusion matrix for selected model
        st.markdown(f'<div class="section-header">Confusion Matrix – {model_choice}</div>', unsafe_allow_html=True)
        classes = le_risk.classes_
        preds_sel = results[model_choice]['preds']
        y_test_sel = results[model_choice]['y_test']
        cm = confusion_matrix(y_test_sel, preds_sel)
        class_labels = [le_risk.inverse_transform([i])[0] for i in range(len(classes))]

        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=class_labels, y=class_labels,
            colorscale='Teal',
            text=cm, texttemplate="%{text}",
            showscale=True
        ))
        fig_cm.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#aabbcc'), height=350,
            xaxis_title='Predicted', yaxis_title='Actual',
            margin=dict(l=10,r=10,t=20,b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ────────── TAB 3: DATASET ───────────────
    with tab3:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip([c1,c2,c3,c4], [
            ("Total Records", len(df)),
            ("Features", len(df.columns)-1),
            ("High Risk %", f"{(df['Risk']=='High').mean()*100:.1f}%"),
            ("States Covered", df['State'].nunique())
        ]):
            with col:
                st.markdown(f"""<div class="metric-box"><h3>{val}</h3><p>{label}</p></div>""",
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Filter
        state_filter = st.multiselect("Filter by State", sorted(df['State'].unique()),
                                       default=sorted(df['State'].unique())[:3])
        if state_filter:
            df_show = df[df['State'].isin(state_filter)]
        else:
            df_show = df
        st.dataframe(df_show.head(200).style.applymap(
            lambda v: 'color: #ff4444' if v=='High' else ('color: #ffaa00' if v=='Medium'
                      else ('color: #00cc66' if v=='Low' else '')),
            subset=['Risk']
        ), use_container_width=True, height=400)

    # ────────── TAB 4: EDA ───────────────────
    with tab4:
        st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Risk distribution
            risk_counts = df['Risk'].value_counts().reset_index()
            risk_counts.columns = ['Risk','Count']
            fig_pie = px.pie(risk_counts, names='Risk', values='Count',
                             color='Risk',
                             color_discrete_map={'High':'#ff4444','Medium':'#ffaa00','Low':'#00cc66'},
                             title='Risk Level Distribution')
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#aabbcc'),
                                  height=320, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Rainfall vs Depth scatter
            sample = df.sample(500, random_state=1)
            fig_sc = px.scatter(sample, x='Rainfall_mm', y='Groundwater_Depth_m',
                                color='Risk',
                                color_discrete_map={'High':'#ff4444','Medium':'#ffaa00','Low':'#00cc66'},
                                title='Rainfall vs Groundwater Depth',
                                labels={'Rainfall_mm':'Rainfall (mm)','Groundwater_Depth_m':'GW Depth (m)'},
                                opacity=0.7)
            fig_sc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#aabbcc'), height=320,
                                  xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                                  yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                                  margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_sc, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            # Risk by Irrigation Type
            irr_risk = df.groupby(['Irrigation_Type','Risk']).size().reset_index(name='Count')
            fig_irr = px.bar(irr_risk, x='Irrigation_Type', y='Count', color='Risk',
                             color_discrete_map={'High':'#ff4444','Medium':'#ffaa00','Low':'#00cc66'},
                             title='Risk by Irrigation Type', barmode='group')
            fig_irr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='#aabbcc'), height=320,
                                   xaxis=dict(showgrid=False),
                                   yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                                   margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_irr, use_container_width=True)

        with col4:
            # GW Depth distribution by State
            fig_box = px.box(df, x='State', y='Groundwater_Depth_m', color='Risk',
                              color_discrete_map={'High':'#ff4444','Medium':'#ffaa00','Low':'#00cc66'},
                              title='GW Depth by State')
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='#aabbcc'), height=320,
                                   xaxis=dict(showgrid=False, tickangle=30),
                                   yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                                   margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_box, use_container_width=True)

        # Correlation heatmap
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        num_cols = ['Rainfall_mm','Groundwater_Depth_m','Temperature_C','Soil_Moisture']
        corr = df[num_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu', zmid=0,
            text=np.round(corr.values,2), texttemplate="%{text}"
        ))
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#aabbcc'),
                                height=350, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig_corr, use_container_width=True)


if __name__ == "__main__":
    main()
