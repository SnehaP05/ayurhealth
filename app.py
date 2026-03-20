import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AyurAI — Holistic Health Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0f1117;
    --surface:   #1a1e2a;
    --card:      #20263a;
    --border:    #2e3552;
    --gold:      #c9a84c;
    --gold-dim:  #8a6f2e;
    --green:     #4caf7d;
    --saffron:   #e8894a;
    --text:      #e8e4d9;
    --muted:     #8a8fa8;
    --vata:      #7c9ecf;
    --pitta:     #e07b5a;
    --kapha:     #6dbf82;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f1117 0%, #1a1e2a 50%, #12180f 100%);
    border-bottom: 1px solid var(--border);
    padding: 3rem 2rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(201,168,76,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--gold);
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--muted);
    margin-top: 0.6rem;
    font-weight: 300;
    letter-spacing: 0.04em;
}
.mandala {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 1rem;
    filter: sepia(0.4) saturate(2) hue-rotate(10deg);
}

/* ── Section headings ── */
.section-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--gold);
    border-left: 3px solid var(--gold);
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem;
    font-style: italic;
}

/* ── Cards ── */
.ayur-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.ayur-card:hover { border-color: var(--gold-dim); }

/* ── Dosha badge ── */
.dosha-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.dosha-vata  { background: rgba(124,158,207,0.15); color: var(--vata);   border: 1px solid var(--vata);   }
.dosha-pitta { background: rgba(224,123,90,0.15);  color: var(--pitta);  border: 1px solid var(--pitta);  }
.dosha-kapha { background: rgba(109,191,130,0.15); color: var(--kapha);  border: 1px solid var(--kapha);  }

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-tile {
    flex: 1; min-width: 120px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-tile .val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: var(--gold);
}
.metric-tile .lbl { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }

/* ── Result block ── */
.result-disease {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    color: var(--saffron);
    margin: 0.5rem 0;
}
.herb-chip {
    display: inline-block;
    background: rgba(76,175,125,0.12);
    color: var(--green);
    border: 1px solid rgba(76,175,125,0.3);
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.82rem;
    margin: 0.2rem;
}
.tip-item {
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-size: 0.92rem;
}
.tip-item:last-child { border-bottom: none; }
.tip-icon { color: var(--gold); margin-right: 0.5rem; }

/* ── Wellness footer ── */
.wellness-msg {
    text-align: center;
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.3rem;
    color: var(--gold);
    padding: 2rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

/* ── Streamlit overrides ── */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select,
textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
div[data-testid="stSlider"] > div { color: var(--gold) !important; }
div.stButton > button {
    background: var(--gold) !important;
    color: #0f1117 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2.5rem !important;
    font-size: 1rem !important;
    width: 100% !important;
    letter-spacing: 0.04em;
    transition: opacity 0.2s !important;
}
div.stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="mandala">🪷</span>
  <h1 class="hero-title">AyurAI</h1>
  <p class="hero-subtitle">Ancient wisdom · Modern intelligence · Holistic health prediction</p>
</div>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "result" not in st.session_state:
    st.session_state.result = None

# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.submitted:

    col_form, col_info = st.columns([3, 1], gap="large")

    with col_form:

        # ── Personal Details ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">Personal Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Full name", placeholder="e.g. Priya Sharma")
        with c2:
            gender = st.selectbox("Gender", ["Select", "Female", "Male", "Non-binary", "Prefer not to say"])
        with c3:
            age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

        location = st.text_input("Location (City, State)", placeholder="e.g. Kolkata, West Bengal")

        # ── Clinical Metrics ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">Clinical Metrics</div>', unsafe_allow_html=True)
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, step=0.1, format="%.1f")
        with c5:
            bp = st.number_input("Blood Pressure (systolic mmHg)", min_value=60, max_value=220, value=120, step=1)
        with c6:
            sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=600, value=95, step=1)
        with c7:
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=500, value=185, step=1)

        # ── Conditions ───────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Existing Conditions</div>', unsafe_allow_html=True)
        c8, c9, c10 = st.columns(3)
        with c8:
            thyroid = st.selectbox("Thyroid disorder", ["No", "Hypothyroid", "Hyperthyroid"])
        with c9:
            smoking = st.selectbox("Smoking", ["No", "Former smoker", "Current smoker"])
        with c10:
            asthma = st.selectbox("Asthma", ["No", "Mild", "Moderate", "Severe"])

        # ── Stress & Symptoms ────────────────────────────────────────────────
        st.markdown('<div class="section-label">Lifestyle & Symptoms</div>', unsafe_allow_html=True)
        stress = st.slider("Stress Level", min_value=1, max_value=10, value=5,
                           help="1 = Very low · 10 = Extremely high")
        symptoms = st.text_area(
            "Describe your symptoms",
            placeholder="e.g. Persistent fatigue, dry skin, cold hands and feet, difficulty concentrating for the past 3 weeks...",
            height=110
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🌿 Analyse & Get Ayurvedic Recommendations"):
            if not name.strip():
                st.warning("Please enter your name.")
            elif gender == "Select":
                st.warning("Please select a gender.")
            elif not symptoms.strip():
                st.warning("Please describe your symptoms.")
            else:
                # Store inputs in session
                st.session_state.inputs = dict(
                    name=name, gender=gender, age=age, location=location,
                    bmi=bmi, bp=bp, sugar=sugar, cholesterol=cholesterol,
                    thyroid=thyroid, smoking=smoking, asthma=asthma,
                    stress=stress, symptoms=symptoms
                )
                with st.spinner("Analysing with AI model…"):
                    time.sleep(1.5)   # replace with real model call
                st.session_state.submitted = True
                st.rerun()

    with col_info:
        st.markdown("""
<div class="ayur-card" style="margin-top:2rem">
  <div style="font-family:'Playfair Display',serif;font-size:1rem;color:var(--gold);margin-bottom:0.8rem">
    The three Doshas
  </div>
  <div style="margin-bottom:0.8rem">
    <span class="dosha-badge dosha-vata">Vata</span>
    <p style="color:var(--muted);font-size:0.82rem;margin-top:0.4rem">
      Air &amp; space. Governs movement, breathing, circulation. Imbalance → anxiety, dryness, insomnia.
    </p>
  </div>
  <div style="margin-bottom:0.8rem">
    <span class="dosha-badge dosha-pitta">Pitta</span>
    <p style="color:var(--muted);font-size:0.82rem;margin-top:0.4rem">
      Fire &amp; water. Governs digestion, metabolism, intelligence. Imbalance → inflammation, acidity, anger.
    </p>
  </div>
  <div>
    <span class="dosha-badge dosha-kapha">Kapha</span>
    <p style="color:var(--muted);font-size:0.82rem;margin-top:0.4rem">
      Earth &amp; water. Governs structure, immunity, lubrication. Imbalance → congestion, weight gain, lethargy.
    </p>
  </div>
</div>

<div class="ayur-card">
  <div style="font-family:'Playfair Display',serif;font-size:1rem;color:var(--gold);margin-bottom:0.6rem">
    How it works
  </div>
  <div style="color:var(--muted);font-size:0.82rem;line-height:1.7">
    Your inputs are cleaned, normalised, and passed to a trained ML model. The model predicts the most likely
    condition, which is then mapped to your Ayurvedic dosha. A rule-based engine produces herb
    recommendations, a diet plan, yoga poses, and lifestyle guidance tailored to your dosha and disease.
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    inp = st.session_state.inputs

    # ── Mock prediction (replace with real model) ────────────────────────────
    # In production: load pkl model, preprocess inputs, call model.predict()
    from model import predict_disease, identify_dosha, get_recommendations
    disease, confidence = predict_disease(inp)
    dosha = identify_dosha(inp)
    rec   = get_recommendations(disease, dosha)

    # ── Greeting ─────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="padding:1.5rem 0 0.5rem">
  <span style="color:var(--muted);font-size:0.9rem">Report for</span>
  <span style="font-family:'Playfair Display',serif;font-size:1.6rem;color:var(--text);margin-left:0.5rem">{inp['name']}</span>
  <span style="color:var(--muted);font-size:0.85rem;margin-left:1rem">{inp['age']} yrs · {inp['gender']} · {inp['location']}</span>
</div>
""", unsafe_allow_html=True)

    # ── Metric tiles ──────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="metric-row">
  <div class="metric-tile"><div class="val">{inp['bmi']}</div><div class="lbl">BMI</div></div>
  <div class="metric-tile"><div class="val">{inp['bp']}</div><div class="lbl">BP (mmHg)</div></div>
  <div class="metric-tile"><div class="val">{inp['sugar']}</div><div class="lbl">Sugar (mg/dL)</div></div>
  <div class="metric-tile"><div class="val">{inp['cholesterol']}</div><div class="lbl">Cholesterol</div></div>
  <div class="metric-tile"><div class="val">{inp['stress']}/10</div><div class="lbl">Stress</div></div>
</div>
""", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # ── Disease & Dosha ───────────────────────────────────────────────────
        dosha_cls = f"dosha-{dosha.lower()}"
        st.markdown(f"""
<div class="ayur-card">
  <div style="color:var(--muted);font-size:0.8rem;letter-spacing:0.1em;text-transform:uppercase">Predicted condition</div>
  <div class="result-disease">{disease}</div>
  <div style="margin-top:0.5rem">
    Confidence: <strong style="color:var(--gold)">{confidence}%</strong>
    &nbsp;&nbsp;
    Dosha: <span class="dosha-badge {dosha_cls}">{dosha}</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Herbal Medicine ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">Ayurvedic Herbs & Medicines</div>', unsafe_allow_html=True)
        herbs_html = "".join(f'<span class="herb-chip">🌿 {h}</span>' for h in rec["herbs"])
        st.markdown(f'<div class="ayur-card">{herbs_html}<p style="color:var(--muted);font-size:0.82rem;margin-top:0.8rem">{rec["remedy"]}</p></div>',
                    unsafe_allow_html=True)

        # ── Diet Plan ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Diet Plan</div>', unsafe_allow_html=True)
        diet_html = "".join(f'<div class="tip-item"><span class="tip-icon">◆</span>{d}</div>' for d in rec["diet"])
        st.markdown(f'<div class="ayur-card">{diet_html}</div>', unsafe_allow_html=True)

        # ── Yoga ──────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Yoga & Exercise</div>', unsafe_allow_html=True)
        yoga_html = "".join(f'<div class="tip-item"><span class="tip-icon">◇</span>{y}</div>' for y in rec["yoga"])
        st.markdown(f'<div class="ayur-card">{yoga_html}</div>', unsafe_allow_html=True)

        # ── Lifestyle ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Lifestyle Tips</div>', unsafe_allow_html=True)
        life_html = "".join(f'<div class="tip-item"><span class="tip-icon">◈</span>{l}</div>' for l in rec["lifestyle"])
        st.markdown(f'<div class="ayur-card">{life_html}</div>', unsafe_allow_html=True)

    with col_right:
        # ── Confidence gauge ──────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Model confidence", 'font': {'color': '#8a8fa8', 'size': 13}},
            number={'suffix': "%", 'font': {'color': '#c9a84c', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#2e3552'},
                'bar':  {'color': '#c9a84c'},
                'bgcolor': '#1a1e2a',
                'bordercolor': '#2e3552',
                'steps': [
                    {'range': [0, 50],  'color': '#1a1e2a'},
                    {'range': [50, 75], 'color': '#20263a'},
                    {'range': [75, 100],'color': '#252c40'},
                ],
                'threshold': {'line': {'color': '#e8894a', 'width': 2}, 'value': confidence}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e8e4d9',
            margin=dict(t=40, b=10, l=20, r=20),
            height=220
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Dosha radar ───────────────────────────────────────────────────────
        dosha_scores = rec.get("dosha_scores", {"Vata": 60, "Pitta": 35, "Kapha": 45})
        categories = list(dosha_scores.keys())
        values     = list(dosha_scores.values())
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(201,168,76,0.15)',
            line=dict(color='#c9a84c', width=2),
            marker=dict(color='#c9a84c', size=6)
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,100],
                                gridcolor='#2e3552', tickcolor='#2e3552',
                                tickfont=dict(color='#8a8fa8', size=10)),
                angularaxis=dict(gridcolor='#2e3552',
                                 tickfont=dict(color='#e8e4d9', size=12))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(text='Dosha balance', font=dict(color='#8a8fa8', size=13), x=0.5),
            margin=dict(t=50, b=20, l=30, r=30),
            height=280
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Disease distribution pie ──────────────────────────────────────────
        location_data = rec.get("location_data", {
            "labels": ["Hypothyroidism", "Diabetes", "Hypertension", "Asthma", "Other"],
            "values": [28, 22, 19, 14, 17]
        })
        fig_pie = go.Figure(go.Pie(
            labels=location_data["labels"],
            values=location_data["values"],
            hole=0.45,
            marker=dict(colors=['#c9a84c','#e8894a','#4caf7d','#7c9ecf','#e07b5a'],
                        line=dict(color='#0f1117', width=1)),
            textfont=dict(color='#e8e4d9', size=11)
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(text=f'Disease distribution — {inp["location"]}',
                       font=dict(color='#8a8fa8', size=12), x=0.5),
            legend=dict(font=dict(color='#8a8fa8', size=10), bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=40, b=10, l=10, r=10),
            height=280
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # ── Age vs disease bar (simulated regional) ───────────────────────────
        age_groups  = ["0–20", "21–35", "36–50", "51–65", "65+"]
        age_counts  = [8, 18, 34, 27, 13]
        fig_bar = go.Figure(go.Bar(
            x=age_groups, y=age_counts,
            marker_color='#c9a84c',
            marker_line_color='#8a6f2e',
            marker_line_width=0.5
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=dict(text=f'{disease} by age group (%)',
                       font=dict(color='#8a8fa8', size=12), x=0.5),
            xaxis=dict(tickfont=dict(color='#8a8fa8'), gridcolor='#2e3552', linecolor='#2e3552'),
            yaxis=dict(tickfont=dict(color='#8a8fa8'), gridcolor='#2e3552', linecolor='#2e3552'),
            margin=dict(t=40, b=20, l=30, r=10),
            height=240
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Download Report ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Health Report</div>', unsafe_allow_html=True)
    report_text = f"""
AyurAI — Health Report
Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}
{'─'*50}
Patient    : {inp['name']}
Age / Gender: {inp['age']} / {inp['gender']}
Location   : {inp['location']}

CLINICAL METRICS
  BMI          : {inp['bmi']}
  Blood Pressure: {inp['bp']} mmHg (systolic)
  Blood Sugar  : {inp['sugar']} mg/dL
  Cholesterol  : {inp['cholesterol']} mg/dL
  Stress Level : {inp['stress']}/10
  Thyroid      : {inp['thyroid']}
  Smoking      : {inp['smoking']}
  Asthma       : {inp['asthma']}

SYMPTOMS
  {inp['symptoms']}

{'─'*50}
PREDICTION
  Disease      : {disease}
  Confidence   : {confidence}%
  Dosha        : {dosha}

AYURVEDIC HERBS
  {', '.join(rec['herbs'])}

ORGANIC REMEDY
  {rec['remedy']}

DIET PLAN
  {'  '.join(['- ' + d for d in rec['diet']])}

YOGA & EXERCISE
  {'  '.join(['- ' + y for y in rec['yoga']])}

LIFESTYLE TIPS
  {'  '.join(['- ' + l for l in rec['lifestyle']])}
{'─'*50}
"Stay healthy, be happy, take care 🌿"
    AyurAI — Ancient wisdom · Modern intelligence
"""
    col_dl, col_back = st.columns([1, 1])
    with col_dl:
        st.download_button(
            label="⬇  Download Full Report (.txt)",
            data=report_text,
            file_name=f"AyurAI_Report_{inp['name'].replace(' ','_')}.txt",
            mime="text/plain"
        )
    with col_back:
        if st.button("← New Analysis"):
            st.session_state.submitted = False
            st.rerun()

    # ── Wellness message ──────────────────────────────────────────────────────
    st.markdown("""
<div class="wellness-msg">
  "Stay healthy, be happy, take care 🌿"
</div>
""", unsafe_allow_html=True)
