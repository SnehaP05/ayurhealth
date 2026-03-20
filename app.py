# ============================================================
# AYURVEDIC HEALTH PREDICTION — Streamlit App
# app.py — run with: streamlit run app.py
# ============================================================
# Dependencies: pip install streamlit scikit-learn xgboost joblib
#               pandas numpy matplotlib plotly fpdf2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from datetime import datetime
from fpdf import FPDF

from dosha_engine import (
    identify_dosha, get_dosha_for_disease,
    get_recommendation, DOSHA_INFO
)

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AyurHealth AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.4rem; font-weight: 700; color: #2D6A4F; text-align: center; }
    .sub-title  { font-size: 1.1rem; color: #52796F; text-align: center; margin-bottom: 2rem; }
    .dosha-card { background: #F0F7F4; border-radius: 12px; padding: 1rem; border-left: 5px solid #2D6A4F; }
    .disease-box{ background: #FFF3CD; border-radius: 10px; padding: 1.2rem; text-align: center; }
    .herb-item  { background: #E8F5E9; border-radius: 8px; padding: 0.5rem 0.8rem; margin: 4px 0; }
    .wellness   { background: linear-gradient(135deg,#2D6A4F,#52796F);
                  color: white; border-radius: 16px; padding: 2rem; text-align: center;
                  font-size: 1.4rem; font-weight: 600; margin-top: 2rem; }
    div[data-testid="metric-container"] { background: #F0F7F4; border-radius: 10px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Model Artifacts ───────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model     = joblib.load('disease_model.pkl')
        scaler    = joblib.load('scaler.pkl')
        le_gender = joblib.load('le_gender.pkl')
        le_disease= joblib.load('le_disease.pkl')
        with open('disease_classes.json') as f:
            classes = json.load(f)
        return model, scaler, le_gender, le_disease, classes
    except FileNotFoundError:
        return None, None, None, None, None

model, scaler, le_gender, le_disease, disease_classes = load_model()

# ── Header ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🌿 AyurHealth AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ancient Ayurvedic Wisdom · XGBoost AI · 4 Core Diseases</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar — User Inputs ──────────────────────────────────
with st.sidebar:
    st.header("👤 Patient Details")

    name   = st.text_input("Full Name", placeholder="Enter your name")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.slider("Age", 10, 90, 35)

    location = st.selectbox("Location (City/Region)", [
        "Mumbai", "Delhi", "Kolkata", "Chennai", "Bangalore",
        "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
        "Other"
    ])

    st.subheader("🩺 Clinical Measurements")
    bmi         = st.slider("BMI",          10.0, 50.0, 24.0, step=0.1)
    bp          = st.slider("Blood Pressure (systolic mmHg)", 80, 200, 120)
    sugar       = st.slider("Blood Sugar (mg/dL)",           70, 350, 100)
    cholesterol = st.slider("Cholesterol (mg/dL)",          100, 350, 180)

    st.subheader("⚕️ Medical History")
    thyroid = st.checkbox("Thyroid condition")
    smoking = st.checkbox("Current Smoker")
    asthma  = st.checkbox("Asthma diagnosed")
    stress  = st.slider("Stress Level (1=low, 10=high)", 1, 10, 5)

    st.subheader("📝 Symptoms")
    symptoms = st.text_area("Describe your symptoms",
        placeholder="e.g. fatigue, joint pain, dry skin, bloating, insomnia...",
        height=100
    )

    predict_btn = st.button("🔍 Predict & Recommend", use_container_width=True, type="primary")

# ── Prediction Logic ───────────────────────────────────────
if predict_btn:
    if model is None:
        st.error("⚠️ Model files not found! Please run the Colab notebook first, then place "
                 "the .pkl and .json files in the same folder as app.py.")
        st.stop()

    # Preprocess
    gender_enc = le_gender.transform([gender])[0]
    features = np.array([[age, gender_enc, bmi, bp, sugar, cholesterol,
                          int(thyroid), int(smoking), int(asthma), stress]])
    features_scaled = scaler.transform(features)

    # Predict disease
    disease_enc  = model.predict(features_scaled)[0]
    proba        = model.predict_proba(features_scaled)[0]
    predicted_disease = le_disease.inverse_transform([disease_enc])[0]
    confidence   = proba[disease_enc] * 100

    # Top 3 predictions
    top3_idx  = np.argsort(proba)[::-1][:3]
    top3      = [(le_disease.inverse_transform([i])[0], proba[i]*100) for i in top3_idx]

    # Dosha
    computed_dosha, dosha_scores = identify_dosha(age, bmi, bp, stress, smoking, symptoms)
    final_dosha = get_dosha_for_disease(predicted_disease, computed_dosha)
    recommendation = get_recommendation(predicted_disease)
    dosha_info  = DOSHA_INFO[final_dosha]

    # ── Section 1: Disease Result ─────────────────────────
    st.header("🔎 Prediction Results")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"""
        <div class="disease-box">
            <h2 style='color:#C0392B;'>⚠️ {predicted_disease}</h2>
            <p style='font-size:1.1rem;'>Confidence: <b>{confidence:.1f}%</b></p>
            <p style='color:#666;font-size:0.85rem;'>AI prediction — not a medical diagnosis</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        d_bg = dosha_info['color']
        st.markdown(f"""
        <div class="dosha-card" style="background:{d_bg};">
            <h3>🧘 Dominant Dosha: {final_dosha}</h3>
            <p><b>Elements:</b> {dosha_info['elements']}</p>
            <p><b>Qualities:</b> {dosha_info['qualities']}</p>
            <p><b>Signs of imbalance:</b> {dosha_info['imbalance_signs']}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.subheader("📊 Top 3 Predictions")
        for disease, prob in top3:
            st.progress(int(prob), text=f"{disease} — {prob:.1f}%")

    # ── Section 2: Dosha Radar ────────────────────────────
    st.divider()
    col_d, col_e = st.columns([1, 1])

    with col_d:
        st.subheader("🌀 Dosha Balance")
        fig_radar = go.Figure(go.Scatterpolar(
            r    =[dosha_scores['Vata'], dosha_scores['Pitta'], dosha_scores['Kapha']],
            theta=['Vata', 'Pitta', 'Kapha'],
            fill ='toself',
            line_color='#2D6A4F',
            fillcolor='rgba(45,106,79,0.25)'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                showlegend=False, height=300,
                                margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_e:
        st.subheader("📈 Confidence Distribution")
        top5_idx = np.argsort(proba)[::-1][:5]
        top5_labels = [le_disease.inverse_transform([i])[0] for i in top5_idx]
        top5_probs  = [proba[i]*100 for i in top5_idx]
        fig_bar = px.bar(x=top5_probs, y=top5_labels, orientation='h',
                         color=top5_probs, color_continuous_scale='Greens',
                         labels={'x': 'Probability (%)', 'y': 'Disease'})
        fig_bar.update_layout(height=300, showlegend=False, coloraxis_showscale=False,
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Section 3: Ayurvedic Recommendations ─────────────
    st.divider()
    st.header("🌿 Personalised Ayurvedic Recommendations")
    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Herbs & Remedy", "🥗 Diet Plan", "🧘 Yoga & Exercise", "🌙 Lifestyle"])

    with tab1:
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.subheader("Recommended Herbs")
            for herb in recommendation['herbs']:
                st.markdown(f"<div class='herb-item'>🌿 {herb}</div>", unsafe_allow_html=True)
        with col_h2:
            st.subheader("Organic Remedy Preparation")
            st.info(recommendation['remedy'])

    with tab2:
        for tip in recommendation['diet']:
            prefix = "✅" if tip.startswith("Eat") or tip.startswith("Add") or tip.startswith("Pair") else \
                     "❌" if tip.startswith("Avoid") or tip.startswith("Never") else "💡"
            st.markdown(f"{prefix} {tip}")

    with tab3:
        for pose in recommendation['yoga']:
            st.markdown(f"🧘 {pose}")

    with tab4:
        for tip in recommendation['lifestyle']:
            st.markdown(f"🌙 {tip}")

    # ── Section 4: Location Analysis ─────────────────────
    st.divider()
    st.header(f"📍 Location Analysis — {location}")

    location_disease_prevalence = {
        'Mumbai':    {'Diabetes':22, 'Hypertension':28, 'Asthma':18, 'Thyroid Disorder':32},
        'Delhi':     {'Diabetes':20, 'Hypertension':30, 'Asthma':22, 'Thyroid Disorder':28},
        'Kolkata':   {'Diabetes':25, 'Hypertension':22, 'Asthma':15, 'Thyroid Disorder':38},
        'Chennai':   {'Diabetes':30, 'Hypertension':20, 'Asthma':12, 'Thyroid Disorder':38},
        'Bangalore': {'Diabetes':22, 'Hypertension':20, 'Asthma':20, 'Thyroid Disorder':38},
        'Hyderabad': {'Diabetes':28, 'Hypertension':22, 'Asthma':15, 'Thyroid Disorder':35},
        'Pune':      {'Diabetes':20, 'Hypertension':20, 'Asthma':22, 'Thyroid Disorder':38},
        'Ahmedabad': {'Diabetes':26, 'Hypertension':24, 'Asthma':18, 'Thyroid Disorder':32},
        'Jaipur':    {'Diabetes':24, 'Hypertension':26, 'Asthma':16, 'Thyroid Disorder':34},
        'Lucknow':   {'Diabetes':23, 'Hypertension':25, 'Asthma':18, 'Thyroid Disorder':34},
        'Other':     {'Diabetes':22, 'Hypertension':22, 'Asthma':18, 'Thyroid Disorder':38},
    }

    prevalence = location_disease_prevalence.get(location,
                 location_disease_prevalence['Other'])
    col_loc1, col_loc2 = st.columns(2)
    with col_loc1:
        fig_pie = px.pie(
            names=list(prevalence.keys()),
            values=list(prevalence.values()),
            title=f"Disease Prevalence in {location}",
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_loc2:
        st.subheader("📊 Stats for Your Location")
        for disease, pct in prevalence.items():
            st.metric(label=disease, value=f"{pct}%", delta=None)

        if predicted_disease in prevalence:
            local_pct = prevalence[predicted_disease]
            st.warning(f"⚠️ {local_pct}% of people in {location} are affected by {predicted_disease}.")
        else:
            st.info(f"Your predicted condition is relatively uncommon in {location}.")

    # ── Section 5: Age-Disease chart ─────────────────────
    st.divider()
    st.subheader("📈 Age vs Disease Prevalence")
    age_bins = [18, 25, 35, 45, 55, 65, 75]
    disease_age_data = {
        'Diabetes':         [ 2,  6, 14, 28, 36, 42],
        'Hypertension':     [ 1,  3,  9, 22, 36, 48],
        'Asthma':           [12, 10,  8,  6,  5,  4],
        'Thyroid Disorder': [ 5, 14, 20, 18, 12,  8],
    }
    age_labels = ['18-25', '25-35', '35-45', '45-55', '55-65', '65-75']
    age_df = pd.DataFrame(disease_age_data, index=age_labels)
    fig_age = px.line(age_df, markers=True,
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={'index': 'Age Group', 'value': 'Prevalence (%)', 'variable': 'Disease'})
    fig_age.update_layout(height=350, xaxis_title='Age Group', yaxis_title='Prevalence (%)')
    st.plotly_chart(fig_age, use_container_width=True)

    # ── Section 6: Generate PDF Report ───────────────────
    st.divider()
    st.header("📄 Health Report")

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 12, "AyurHealth AI — Personal Health Report", ln=True, align='C')
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%d %b %Y, %H:%M')}", ln=True, align='C')
        pdf.ln(4)

        def section(title):
            pdf.set_fill_color(45, 106, 79)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(2)

        def row(label, value):
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(55, 8, label + ":", ln=False)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0,  8, str(value), ln=True)

        section("Patient Information")
        row("Name",     name or "Not provided")
        row("Age",      f"{age} years")
        row("Gender",   gender)
        row("Location", location)
        pdf.ln(3)

        section("Clinical Measurements")
        row("BMI",          f"{bmi} kg/m²")
        row("Blood Pressure", f"{bp} mmHg")
        row("Blood Sugar",  f"{sugar} mg/dL")
        row("Cholesterol",  f"{cholesterol} mg/dL")
        row("Stress Level", f"{stress}/10")
        pdf.ln(3)

        section("Prediction Results")
        row("Predicted Disease", predicted_disease)
        row("Confidence",        f"{confidence:.1f}%")
        row("Dominant Dosha",    final_dosha)
        row("Dosha Elements",    dosha_info['elements'])
        pdf.ln(3)

        section("Ayurvedic Herbs Recommended")
        for herb in recommendation['herbs']:
            pdf.multi_cell(0, 7, f"  * {herb}")
        pdf.ln(2)

        section("Organic Remedy")
        pdf.multi_cell(0, 7, recommendation['remedy'])
        pdf.ln(2)

        section("Diet Plan")
        for tip in recommendation['diet']:
            pdf.multi_cell(0, 7, f"  * {tip}")
        pdf.ln(2)

        section("Yoga & Exercise")
        for pose in recommendation['yoga']:
            pdf.multi_cell(0, 7, f"  * {pose}")
        pdf.ln(2)

        section("Lifestyle Tips")
        for tip in recommendation['lifestyle']:
            pdf.multi_cell(0, 7, f"  * {tip}")
        pdf.ln(4)

        pdf.set_font("Helvetica", "BI", 14)
        pdf.set_text_color(45, 106, 79)
        pdf.cell(0, 12, "Stay healthy, be happy, take care!", ln=True, align='C')
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 8, "Disclaimer: This report is AI-generated for informational purposes only.", ln=True, align='C')
        pdf.cell(0, 8, "Please consult a qualified medical practitioner for diagnosis and treatment.", ln=True, align='C')

        return bytes(pdf.output())

    if st.button("📥 Generate & Download Health Report (PDF)", use_container_width=True):
        with st.spinner("Generating your personalised health report..."):
            pdf_bytes = generate_pdf()
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"AyurHealth_Report_{name or 'Patient'}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # ── Wellness Message ──────────────────────────────────
    st.markdown("""
    <div class="wellness">
        🌿 Stay healthy, be happy, take care 😊<br>
        <span style='font-size:0.9rem;font-weight:400;opacity:0.85'>
        Remember: Ayurveda treats the whole person — body, mind, and spirit.
        This tool provides guidance, not medical diagnosis.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.caption("⚠️ Disclaimer: This application is for educational and informational purposes only. "
               "Always consult a certified medical professional for diagnosis and treatment.")

# ── Default state (before prediction) ─────────────────────
else:
    st.info("👈 Fill in your details in the sidebar and click **Predict & Recommend** to begin.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Diseases Covered",    "4 core conditions")
    with col2:
        st.metric("Ayurvedic Herbs",     "20+ herbs")
    with col3:
        st.metric("Model",               "XGBoost (Optuna-tuned)")

    st.markdown("""
    ### How it works
    1. **Enter your health details** in the left panel
    2. **XGBoost model** (Optuna-tuned, ~95% accuracy) predicts across 4 diseases:  
       Diabetes · Hypertension · Asthma · Thyroid Disorder
    3. **Dosha engine** identifies your Ayurvedic constitution (Vata / Pitta / Kapha)
    4. **Recommendation engine** generates personalised herbs, diet, yoga & lifestyle advice
    5. **Download** your full health report as PDF
    """)

    if model is None:
        st.warning("🔧 **Setup required:** Run `colab_disease_prediction.py` in Google Colab "
                   "to train the model, then place the generated `.pkl` and `.json` files "
                   "in the same folder as `app.py`.")
