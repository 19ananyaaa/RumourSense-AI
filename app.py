
# RUMOURSENSE AI

import streamlit as st
import pickle
import re

# LOAD MODEL
model = pickle.load(open("rumour_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# CLEAN TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# PREDICT
def predict(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0][1]
    label = "Rumour" if prob > 0.5 else "Real"
    return label, prob

# RISK
def get_risk(prob):
    if prob < 0.4:
        return "LOW", "#22c55e"
    elif prob < 0.75:
        return "MEDIUM", "#facc15"
    else:
        return "HIGH", "#ef4444"

# PAGE CONFIG
st.set_page_config(page_title="RumourSense AI", layout="centered")

# CSS 
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg,#0a0f1f,#020617);
}

/* Container */
.block-container {
    max-width: 900px;
    margin: auto;
}

/* Title */
.title {
    text-align:center;
    font-size:60px;
    font-weight:700;
    background: linear-gradient(90deg,#00c6ff,#7cffcb);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* Subtitle */
.subtitle {
    text-align:center;
    color:#9ca3af;
    letter-spacing:2px;
    margin-bottom:30px;
}

/* Input */
textarea {
    background:#111827 !important;
    color:white !important;
    border-radius:16px !important;
    padding:14px;
    font-size:16px;
}

/* Button */
.stButton>button {
    display:block;
    margin:20px auto;
    background: linear-gradient(90deg,#00c6ff,#7cffcb);
    color:black;
    font-weight:bold;
    border-radius:14px;
    padding:14px 35px;
    transition:0.3s;
}
.stButton>button:hover {
    box-shadow:0 0 25px #00c6ff;
    transform:scale(1.05);
}

/* Section */
.section {
    margin-top:40px;
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg,#00c6ff,#3b82f6);
}

/* CARDS */
.card {
    background: rgba(255,255,255,0.05);
    padding:30px;
    border-radius:18px;
    text-align:center;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.08);
}

/* Card Title */
.card-title {
    font-size:14px;
    color:#9ca3af;
    margin-bottom:10px;
}

/* Card Value */
.card-value {
    font-size:36px;
    font-weight:bold;
}

/* Dot */
.dot {
    height:18px;
    width:18px;
    border-radius:50%;
    display:inline-block;
    margin-right:10px;
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="title">🌐 RumourSense AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NEURO-FUZZY EARLY RUMOUR SPREAD PREDICTION</div>', unsafe_allow_html=True)

# INPUT
user_input = st.text_area("Paste tweet / news / message here...", height=120)

# BUTTON
if st.button("🚀 Predict Rumour Spread"):

    label, prob = predict(user_input)
    risk, color = get_risk(prob)
    percent = round(prob * 100, 1)

    st.markdown('<div class="section">', unsafe_allow_html=True)

    st.markdown("### 📊 Neuro-Analysis Result")
    st.write(f"AI Confidence Match: {percent}%")

    # Progress bar
    st.progress(prob)

    # CARDS (PERFECT LAYOUT)
    
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">FINAL VERDICT</div>
            <div class="card-value">
                {label} {"❌" if label=="Rumour" else "✅"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">RISK INDEX</div>
            <div class="card-value">{prob:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">THREAT LEVEL</div>
            <div class="card-value">
                <span class="dot" style="background:{color}"></span>
                {risk}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)