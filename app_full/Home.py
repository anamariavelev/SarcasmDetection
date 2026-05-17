import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="AfectAI — Emotion & Sarcasm Detector",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.hero-title {
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #48C9B0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    font-size: 1.2rem; color: #aaa; margin-top: 0.3rem; margin-bottom: 2rem;
}
.stat-card {
    background: #1e1e2e; border-radius: 12px; padding: 1.5rem;
    text-align: center; border: 1px solid #333;
}
.stat-num { font-size: 2.4rem; font-weight: 700; color: #6C63FF; }
.stat-label { color: #aaa; font-size: 0.9rem; margin-top: 0.3rem; }
.feature-card {
    background: #1e1e2e; border-radius: 12px; padding: 1.4rem;
    border-left: 4px solid #6C63FF; margin-bottom: 1rem;
}
.feature-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.3rem; }
.badge {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 0.2rem;
}
.badge-purple { background: #2d2b55; color: #6C63FF; }
.badge-teal   { background: #1a3a37; color: #48C9B0; }
.badge-orange { background: #3a2a1a; color: #F39C12; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
col_hero, col_img = st.columns([3, 1])
with col_hero:
    st.markdown('<p class="hero-title">🧠 AfectAI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Procesare afectivă automată a textelor informale — '
        'clasificare multi-etichetă a emoțiilor și detectarea sarcasmului '
        'prin modele Transformer multilinguale.</p>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <span class="badge badge-purple">XLM-RoBERTa</span>
    <span class="badge badge-teal">GoEmotions</span>
    <span class="badge badge-teal">SARC</span>
    <span class="badge badge-orange">Multi-Task Learning</span>
    <span class="badge badge-purple">PyTorch</span>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶  Încearcă Demo-ul", type="primary", use_container_width=False):
        st.switch_page("pages/1_Demo.py")

# ── Stats ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    ("28", "categorii emoționale"),
    ("70k", "exemple antrenament"),
    ("0.294", "F1-micro emoții"),
    ("0.487", "F1 sarcasm"),
    ("3", "metode detecție sarcasm"),
]
for col, (num, label) in zip([c1,c2,c3,c4,c5], stats):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-num">{num}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# ── Features ──────────────────────────────────────────────────
st.subheader("Ce face aplicația")
f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🎭 Clasificare Multi-Emoție</div>
        Detectează simultan până la 28 de emoții fine-grained (admiration, joy, anger, fear…)
        din text informal. Pragul optim de 0.85 este calibrat automat pe setul de validare.
    </div>""", unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🎯 Detecție Sarcasm în 3 Straturi</div>
        <b>Neural</b> — cap antrenat pe 20k exemple SARC reale.<br>
        <b>Rule-based</b> — co-ocurența emoțiilor contradictorii.<br>
        <b>Contrast</b> — propoziție pozitivă urmată de context negativ.
    </div>""", unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🌍 Model Multilingual</div>
        Bazat pe XLM-RoBERTa, pre-antrenat pe 100 de limbi. Deși antrenat pe
        engleză (GoEmotions + SARC), arhitectura permite extensie zero-shot
        la texte în română sau alte limbi.
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── About ─────────────────────────────────────────────────────
st.subheader("Despre proiect")
ab1, ab2 = st.columns([2, 1])
with ab1:
    st.markdown("""
    Această aplicație este implementarea practică a lucrării de disertație
    **"Procesarea Afectivă Automată a Textelor Informale: Clasificare Multi-Etichetă
    a Emoțiilor și Detectarea Sarcasmului prin Modele Transformer Multilinguale"**.

    Sistemul folosește **fine-tuning parțial** al modelului XLM-RoBERTa (ultimele 2 din 12
    blocuri transformer) și o arhitectură **multi-task** cu două capete de clasificare
    independente — antrenate simultan pe GoEmotions (50k) și SARC (20k).

    Studiul ablation demonstrează că integrarea datelor SARC crește F1-ul detecției
    sarcasmului de la **0.000 → 0.487**, cu un recall de **87.7%**.
    """)
with ab2:
    st.markdown("""
<div style="background:#1e1e2e;border:1px solid #333;border-radius:10px;padding:1.2rem 1.4rem;line-height:2">
<div>👩‍🎓 <b>Student:</b> Ana-Maria Velev</div>
<div>👨‍🏫 <b>Coordonator:</b> Lect. dr. ing. Claudiu Domuța</div>
<div>🎓 <b>Specializare:</b> Informatică Aplicată în Ingineria Sistemelor Complexe</div>
<div>🏛️ <b>Facultatea de Automatică și Calculatoare, UTCN</b></div>
<div>📅 <b>An universitar:</b> 2025–2026</div>
<div>🛠️ <b>Stack:</b> Python · PyTorch · HuggingFace Transformers · Streamlit · Google Colab</div>
</div>
""", unsafe_allow_html=True)
