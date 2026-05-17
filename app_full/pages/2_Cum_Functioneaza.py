import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Cum funcționează — AfectAI", page_icon="⚙️", layout="wide")

st.title("⚙️ Cum funcționează")
st.caption("Arhitectura sistemului, datele de antrenament și metodele de detecție")

st.divider()

# ── Arhitectură ───────────────────────────────────────────────
st.subheader("1. Arhitectura modelului")
col_arch, col_desc = st.columns([1, 1])

with col_arch:
    st.code("""
Input text
   ↓
Tokenizare XLM-R (SentencePiece, max 128 tokeni)
   ↓
┌─────────────────────────────────┐
│  XLM-RoBERTa Encoder (12 blocuri)│
│  ● Blocurile 1-10:  FROZEN      │
│  ● Blocurile 11-12: TRAINABLE   │
└─────────────────────────────────┘
   ↓
Reprezentare [CLS] (768 dimensiuni)
   ↓
Dropout (p=0.3)
   ├──→ emotions_head: Linear(768→28)
   │         ↓ sigmoid per clasă
   │    28 probabilități emoții
   │
   └──→ sarcasm_head: Linear(768→1)
              ↓ sigmoid
         1 probabilitate sarcasm
""", language=None)

with col_desc:
    st.markdown("""
    **De ce fine-tuning parțial?**

    XLM-RoBERTa are ~270M parametri. Antrenarea completă ar necesita
    resurse enorme și ar duce la overfitting pe cele 70k exemple.
    Înghețând primele 10 blocuri, păstrăm cunoștințele lingvistice
    generale și antrenăm doar ~36M parametri (13%).

    **De ce două capete?** (Multi-Task Learning)

    Capul de emoții și cel de sarcasm partajează același backbone —
    reprezentările contextuale bogate ale XLM-R. Antrenamentul
    simultan permite transferul implicit de informație între cele
    două sarcini corelate semantic.

    **BCEWithLogitsLoss** (nu CrossEntropy)

    Clasificarea multi-etichetă necesită predicții independente
    per clasă — sigmoid, nu softmax. Un text poate fi simultan
    *joy* și *gratitude*, fără să fie forțat la o singură clasă.
    """)

st.divider()

# ── Date ──────────────────────────────────────────────────────
st.subheader("2. Datele de antrenament")
d1, d2 = st.columns(2)

with d1:
    st.markdown("#### GoEmotions")
    st.markdown("""
    - **Sursă:** Google Research (2020)
    - **Volum:** 58.009 comentarii Reddit → 50k sample
    - **Etichete:** 28 emoții fine-grained (multi-label)
    - **Dezechilibru:** *neutral* 27% ↔ *grief* 0.1%
    - **Soluție:** `pos_weight` per clasă

    Exemplu de adnotare:
    > *"I can't believe they actually did it!"*
    > → `excitement`, `surprise`
    """)

with d2:
    st.markdown("#### SARC — Self-Annotated Reddit Corpus")
    st.markdown("""
    - **Sursă:** Khodak et al. (2018)
    - **Volum:** 1M+ comentarii → 20k sample (10k/10k)
    - **Etichete:** sarcasm binar (auto-adnotat cu `\\s`)
    - **Rol:** antrenament cap sarcasm cu etichete reale
    - **Mascare loss:** exemplele SARC nu au emoții →
      `loss_emoții` se calculează doar pe GoEmotions

    Exemplu:
    > *"Oh sure, because government always knows best \\s"*
    > → `sarcasm = 1`
    """)

# ── Dezechilibru ──────────────────────────────────────────────
st.markdown("**Impact integrare SARC:**")
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    name="Fără SARC", x=["F1 head-only", "Recall", "Precision"],
    y=[0.000, 0.000, 0.000], marker_color="#e74c3c"
))
fig_bar.add_trace(go.Bar(
    name="Cu SARC", x=["F1 head-only", "Recall", "Precision"],
    y=[0.487, 0.877, 0.337], marker_color="#2ecc71"
))
fig_bar.update_layout(
    barmode="group", height=280, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"), legend=dict(orientation="h"),
    yaxis=dict(range=[0, 1]),
)
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── 3 metode sarcasm ──────────────────────────────────────────
st.subheader("3. Cele 3 metode de detecție a sarcasmului")

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("""
    #### 🤖 Neural (head)
    Capul `sarcasm_head: Linear(768→1)` antrenat direct
    pe GoEmotions (pseudo-etichete din co-ocurența emoțiilor)
    **+** SARC (etichete reale).

    **Threshold:** 0.62

    **Avantaj:** prinde sarcasm subtil cu markeri
    lexicali ("totally", "yeah sure")

    **Dezavantaj:** limitată de distribuția SARC
    (stil Reddit, engleză informală)
    """)

with m2:
    st.markdown("""
    #### 📋 Rule-based
    Verifică dacă textul activează simultan o emoție
    **pozitivă dominantă** (≥65%, în top-3) și una
    **negativă reală** (≥50%, ratio ≥45%).

    18 perechi SARCASM_COMBOS:
    `joy+anger`, `excitement+annoyance`,
    `approval+disapproval`...

    **Avantaj:** interpretabil, stabil

    **Dezavantaj:** necesită ambele emoții explicit prezente
    """)

with m3:
    st.markdown("""
    #### 🔀 Contrast inter-propoziție
    Împarte textul la punctuație și rulează modelul
    pe fiecare parte separat.

    Dacă **prima parte e pozitivă** (≥55%) și
    **a doua nu e pozitivă** → sarcasm prin contrast.

    *"Wonderful! My coffee spilled all over my laptop."*
    → "Wonderful" = pozitiv → "coffee spilled" = negativ

    **Avantaj:** prinde "Wonderful! + situație negativă"
    """)

st.divider()
st.subheader("4. Rezultate experimentale")
r1, r2, r3, r4 = st.columns(4)
r1.metric("F1-micro emoții", "0.294", help="GoEmotions 50k, threshold 0.85")
r2.metric("F1 sarcasm", "0.487", help="Cu SARC integrat")
r3.metric("Recall sarcasm", "87.7%", help="Detectează 87.7% din textele sarcastice")
r4.metric("Threshold optim", "0.85", help="Grid search [0.20–0.90]")
