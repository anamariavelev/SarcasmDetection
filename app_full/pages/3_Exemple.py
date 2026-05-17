import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
from utils.model import load_model, full_analysis, POSITIVE, NEGATIVE, EMOTION_EMOJI

st.set_page_config(page_title="Exemple — AfectAI", page_icon="📚", layout="wide")

st.title("📚 Galerie de exemple")
st.caption("Exemple reprezentative care ilustrează capacitățile și limitările modelului")

with st.spinner("⏳ Se încarcă modelul..."):
    model, tokenizer, labels, threshold = load_model()

EXAMPLES = {
    "Sarcasm explicit (neural)": [
        ("Yeah sure, because that ALWAYS works out perfectly.", True),
        ("Oh wow, what a GREAT idea. Totally not predictable at all.", True),
    ],
    "Sarcasm prin contrast": [
        ("Wonderful! My coffee spilled all over my laptop again.", True),
        ("Great news! The flight got cancelled and I missed my connection.", True),
    ],
    "Emoții pozitive genuine": [
        ("I am so happy we were able to come here and be part of all this.", False),
        ("Thank you so much, this really means everything to me!", False),
    ],
    "Emoții negative genuine": [
        ("I'm afraid this is not going to work out.", False),
        ("This is absolutely disgusting. I can't believe they did that.", False),
    ],
    "Ambiguu / dificil": [
        ("Oh great, we get to do this again.", True),
        ("I love how they never tell us anything in advance.", True),
    ],
}

for category, examples in EXAMPLES.items():
    st.subheader(category)
    cols = st.columns(len(examples))
    for col, (text, expected_sarc) in zip(cols, examples):
        with col:
            with st.spinner(f"Analizez…"):
                r = full_analysis(text, model, tokenizer, labels, threshold)

            sarc_ok = r["is_sarcastic"] == expected_sarc
            status = "✅ corect" if sarc_ok else "❌ greșit"

            with st.container(border=True):
                st.markdown(f"*\"{text}\"*")
                st.markdown(f"**Așteptat:** {'sarcastic' if expected_sarc else 'non-sarcastic'} · {status}")

                sarc_color = "#e74c3c" if r["is_sarcastic"] else "#27ae60"
                sarc_label = "⚠️ Sarcasm" if r["is_sarcastic"] else "✅ Normal"
                st.markdown(f"**Detectat:** :{('red' if r['is_sarcastic'] else 'green')}[{sarc_label}]")

                if r["active"]:
                    emotions_str = " · ".join(
                        f"{EMOTION_EMOJI.get(l,'')} {l} `{p:.0%}`"
                        for l, p in r["active"][:3]
                    )
                    st.caption(f"Emoții: {emotions_str}")
                else:
                    top1_lbl, top1_p = r["scored"][0]
                    st.caption(f"Top: {EMOTION_EMOJI.get(top1_lbl,'')} {top1_lbl} {top1_p:.0%}")

                methods_on = []
                if r["neural_pos"]:  methods_on.append("neural")
                if r["rule_pos"]:    methods_on.append("rule")
                if r["contrast_hit"]: methods_on.append("contrast")
                if methods_on:
                    st.caption(f"Metode: {', '.join(methods_on)}")

    st.divider()

# ── Summary ───────────────────────────────────────────────────
st.subheader("Sumar performanță pe exemple")
total = sum(len(v) for v in EXAMPLES.values())

results = []
for category, examples in EXAMPLES.items():
    correct = 0
    for text, expected in examples:
        r = full_analysis(text, model, tokenizer, labels, threshold)
        if r["is_sarcastic"] == expected:
            correct += 1
    results.append((category, correct, len(examples)))

cats   = [r[0] for r in results]
scores = [r[1]/r[2] for r in results]
colors = ["#2ecc71" if s >= 0.75 else "#F39C12" if s >= 0.5 else "#e74c3c" for s in scores]

fig = go.Figure(go.Bar(
    x=cats, y=scores,
    marker_color=colors,
    text=[f"{c}/{t}" for _, c, t in results],
    textposition="outside",
))
fig.update_layout(
    yaxis=dict(range=[0, 1.2], tickformat=".0%"),
    height=300, margin=dict(t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
)
st.plotly_chart(fig, use_container_width=True)
