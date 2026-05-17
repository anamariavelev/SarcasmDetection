import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
from utils.model import load_model, full_analysis, POSITIVE, NEGATIVE, EMOTION_EMOJI

st.set_page_config(page_title="Demo — AfectAI", page_icon="🎭", layout="wide")

st.markdown("""
<style>
.sarc-box-yes {
    background: linear-gradient(135deg, #3d1515, #5c1f1f);
    border: 1px solid #e74c3c; border-radius: 10px;
    padding: 0.7rem 1.1rem; margin-bottom: 0.4rem;
}
.sarc-box-no {
    background: linear-gradient(135deg, #0f2d1f, #1a4a2e);
    border: 1px solid #27ae60; border-radius: 10px;
    padding: 0.7rem 1.1rem; margin-bottom: 0.4rem;
}
.sarc-title { font-size: 1.1rem; font-weight: 700; }
.method-pill {
    display: inline-block; padding: 0.15rem 0.6rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin: 0.1rem;
}
.pill-on  { background: #2d4a2d; color: #2ecc71; border: 1px solid #2ecc71; }
.pill-off { background: #2a2a2a; color: #666;    border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

EXAMPLES = [
    "Oh great, another Monday morning meeting that could have been an email!",
    "I am so happy we were able to come here and be part of all this.",
    "Wonderful! My coffee spilled all over my laptop again.",
    "Yeah sure, because that ALWAYS works out so well...",
    "I'm afraid it's not going to work out this time.",
    "This is absolutely disgusting, I can't believe they did that.",
    "Thank you so much for your help, it really means a lot to me!",
]

st.title("🎭 Demo — Analiză Afectivă")
st.caption("Introdu un text sau alege un exemplu · Modelul detectează emoții și sarcasm în timp real")

with st.spinner("⏳ Se încarcă modelul..."):
    model, tokenizer, labels, threshold = load_model()

st.divider()

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# Layout principal: stânga (input + rezultate) | dreapta (exemple)
col_main, col_examples = st.columns([3, 1])

with col_examples:
    st.markdown("**Exemple rapide**")
    for ex in EXAMPLES:
        short = ex[:45] + "…" if len(ex) > 45 else ex
        if st.button(short, key=ex, use_container_width=True):
            st.session_state["input_text"] = ex

with col_main:
    text_input = st.text_area(
        "Text de analizat:", height=100,
        value=st.session_state["input_text"],
        placeholder='e.g. "Oh great, another meeting!"',
        label_visibility="collapsed",
    )
    st.session_state["input_text"] = text_input

    btn_col, _ = st.columns([1, 2])
    with btn_col:
        run = st.button("🔍 Analizează", type="primary", use_container_width=True)

    # ── Rezultate — rămân în col_main ─────────────────────────
    if run and text_input and text_input.strip():
        with st.spinner("Se procesează..."):
            r = full_analysis(text_input.strip(), model, tokenizer, labels, threshold)

        # ── Rând 1: badge sarcasm + gauge ───────────────────────
        s_col, g_col = st.columns([2, 1])
        with s_col:
            box_class = "sarc-box-yes" if r["is_sarcastic"] else "sarc-box-no"
            icon  = "⚠️" if r["is_sarcastic"] else "✅"
            label = "Sarcasm detectat" if r["is_sarcastic"] else "Non-sarcastic"
            st.markdown(f"""
            <div class="{box_class}">
                <div class="sarc-title">{icon} {label}</div>
            </div>""", unsafe_allow_html=True)

            methods = [("Neural", r["neural_pos"]), ("Rule-based", r["rule_pos"]), ("Contrast", r["contrast_hit"])]
            pills = "".join(
                f'<span class="method-pill {"pill-on" if on else "pill-off"}">{"✓" if on else "✗"} {name}</span>'
                for name, on in methods
            )
            st.markdown(pills, unsafe_allow_html=True)
            if r["rule_hits"]:
                pos_e, neg_e, pp, pn = r["rule_hits"][0]
                st.caption(f"Rule: **{pos_e}** ({pp:.0%}) + **{neg_e}** ({pn:.0%})")
            if r["contrast_detail"]:
                a, b = r["contrast_detail"]
                st.caption(f"Contrast: «{a[0]}» → {a[1]} ➜ «{b[0]}» → {b[1]}")

        with g_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r["sarc_neural"] * 100,
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 9}},
                    "bar": {"color": "#e74c3c" if r["neural_pos"] else "#27ae60"},
                    "steps": [
                        {"range": [0, 62],   "color": "#1a2a1a"},
                        {"range": [62, 100], "color": "#3d1515"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": 62}
                },
                number={"suffix": "%", "font": {"size": 16}},
            ))
            fig_gauge.update_layout(
                height=140, margin=dict(t=10, b=0, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)", font_color="white"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Rând 2: grafic emoții ────────────────────────────
        top = r["scored"][:6]
        labels_plot = [f"{EMOTION_EMOJI.get(l,'')} {l}" for l, _ in top]
        values      = [p for _, p in top]
        active_set  = {l for l, _ in r["active"]}

        # Colorează mereu după polaritate; dimmed dacă sub prag
        def bar_color(lbl):
            base = "#2ecc71" if lbl in POSITIVE else "#e74c3c" if lbl in NEGATIVE else "#F39C12"
            return base if lbl in active_set else (
                "#1a4a2a" if lbl in POSITIVE else "#4a1a1a" if lbl in NEGATIVE else "#4a3a10"
            )

        fig = go.Figure(go.Bar(
            x=values, y=labels_plot, orientation="h",
            marker_color=[bar_color(l) for l, _ in top],
            text=[f"{v:.0%}" for v in values], textposition="outside",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig.add_vline(x=threshold, line_dash="dot", line_color="#555",
                      annotation_text="prag", annotation_font_color="#666",
                      annotation_font_size=10)
        fig.update_layout(
            xaxis=dict(range=[0, 1.18], tickformat=".0%", showgrid=False, showticklabels=False),
            yaxis=dict(autorange="reversed"),
            height=210, margin=dict(t=6, b=6, l=5, r=50),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

        if r["active"]:
            detected = " · ".join(
                f"{EMOTION_EMOJI.get(l,'')} **{l}** `{p:.0%}`" for l, p in r["active"]
            )
            st.caption(f"Detectate (>{threshold}): {detected}")
        else:
            st.caption(f"Nicio emoție nu depășește pragul {threshold} — probabil neutru.")

    elif run:
        st.warning("Introdu un text mai întâi.")
