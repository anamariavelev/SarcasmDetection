import os, re, torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import streamlit as st

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "saved_model_fast"))
COMPACT  = os.path.join(_ROOT, "compact_checkpoint.pth")
LEGACY   = os.path.join(_ROOT, "heads_only.pth")
TOK_DIR  = _ROOT

SARCASM_COMBOS = {
    ("amusement","annoyance"),("approval","disapproval"),("joy","anger"),
    ("optimism","disappointment"),("excitement","disgust"),("love","anger"),
    ("gratitude","annoyance"),("joy","disgust"),("excitement","annoyance"),
    ("admiration","disgust"),("love","disgust"),("optimism","anger"),
    ("joy","disapproval"),("amusement","disgust"),("pride","embarrassment"),
    ("excitement","disappointment"),("approval","anger"),("admiration","annoyance"),
}

POSITIVE = {"admiration","amusement","approval","caring","desire","excitement",
            "gratitude","joy","love","optimism","pride","relief"}
NEGATIVE = {"anger","annoyance","disappointment","disapproval","disgust",
            "embarrassment","fear","grief","nervousness","remorse","sadness"}

EMOTION_EMOJI = {
    "admiration":"🌟","amusement":"😄","anger":"😠","annoyance":"😒",
    "approval":"👍","caring":"🤗","confusion":"😕","curiosity":"🤔",
    "desire":"💫","disappointment":"😞","disapproval":"👎","disgust":"🤢",
    "embarrassment":"😳","excitement":"🎉","fear":"😨","gratitude":"🙏",
    "grief":"😢","joy":"😊","love":"❤️","nervousness":"😰",
    "optimism":"🌈","pride":"😤","realization":"💡","relief":"😮‍💨",
    "remorse":"😔","sadness":"😭","surprise":"😲","neutral":"😐",
}

EMOTION_COLOR = {e: ("green" if e in POSITIVE else "red" if e in NEGATIVE else "orange")
                 for e in EMOTION_EMOJI}


class MultiHeadXLMR(nn.Module):
    def __init__(self, backbone, dropout, n_emotions):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(dropout)
        self.emotions_head = nn.Linear(768, n_emotions)
        self.sarcasm_head  = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0, :])
        return self.emotions_head(cls), self.sarcasm_head(cls)


@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
    if os.path.exists(COMPACT):
        ck = torch.load(COMPACT, map_location="cpu", weights_only=False)
        backbone = AutoModel.from_pretrained(ck["base_model"])
        model = MultiHeadXLMR(backbone, ck["dropout"], len(ck["labels_list"]))
        remapped = {
            k.replace("base.", "backbone.", 1) if k.startswith("base.") else k: v
            for k, v in ck["fine_tuned_weights"].items()
        }
        state = model.state_dict(); state.update(remapped); model.load_state_dict(state)
    else:
        ck = torch.load(LEGACY, map_location="cpu", weights_only=False)
        backbone = AutoModel.from_pretrained(ck["base_model"])
        model = MultiHeadXLMR(backbone, ck["dropout"], len(ck["labels_list"]))
        model.emotions_head.load_state_dict(ck["emotions_head"])
        model.sarcasm_head.load_state_dict(ck["sarcasm_head"])
    model.eval()
    return model, tokenizer, ck["labels_list"], ck["best_threshold"]


def run_inference(text, model, tokenizer, labels, threshold):
    enc = tokenizer(text, return_tensors="pt", max_length=128,
                    truncation=True, padding="max_length")
    with torch.no_grad():
        logits_emo, logit_sarc = model(enc["input_ids"], enc["attention_mask"])
    probs = torch.sigmoid(logits_emo)[0].tolist()
    sarc_prob = torch.sigmoid(logit_sarc)[0].item()
    scored = sorted(zip(labels, probs), key=lambda x: -x[1])
    prob_dict = dict(zip(labels, probs))
    return scored, sarc_prob, prob_dict


def rule_based_sarcasm(prob_dict):
    top3 = {lbl for lbl, _ in sorted(prob_dict.items(), key=lambda x: -x[1])[:3]}
    triggered = []
    for (pos_emo, neg_emo) in SARCASM_COMBOS:
        p_pos, p_neg = prob_dict.get(pos_emo, 0), prob_dict.get(neg_emo, 0)
        if p_pos >= 0.65 and pos_emo in top3 and p_neg >= 0.50 and p_neg/p_pos >= 0.45:
            triggered.append((pos_emo, neg_emo, p_pos, p_neg))
    return triggered


def contrast_sarcasm(text, model, tokenizer, labels):
    parts = [p.strip() for p in re.split(r'[.!?,;]+', text) if len(p.strip()) > 4]
    if len(parts) < 2:
        return False, None
    scores = []
    for part in parts[:3]:
        scored, _, pd = run_inference(part, model, tokenizer, labels, 0.5)
        top_lbl, top_prob = scored[0]
        polarity = "pos" if top_lbl in POSITIVE else "neg" if top_lbl in NEGATIVE else "neu"
        scores.append((part, top_lbl, top_prob, polarity))
    for i in range(len(scores) - 1):
        a, b = scores[i], scores[i+1]
        if a[3]=="pos" and a[2]>=0.55 and b[3]=="neg" and b[2]>=0.40:
            return True, (a, b)
        if a[3]=="pos" and a[2]>=0.75 and b[3]!="pos":
            return True, (a, b)
    return False, None


def full_analysis(text, model, tokenizer, labels, threshold):
    scored, sarc_neural, prob_dict = run_inference(text, model, tokenizer, labels, threshold)
    rule_hits  = rule_based_sarcasm(prob_dict)
    c_hit, c_detail = contrast_sarcasm(text, model, tokenizer, labels)
    neural_pos = sarc_neural >= 0.62
    rule_pos   = len(rule_hits) > 0
    is_sarc    = neural_pos or rule_pos or c_hit
    active     = [(l, p) for l, p in scored if p >= threshold]
    return {
        "scored": scored, "prob_dict": prob_dict,
        "sarc_neural": sarc_neural, "rule_hits": rule_hits,
        "contrast_hit": c_hit, "contrast_detail": c_detail,
        "neural_pos": neural_pos, "rule_pos": rule_pos,
        "is_sarcastic": is_sarc, "active": active,
    }
