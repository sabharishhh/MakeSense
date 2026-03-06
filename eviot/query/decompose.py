import re
import torch
import streamlit as st
from eviot.encoders.encoder import Encoder

@st.cache_resource
def load_encoder():
    return Encoder()

encoder = load_encoder()

STOPWORDS = {
    "what", "how", "does", "do", "can", "is", "are",
    "the", "a", "an", "of", "in", "on", "for", "to", "with"
}

INTERROGATIVES = {
    "what", "how", "does", "do", "can", "is", "are"
}

def normalize(text: str) -> str:
    return text.lower().strip()


def is_interrogative(text: str) -> bool:
    tokens = text.split()
    return tokens and tokens[0] in INTERROGATIVES

def tokenize(text: str):
    return re.findall(r"[A-Za-z]+", text.lower())

def generate_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def suppress_subphrases(phrases):
    final = []
    for p in phrases:
        drop = False
        for q in phrases:
            if p == q:
                continue
            if p in q and len(q.split()) > len(p.split()):
                drop = True
                break
        if not drop:
            final.append(p)
    return final


def extract_phrases(query: str, max_phrases: int = 10):

    tokens = tokenize(query)

    query_tokens = set(tokens)

    candidates = []

    for t in tokens:
        if t not in STOPWORDS:
            candidates.append(t)

    for bg in generate_ngrams(tokens, 2):
        words = bg.split()
        if all(w not in STOPWORDS for w in words):
            candidates.append(bg)

    for tg in generate_ngrams(tokens, 3):
        words = tg.split()
        if sum(w not in STOPWORDS for w in words) >= 2:
            candidates.append(tg)

    seen = set()
    candidates = [
        normalize(c) for c in candidates
        if not (normalize(c) in seen or seen.add(normalize(c)))
    ]

    filtered = []
    for c in candidates:

        if is_interrogative(c):
            continue

        overlap = len(set(c.split()) & query_tokens) / max(len(query_tokens), 1)

        if overlap < 0.7:
            filtered.append(c)

    if not filtered:
        filtered = tokens

    embs = encoder.encode(filtered)

    keep = []

    for i, e in enumerate(embs):
        if all(torch.cosine_similarity(e, embs[j], dim=0) < 0.9 for j in keep):
            keep.append(i)

    phrases = suppress_subphrases([filtered[i] for i in keep])
    phrases = phrases[:max_phrases]

    return phrases, encoder.encode(phrases)