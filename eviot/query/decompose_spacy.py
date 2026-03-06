import torch
import spacy
from spacy.cli import download
import streamlit as st
from eviot.encoders.encoder import Encoder

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_encoder():
    return Encoder()

nlp = load_spacy()
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

def content_words(span):
    return [
        t.text.lower()
        for t in span
        if t.is_alpha and t.text.lower() not in STOPWORDS
    ]

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

    doc = nlp(query)
    query_tokens = {t.text.lower() for t in doc if t.is_alpha}
    candidates = []

    for chunk in doc.noun_chunks:
        if len(content_words(chunk)) >= 2:
            candidates.append(chunk.text)

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.text.lower() not in STOPWORDS:
            candidates.append(token.text)

    for token in doc:
        if token.pos_ == "VERB":
            span = doc[token.left_edge.i: token.right_edge.i + 1]
            if len(content_words(span)) >= 2:
                candidates.append(span.text)

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
        filtered = [t.text.lower() for t in doc if t.is_alpha]

    embs = encoder.encode(filtered)

    keep = []

    for i, e in enumerate(embs):
        if all(torch.cosine_similarity(e, embs[j], dim=0) < 0.9 for j in keep):
            keep.append(i)

    phrases = suppress_subphrases([filtered[i] for i in keep])
    phrases = phrases[:max_phrases]

    return phrases, encoder.encode(phrases)

def group_phrases_into_facets(phrases, phrase_embs):

    facets = {
        "mechanism": [],
        "attribution": [],
        "definition": [],
        "other": []
    }

    for i, p in enumerate(phrases):

        if any(x in p for x in ["who", "developed", "invented", "created"]):
            facets["attribution"].append(i)

        elif any(x in p for x in ["how", "process", "work", "mechanism"]):
            facets["mechanism"].append(i)

        elif any(x in p for x in ["what", "define", "meaning"]):
            facets["definition"].append(i)

        else:
            facets["other"].append(i)

    facet_embs = {}

    for k, idxs in facets.items():
        if idxs:
            facet_embs[k] = phrase_embs[idxs]

    return facet_embs