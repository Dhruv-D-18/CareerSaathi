# 🎓 CareerSaathi: Cognitive Career & Admissions Advisory Suite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Sentence-BERT](https://img.shields.io/badge/Sentence--BERT-all--MiniLM--L6--v2-orange.svg?style=flat)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![spaCy](https://img.shields.io/badge/spaCy-en__core__web__sm-09A3D5?style=flat&logo=spacy)](https://spacy.io/)
[![Ollama](https://img.shields.io/badge/Local_SLM-Gemma--2:2B-black?style=flat&logo=ollama)](https://ollama.com/)
[![Institution](https://img.shields.io/badge/Navrachana_University-Vadodara-a81d24.svg?style=flat)](https://nuv.ac.in/)

> **Cognitive AI & Natural Language Processing Career Advisory System**  
> **Student Name:** Dhruv Dharikar  
> **Roll Number:** 23000527  
> **Faculty Guide:** Dr. Jaideepsinh Raulji  
> **Department:** Computer Science & Engineering, School of Engineering & Technology, Navrachana University, Vadodara  

---

## 🌟 Executive Summary

**CareerSaathi** is an explainable, mixed-initiative career counselling and admissions advisory agent designed for Indian high school students navigating critical academic crossroads (**After 10th Stream Selection** and **After 12th Undergraduate Degree Discovery**).

Departing from rigid 50-question psychometric questionnaires, CareerSaathi engages students in natural, empathetic dialogue grounded in realistic life scenarios. It models cognitive preferences in continuous space, compiles a live side-by-side **Admissions Dossier Artifact** (inspired by Claude Artifacts), provides **full NLP Explainability (XAI)**, and prints official, university-branded PDF advisories in Navrachana Crimson Red (`#a81d24`).

---

## 🧠 Core NLP & Machine Learning Architecture

```
                  ┌────────────────────────────────────────┐
                  │          Student Utterance             │
                  └───────────────────┬────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
  ┌──────────────────────────┐                 ┌──────────────────────────┐
  │   Linguistic Parser      │                 │   Dense S-BERT Embedder  │
  │   (spaCy POS + Lemmatizer│                 │   (all-MiniLM-L6-v2)     │
  │   & Text Normalization)  │                 │   Mapping text to R^384  │
  └────────────┬─────────────┘                 └─────────────┬────────────┘
               │                                             │
               │ (N-Grams / Morphemes)                       ▼
               │                               ┌──────────────────────────┐
               │                               │ Cosine Sim to 6 Holland  │
               │                               │ RIASEC Centroid Vectors  │
               │                               └─────────────┬────────────┘
               │                                             │
               ▼                                             ▼
  ┌──────────────────────────┐                 ┌──────────────────────────┐
  │ Classical TF-IDF Matrix  │                 │ Temperature-Softmax      │
  │ (V = 439) + Decision Tree│                 │ Probability Distribution │
  │ Node-by-Node XAI Tracer  │                 └─────────────┬────────────┘
  └────────────┬─────────────┘                               │
               │                                             ▼
               │                               ┌──────────────────────────┐
               │                               │ Bayesian Belief Update   │
               │                               │ S_t = α·S_{t-1} + (1-α)a │
               │                               └─────────────┬────────────┘
               │                                             │
               ▼                                             ▼
        ┌───────────────────────────────────────────────────────────┐
        │  Information Theory: Normalized Shannon Entropy (H_norm)  │
        │  Convergence Detection & Indian Stream Alignment Matrix  │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────────────┐
        │        Local On-Device Generation: Gemma-2:2B (Ollama)    │
        │   Contextual Mentoring, Shorthand Resolution, Zero-Preach │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
        ┌───────────────────────────────────────────────────────────┐
        │  Claude-Style Side-by-Side Admissions Dossier Artifact    │
        │  Personalized Grounded Evidence Cards & Printable PDF     │
        └───────────────────────────────────────────────────────────┘
```

1. **Syntactic & Linguistic Morpheme Extraction**:
   * SpaCy tokenization, lemmatization, syntactic dependency parsing, and action verb extraction.
   * Informal abbreviation normalization (`"idk"`, `"coding"`, `"tech"`).
2. **Dense Semantic Embedding Space**:
   * **Sentence-BERT (`all-MiniLM-L6-v2`)** maps utterances into dense vector space $\mathbb{R}^{384}$.
   * Six Holland RIASEC prototype centroids $\mu_k \in \mathbb{R}^{384}$ pre-computed from behavioral anchor corpora.
   * Scaled via Temperature Softmax ($T = 10.0$) across Realistic, Investigative, Artistic, Social, Enterprising, and Conventional poles.
3. **Bayesian Momentum Belief State Updating**:
   * Smooths preference transitions across turns:
     $$S_t = \alpha S_{t-1} + (1 - \alpha) a_t \quad (\alpha = 0.55)$$
   * Prevents single-utterance instability while smoothly accumulating multi-turn student interests.
4. **Information Theory & Shannon Entropy**:
   * Computes normalized entropy:
     $$H_{\text{norm}} = - \frac{1}{\log_2 6} \sum_{k=1}^6 p_k \log_2 p_k$$
   * Automatically detects ambiguity ($H > 0.85$) or convergence ($H < 0.65$) to transition dialogue from probing to synthesis.
5. **Human-in-the-Loop Active Recalibration**:
   * *"I didn't like this suggestion"* dynamically dampens rejected disciplines, incorporates negative feedback constraints, and recalibrates to alternative university pathways.
6. **On-Device Local SLM Generation (Gemma-2:2B via Ollama)**:
   * 100% offline, local GPU inference (NVIDIA RTX 4060).
   * Context-conditioned prompting with strict anti-loop topic memory and natural conversational completion.

---

## 🎨 UI/UX: Navrachana Crimson Red & Claude Split-Screen

* **Signature Palette**: Navrachana University Crimson Red (`#a81d24`), crisp white backgrounds, and refined Lora serif typography.
* **Claude-Style Artifact Split Pane**: Slides out side-by-side upon Turn 5 convergence or recalibration, preserving dialogue context on the left while presenting the live admissions dossier on the right.
* **Grounding Evidence Cards ("Why this fits you")**: Extracts substantive student quotes and generates degree-specific justifications connecting their exact words to program coursework.
* **Official University PDF Export**: High-fidelity `@media print` stylesheet with forced exact color reproduction (`-webkit-print-color-adjust: exact`), rendering the signature crimson hero card, colorful RIASEC bars, and university crest.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
* Python 3.10+
* Local Ollama running `gemma2:2b` (`ollama run gemma2:2b`)

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/Dhruv-D-18/CareerSaathi.git
cd CareerSaathi

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run Application
```powershell
python run.py
```
Open your browser at `http://localhost:8080`.
