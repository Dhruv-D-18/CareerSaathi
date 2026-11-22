"""
Linguistic Preprocessing & Explainable AI (XAI) Feature Attribution Engine
CSE708 NLP Lab Practical — Syllabus Sessions 7–16 (POS, Lemmatization, TF-IDF, Trees)
"""

import re
import spacy
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Import RIASEC definitions
from riasec import RIASEC_CODES

# Load SpaCy pipeline for Sessions 7-10 (Linguistic Processing)
print("  [NLP Engine] Loading SpaCy linguistic parser (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# ─────────────────────────────────────────────────────────────────────────────
# 1. Linguistic Preprocessing & Morpheme Extraction (Sessions 7–10)
# ─────────────────────────────────────────────────────────────────────────────

CHAT_ABBREVIATIONS = {
    r'\bidk\b': 'i do not know',
    r'\bidc\b': 'i do not care',
    r'\bdunno\b': 'do not know',
    r'\btbh\b': 'to be honest',
    r'\bimo\b|\bimho\b': 'in my opinion',
    r'\bcoz\b|\bbc\b|\bbcoz\b': 'because',
    r'\bcs\b': 'computer science',
    r'\bai\b': 'artificial intelligence',
    r'\bml\b': 'machine learning',
    r'\bengg\b': 'engineering',
    r'\bmaths\b': 'mathematics',
    r'\bbiz\b': 'business',
    r'\bgovt\b': 'government',
    r'\bpls\b|\bplz\b': 'please',
    r'\btech\b': 'technology'
}

UNCERTAINTY_PATTERNS = [
    r'\b(i\s*do\s*not\s*know|dont\s*know|don\'t\s*know|idk|dunno)\b',
    r'\b(not\s*sure|no\s*idea|confused|cant\s*decide|can\'t\s*decide)\b',
    r'\b(both|either|any|whatever|anything)\b'
]

def normalize_chat_slang(text: str) -> str:
    """Expands common abbreviations and informal chat slang into canonical English."""
    cleaned = text.lower()
    for pattern, replacement in CHAT_ABBREVIATIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned

QUESTION_PATTERNS = [
    r'\?',
    r'\b(what|why|how|when|where|which|who|whose)\b',
    r'\b(can\s+i|could\s+i|should\s+i|is\s+it|is\s+there|will\s+i|do\s+i|are\s+there)\b',
    r'\b(difference\s+between|tell\s+me\s+about|what\s+about|explain)\b'
]

def is_interrogative(text: str) -> bool:
    """Detects whether user is posing a question rather than purely answering."""
    if not text:
        return False
    cleaned = text.lower().strip()
    if "?" in cleaned:
        return True
    for pat in QUESTION_PATTERNS:
        if re.search(pat, cleaned):
            return True
    return False

def extract_representative_quotes(user_utterances: List[str], top_traits: List[Dict]) -> List[Dict[str, str]]:
    """
    Extracts 2 distinct, representative student quotes from the dialogue to ground the
    Claude-style 'Why this fits you' evidence card.
    """
    valid = []
    # Filter out trivial button responses or 1-word inputs
    for u in user_utterances:
        clean = u.strip()
        if len(clean) >= 12 and clean.lower() not in ["10th standard", "12th standard", "other", "science (pcm)", "science (pcb)", "commerce & finance", "liberal studies / arts"]:
            valid.append(clean)
            
    if not valid:
        return [
            {
                "quote": "Your exploration across technical and structured challenges",
                "explanation": f"Points to strong {top_traits[0]['name']} aptitude, which requires analytical curiosity."
            }
        ]
        
    t1_name = top_traits[0]["name"]
    t2_name = top_traits[1]["name"] if len(top_traits) > 1 else "Strategic"
    
    quotes = []
    # Quote 1: Earliest substantial utterance (interests/instinct)
    q1 = valid[0]
    if len(q1) > 85:
        q1 = q1[:82] + "..."
    quotes.append({
        "quote": q1,
        "explanation": f"Points to natural {t1_name.lower()} instincts, which this discipline actively demands."
    })
    
    # Quote 2: Later utterance (thinking style or execution)
    if len(valid) > 1:
        q2 = valid[-1] if valid[-1] != valid[0] else (valid[1] if len(valid) > 2 else valid[-1])
        if len(q2) > 85:
            q2 = q2[:82] + "..."
        quotes.append({
            "quote": q2,
            "explanation": f"Demonstrates {t2_name.lower()} problem-solving habits that succeed in this career pathway."
        })
    elif len(valid) == 1:
        quotes.append({
            "quote": "Consistent preference for structured execution",
            "explanation": "Highlights your systematic approach to breaking down complex challenges."
        })
        
    return quotes

def detect_uncertainty(text: str) -> bool:
    """Detects whether user is expressing hesitation, indecisiveness, or saying 'idk'."""
    cleaned = normalize_chat_slang(text.lower())
    for pat in UNCERTAINTY_PATTERNS:
        if re.search(pat, cleaned):
            return True
    return False

def preprocess_linguistic_text(text: str) -> Dict:
    """
    Applies Sessions 7-10 NLP pipeline:
    - Text normalization & Abbreviation expansion
    - Tokenization & Lemmatization
    - Syntactic POS Tagging
    - Extracts Action Verbs & Key Noun Chunks
    """
    if not text or not text.strip():
        return {
            "cleaned_text": "",
            "tokens": [],
            "lemmas": [],
            "action_verbs": [],
            "noun_chunks": [],
            "is_uncertain": False
        }
        
    normalized = normalize_chat_slang(text)
    is_uncertain = detect_uncertainty(text)
    doc = nlp(normalized)
    
    tokens = []
    lemmas = []
    action_verbs = []
    
    for token in doc:
        # Filter punctuation and pure numbers
        if not token.is_punct and not token.is_space:
            tokens.append(token.text.lower())
            lemmas.append(token.lemma_.lower())
            
            # Extract Action Verbs (VERB POS tag)
            if token.pos_ in ["VERB", "AUX"] and not token.is_stop:
                action_verbs.append(token.lemma_.lower())
                
    # Extract Key Noun Phrases (Syntactic Parsing)
    noun_chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]
    
    return {
        "cleaned_text": " ".join(lemmas),
        "tokens": tokens,
        "lemmas": lemmas,
        "action_verbs": list(set(action_verbs))[:6],
        "noun_chunks": list(set(noun_chunks))[:6],
        "is_uncertain": is_uncertain
    }

# ─────────────────────────────────────────────────────────────────────────────
# 2. Sparse TF-IDF Vectorization & Classical Tree Classifier (Sessions 11–16)
# ─────────────────────────────────────────────────────────────────────────────

# Build a comprehensive training corpus from RIASEC anchors + synthetic variations
X_train_raw = []
y_train_raw = []

for code, meta in RIASEC_CODES.items():
    anchors = meta["anchors"]
    for sentence in anchors:
        doc = nlp(sentence)
        lemmas = " ".join([token.lemma_.lower() for token in doc if not token.is_punct])
        X_train_raw.append(lemmas)
        y_train_raw.append(code)
        
    # Generate pairwise synthetic mixtures to enrich the feature matrix
    for i in range(len(anchors) - 1):
        combo = anchors[i] + " Furthermore, " + anchors[i + 1]
        doc = nlp(combo)
        lemmas = " ".join([token.lemma_.lower() for token in doc if not token.is_punct])
        X_train_raw.append(lemmas)
        y_train_raw.append(code)

print(f"  [NLP Engine] Fitting TF-IDF N-Gram Vectorizer on {len(X_train_raw)} linguistic samples...")
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
vocab_size = len(tfidf_vectorizer.vocabulary_)
print(f"  [NLP Engine] [OK] TF-IDF Sparse Feature Matrix generated: V = {vocab_size} vocabulary dimensions.")

# Train Random Forest Classifier for smooth feature attribution
print("  [NLP Engine] Fitting Random Forest Classifier (150 Tree Ensembles)...")
rf_classifier = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train_raw)

# Train Decision Tree for explainable node splits
print("  [NLP Engine] Fitting Decision Tree Classifier (max_depth=5 for XAI splits)...")
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train_tfidf, y_train_raw)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Real-Time Explainable AI (XAI) Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_linguistic_attributions(user_text: str) -> List[Dict]:
    """
    Extracts individual words from user text, validates them against the TF-IDF
    vocabulary, and predicts which RIASEC dimension each word attributes to.
    """
    if not user_text or not user_text.strip():
        return []
        
    doc = nlp(user_text)
    feature_names = set(tfidf_vectorizer.get_feature_names_out())
    
    highlights = []
    seen = set()
    
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in feature_names and lemma not in seen and not token.is_stop and len(lemma) > 2:
            seen.add(lemma)
            # Transform single token vector
            vec = tfidf_vectorizer.transform([lemma])
            pred_code = rf_classifier.predict(vec)[0]
            cat_name = RIASEC_CODES[pred_code]["name"]
            
            highlights.append({
                "word": lemma,
                "code": pred_code,
                "category": cat_name,
                "color": RIASEC_CODES[pred_code]["color"]
            })
            
    return highlights[:10]

def trace_decision_tree_path(user_text: str) -> List[Dict]:
    """
    Traces the Decision Tree binary split path step-by-step for the user input.
    Provides verifiable XAI path logs for Sessions 14-16.
    """
    if not user_text or not user_text.strip():
        return []
        
    doc = nlp(user_text)
    lemmatized = " ".join([token.lemma_.lower() for token in doc if not token.is_punct])
    vec = tfidf_vectorizer.transform([lemmatized]).toarray()[0]
    
    tree = dt_classifier.tree_
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    path = []
    node_id = 0
    
    # Traverse until leaf node
    while tree.children_left[node_id] != tree.children_right[node_id]:
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_idx]
        
        user_val = vec[feature_idx]
        is_greater = user_val > threshold
        
        desc = f"Does answer emphasize '{feature_name}' keyword?"
        decision = f"Yes (weight {float(user_val):.2f} > {float(threshold):.2f})" if is_greater else f"No (weight {float(user_val):.2f} <= {float(threshold):.2f})"
        
        path.append({
            "node_id": int(node_id),
            "feature": feature_name,
            "description": desc,
            "decision": decision,
            "outcome": "right" if is_greater else "left"
        })
        
        if is_greater:
            node_id = tree.children_right[node_id]
        else:
            node_id = tree.children_left[node_id]
            
    # Leaf reached
    leaf_val = tree.value[node_id][0]
    class_idx = np.argmax(leaf_val)
    predicted_code = dt_classifier.classes_[class_idx]
    
    path.append({
        "node_id": int(node_id),
        "feature": "leaf",
        "description": "Decision Node Finalized",
        "decision": f"Dominant Bias: {RIASEC_CODES[predicted_code]['name']} ({predicted_code})",
        "outcome": "leaf"
    })
    
    return path[:5]
