"""
Linguistic Preprocessing & Explainable AI (XAI) Feature Attribution Engine
Natural Language Processing, Lemmatization, TF-IDF Vectorization & Decision Trees
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

# Load SpaCy pipeline for Linguistic Processing
print("  [NLP Engine] Loading SpaCy linguistic parser (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# ─────────────────────────────────────────────────────────────────────────────
# 1. Linguistic Preprocessing & Morpheme Extraction
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

# Strict question detection: Only triggers when user is genuinely querying the counselor
EXPLICIT_QUESTION_STARTS = [
    r'^(what|why|when|where|which|who)\s+(is|are|do|does|can|will|should|would|types|kind)\b',
    r'^(can\s+i|could\s+i|should\s+i|is\s+it|is\s+there|will\s+i|do\s+i|how\s+do\s+i|how\s+can\s+i)\b',
    r'^(difference\s+between|tell\s+me\s+about|what\s+about|explain\s+to\s+me)\b'
]

def is_interrogative(text: str) -> bool:
    """
    Detects whether user is posing a direct question to the counselor.
    Ignores descriptive statements like 'like how we can automate things'.
    """
    if not text:
        return False
    cleaned = text.lower().strip()
    
    # Direct question mark at the end or inside
    if "?" in cleaned:
        return True
        
    # Check if utterance begins with a direct interrogative prompt
    for pat in EXPLICIT_QUESTION_STARTS:
        if re.search(pat, cleaned):
            return True
            
    return False

def extract_representative_quotes(
    user_utterances: List[str], 
    top_traits: List[Dict],
    primary_degree: Dict = None
) -> List[Dict[str, str]]:
    """
    Extracts 2 distinct, substantive student quotes from the dialogue and grounds them
    SPECIFICALLY in the recommended university degree & Indian stream.
    Filters out generic selections ('the first one', 'yes', 'coding', 'i did not get your question').
    """
    junk_exact = {
        "the first one", "the second one", "first one", "second one", "first", "second",
        "1", "2", "option 1", "option 2", "yes", "no", "ok", "okay", "sure", "coding", "code",
        "i did not get your question", "i didnot get your question", "i did not understand",
        "i didnot understand", "what do you mean", "none", "neither", "both", "teamwork", "team"
    }
    
    valid = []
    for u in user_utterances:
        clean = u.strip()
        clean_lower = clean.lower().rstrip(".,!?")
        if clean_lower in junk_exact:
            continue
        if len(clean) < 12:
            continue
        if clean_lower in ["10th standard", "12th standard", "other", "science (pcm)", "science (pcb)", "commerce & finance", "liberal studies / arts"]:
            continue
        if clean_lower.startswith("i don't like") or clean_lower.startswith("i didnt like"):
            continue
        valid.append(clean)
            
    deg_name = primary_degree.get("degree", "your recommended program") if primary_degree else "your degree"
    deg_id = primary_degree.get("id", "") if primary_degree else ""
    stream_name = primary_degree.get("stream", "Science") if primary_degree else "Science"
    
    if not valid:
        return [
            {
                "quote": "Your exploration across technical problem-solving and structured systems",
                "explanation": f"Directly aligns with {deg_name} in {stream_name}, requiring analytical reasoning and structured execution."
            }
        ]
        
    def score_quote_for_degree(q_text: str) -> float:
        score = 1.0
        q_low = q_text.lower()
        # High-value domain keywords
        if deg_id == "btech_cs":
            for kw in ["software", "app", "mobile", "backend", "system", "automate", "logic", "algorithm", "hardwares", "user side"]:
                if kw in q_low:
                    score += 3.0
        elif deg_id == "btech_mech_robotics":
            for kw in ["hardware", "robot", "machine", "mechanical", "assemble", "tools", "engine"]:
                if kw in q_low:
                    score += 3.0
        elif deg_id == "bdes_uiux":
            for kw in ["design", "ui", "ux", "visual", "art", "user", "experience", "creative", "screens"]:
                if kw in q_low:
                    score += 3.0
        elif deg_id in ["ca_bcom", "bba_analytics"]:
            for kw in ["business", "marketing", "finance", "money", "audit", "account", "startup"]:
                if kw in q_low:
                    score += 3.0
        elif deg_id in ["mbbs_healthcare", "bsc_lifesciences"]:
            for kw in ["bio", "biology", "patient", "doctor", "health", "hospital", "lab", "medicine"]:
                if kw in q_low:
                    score += 3.0
        # Prefer descriptive sentences over short fragments
        score += min(2.0, len(q_text) / 40.0)
        return score

    # Sort valid quotes by degree relevance
    ranked_quotes = sorted(valid, key=score_quote_for_degree, reverse=True)
    
    # Pick top 2 distinct quotes
    selected = [ranked_quotes[0]]
    if len(ranked_quotes) > 1:
        # Pick second that is distinct
        for cand in ranked_quotes[1:]:
            if cand.lower() != selected[0].lower() and not (cand.lower() in selected[0].lower() or selected[0].lower() in cand.lower()):
                selected.append(cand)
                break
        if len(selected) == 1:
            selected.append(ranked_quotes[1])
            
    def generate_degree_explanation(quote: str) -> str:
        q_low = quote.lower()
        if deg_id == "btech_cs":
            if any(k in q_low for k in ["softwares over hardwares", "prefer software"]):
                return f"Directly confirms your decisive focus on Computer Science and software engineering over mechanical hardware."
            elif any(k in q_low for k in ["backend", "logic behind", "system"]):
                return f"Aligns with core B.Tech CS coursework in distributed backend systems, database architecture, and algorithmic logic."
            elif any(k in q_low for k in ["mobile app", "apps", "app works"]):
                return f"Directly matches modern mobile app engineering, software product design, and client-side architecture."
            elif any(k in q_low for k in ["automate", "manual", "common people"]):
                return f"Reflects the core purpose of AI & Computer Science: automating repetitive tasks to solve real-world human needs."
            elif any(k in q_low for k in ["user side", "issues", "technically"]):
                return f"Exemplifies Human-Centered Software Engineering, bridging user problem discovery with scalable technical code."
            elif any(k in q_low for k in ["team", "collaborat", "divide"]):
                return f"Demonstrates Agile sprint collaboration, an essential skillset for professional software development teams."
            else:
                return f"Demonstrates the algorithmic curiosity and technical problem-solving instinct required in {deg_name}."
        elif deg_id == "bdes_uiux":
            if any(k in q_low for k in ["user side", "user experience", "experience"]):
                return f"Directly reflects Human-Computer Interaction (HCI) and user empathy mapping in {deg_name}."
            elif any(k in q_low for k in ["visual", "screen", "interface", "design"]):
                return f"Matches core studio coursework in digital wireframing, typography, and visual design systems."
            else:
                return f"Demonstrates the creative prototyping and aesthetic design thinking demanded by {deg_name}."
        elif deg_id in ["ca_bcom", "bba_analytics"]:
            if any(k in q_low for k in ["finance", "budget", "money", "audit", "account"]):
                return f"Matches rigorous corporate accounting, taxation compliance, and statutory balance sheet auditing in {deg_name}."
            else:
                return f"Demonstrates the strategic commercial mindset and client leadership needed in {deg_name}."
        elif deg_id in ["mbbs_healthcare", "bsc_lifesciences"]:
            return f"Aligns with empirical laboratory investigation and diagnostic patient care essential for {deg_name}."
        else:
            return f"Grounded in your specific answer, matching core academic expectations for {deg_name} ({stream_name})."

    results = []
    for s_quote in selected:
        disp_quote = s_quote if len(s_quote) <= 90 else s_quote[:87] + "..."
        results.append({
            "quote": disp_quote,
            "explanation": generate_degree_explanation(s_quote)
        })
        
    return results

def detect_uncertainty(text: str) -> bool:
    """Detects whether user is expressing hesitation, indecisiveness, or saying 'idk'."""
    cleaned = normalize_chat_slang(text.lower())
    for pat in UNCERTAINTY_PATTERNS:
        if re.search(pat, cleaned):
            return True
    return False

def preprocess_linguistic_text(text: str) -> Dict:
    """
    Applies NLP Preprocessing & Linguistic Extraction Pipeline:
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
# 2. Sparse TF-IDF Vectorization & Classical Tree Classifier
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
    Provides verifiable XAI path logs for decision tree traversal.
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
