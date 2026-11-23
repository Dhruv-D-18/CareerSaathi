"""
Linguistic Preprocessing & Explainable AI (XAI) Feature Attribution Engine
Natural Language Processing, Lemmatization, TF-IDF Vectorization & Decision Trees
"""

import re
import math
import spacy
from typing import List, Dict, Tuple, Optional
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
    r'^(difference\s+between|tell\s+me\s+about|what\s+about|explain\s+to\s+me)\b'
]

AUXILIARY_QUESTION_STARTS = [
    r'^(can\s+i|could\s+i|should\s+i|is\s+it|is\s+there|will\s+i|do\s+i|how\s+do\s+i|how\s+can\s+i)\b'
]

ADMISSIONS_TOPIC_WORDS = [
    "admission", "admissions", "entrance", "exam", "exams", "jee", "neet", "cuet", "gujcet", 
    "placement", "placements", "salary", "package", "fee", "fees", "scholarship", "syllabus", 
    "curriculum", "cutoff", "eligibility", "nuv", "navrachana", "degree", "college", "course"
]

def is_interrogative(text: str) -> bool:
    """
    Detects whether user is posing a direct question to the counselor.
    Ignores descriptive statements like 'like how we can automate things' or 'is it easy for a customer'.
    """
    if not text:
        return False
    cleaned = text.lower().strip()
    
    # Direct question mark at the end or inside
    if "?" in cleaned:
        return True
        
    # Check explicit WH-question starts
    for pat in EXPLICIT_QUESTION_STARTS:
        if re.search(pat, cleaned):
            return True
            
    # For auxiliary starts (e.g. 'is it', 'can i'), require explicit query intent or topic keywords
    for pat in AUXILIARY_QUESTION_STARTS:
        if re.search(pat, cleaned):
            # Only count as question if user is inquiring about admissions, exams, or courses
            if any(w in cleaned for w in ADMISSIONS_TOPIC_WORDS):
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
    Strictly filters out confusion, meta-dialogue, questions, and generic selections.
    Falls back gracefully to synthesized psychometric dimensions when quotes are sparse.
    """
    JUNK_META_PATTERNS = [
        r'\b(understand|understood)\b.*\b(question|meaning|properly)\b',
        r'\b(did\s*n[\'o]?t|do\s*not|dont|don\'t)\b.*\b(understand|get|follow)\b',
        r'\b(what\s+do\s+you\s+mean|what\s+does\s+that\s+mean|what\s+are\s+you\s+asking)\b',
        r'\b(can\s+you|please)\s+(repeat|rephrase|clarify|explain)\b',
        r'\b(i\s*do\s*not\s*know|dont\s*know|don\'t\s*know|idk|dunno|no\s*idea|not\s*sure)\b',
        r'\b(i\s*would\s*like\s*to\s*know|tell\s*me\s*more|what\s+types?\s+of\s+business)\b',
        r'\b(what\s+kind\s+of|which\s+types?\s+of)\b',
        r'^(the\s+)?(first|second|1st|2nd|1|2|option\s*[12])(\s+one)?$',
        r'^(yes|no|ok|okay|sure|yeah|yep|nope|fine|cool|nice|whatever|anything|both|neither|none)$',
        r'^(coding|code|programming|tech|business|maths|science)$'
    ]
    
    valid = []
    for u in user_utterances:
        clean = u.strip()
        clean_lower = clean.lower().rstrip(".,!?")
        
        # Filter stream tags and negation
        if clean_lower in ["10th standard", "12th standard", "other", "science (pcm)", "science (pcb)", "commerce & finance", "liberal studies / arts"]:
            continue
        if clean_lower.startswith("i don't like") or clean_lower.startswith("i didnt like"):
            continue
            
        # Filter meta-dialogue, confusion, or filler
        is_junk = False
        for pat in JUNK_META_PATTERNS:
            if re.search(pat, clean_lower):
                is_junk = True
                break
        if is_junk:
            continue
            
        # Must be substantive length
        if len(clean) < 14:
            continue
            
        # Must not be a pure question directed at the counselor
        if clean.endswith("?") or clean_lower.startswith(("what is", "how much", "tell me about", "can i")):
            continue
            
        valid.append(clean)
            
    deg_name = primary_degree.get("degree", "your recommended program") if primary_degree else "your degree"
    deg_id = primary_degree.get("id", "") if primary_degree else ""
    stream_name = primary_degree.get("stream", "Science") if primary_degree else "Science"
    
    def get_fallback_card(slot: int) -> Dict[str, str]:
        if deg_id in ["ca_bcom", "bba_analytics"]:
            if slot == 1:
                return {
                    "quote": "Preference for Commercial Strategy & Enterprise",
                    "explanation": f"Consistently favored practical business operations, customer dynamics, and market problem-solving in {deg_name}."
                }
            else:
                return {
                    "quote": "Analytical Financial & Decision Mindset",
                    "explanation": f"Aligned with corporate governance, financial analysis, and organizational leadership required for {deg_name}."
                }
        elif deg_id == "btech_cs":
            if slot == 1:
                return {
                    "quote": "Computational Logic & Algorithmic Problem Solving",
                    "explanation": f"Reflects structured analytical reasoning, software design, and digital system architecture in {deg_name}."
                }
            else:
                return {
                    "quote": "Interest in Scalable Automation & Technology",
                    "explanation": f"Matches core B.Tech coursework in automating real-world workflows and client-side application design."
                }
        elif deg_id == "bdes_uiux":
            if slot == 1:
                return {
                    "quote": "User Empathy & Creative Prototyping",
                    "explanation": f"Directly reflects human-centered digital product design, visual wireframing, and user experience mapping in {deg_name}."
                }
            else:
                return {
                    "quote": "Aesthetic & Interactive System Thinking",
                    "explanation": f"Matches creative studio disciplines in visual communication, interface styling, and design research."
                }
        else:
            if slot == 1:
                return {
                    "quote": f"Strong Alignment with {deg_name}",
                    "explanation": f"Your psychometric profile demonstrates the dedicated competencies and mindset expected in {stream_name}."
                }
            else:
                return {
                    "quote": "Structured Multidisciplinary Problem Solving",
                    "explanation": f"Balances analytical rigor and practical execution required across coursework in {deg_name}."
                }
        
    def score_quote_for_degree(q_text: str) -> float:
        score = 1.0
        q_low = q_text.lower()
        # Dynamically boost score if student quote contains any core keywords of the recommended degree
        if primary_degree and "keywords" in primary_degree:
            for kw in primary_degree["keywords"]:
                if kw in q_low:
                    score += 3.0
        # Give higher weight to descriptive, detailed thoughts over ultra-short phrases
        score += min(2.5, len(q_text) / 35.0)
        return score

    def generate_degree_explanation(quote: str) -> str:
        q_low = quote.lower()
        # 1. Computer Science & Data
        if deg_id in ["btech_cs", "bsc_datascience"]:
            if any(k in q_low for k in ["softwares over hardwares", "prefer software", "code", "coding"]):
                return f"Directly confirms your decisive focus on Computer Science and software engineering."
            elif any(k in q_low for k in ["backend", "logic behind", "system", "algorithm"]):
                return f"Aligns with core coursework in distributed backend systems, algorithms, and computational logic."
            elif any(k in q_low for k in ["mobile app", "apps", "app works", "automate", "automation"]):
                return f"Reflects the core purpose of modern software systems: automating workflows and building scalable user apps."
            else:
                return f"Demonstrates the algorithmic curiosity and technical problem-solving instinct required in {deg_name}."

        # 2. Electronics, Mechanical, Aerospace & Robotics
        elif deg_id in ["btech_ece", "btech_mech", "btech_aerospace", "btech_robotics", "btech_civil", "btech_chemical"]:
            if any(k in q_low for k in ["aerospace", "aeronautical", "plane", "aircraft", "rocket", "space", "flight"]):
                return f"Directly reflects aerodynamic curiosity, flight mechanics, and aerospace propulsion systems in {deg_name}."
            elif any(k in q_low for k in ["circuit", "chip", "electronics", "sensor", "iot", "semiconductor"]):
                return f"Matches core laboratory coursework in microcontrollers, VLSI circuit architecture, and embedded hardware."
            elif any(k in q_low for k in ["robot", "robotics", "machine", "mechanical", "motor", "engine"]):
                return f"Demonstrates hands-on engineering instincts in kinematics, automated machinery, and physical mechatronics."
            else:
                return f"Directly aligns with hands-on physical engineering, prototyping, and technical design in {deg_name}."

        # 3. Architecture & Spatial Design
        elif deg_id == "barch_architecture":
            return f"Demonstrates spatial visualization, structural planning, and sustainable aesthetic design required in {deg_name}."

        # 4. Aviation, Defense & Marine
        elif deg_id in ["bsc_aviation", "defense_nda", "merchant_navy"]:
            return f"Reflects strong situational leadership, operational discipline, and technical navigation demanded by {deg_name}."

        # 5. Clinical Medicine, Healthcare & Dentistry
        elif deg_id in ["mbbs_healthcare", "bds_dental", "bsc_nursing", "ayush_medicine", "allied_health"]:
            return f"Directly confirms empathetic patient-centric care, biological diagnostic curiosity, and hospital healthcare commitment in {deg_name}."

        # 6. Biotech, Pharmacy, Physiotherapy & Veterinary
        elif deg_id in ["b_pharm", "bsc_lifesciences", "bpt_physiotherapy", "bvsc_veterinary"]:
            return f"Demonstrates empirical scientific curiosity, laboratory experimentation, and therapeutic healthcare focus in {deg_name}."

        # 7. Commerce, Accounting, FinTech & Actuarial
        elif deg_id in ["ca_bcom", "bcom_fintech", "actuarial_science", "cs_corporate_law", "cma_cost_accounting"]:
            if any(k in q_low for k in ["finance", "budget", "money", "audit", "account", "tax"]):
                return f"Matches rigorous corporate accounting, taxation governance, and statutory financial auditing in {deg_name}."
            elif any(k in q_low for k in ["stock market", "fintech", "invest", "trading", "risk"]):
                return f"Directly reflects quantitative financial modeling, market analytics, and investment evaluation in {deg_name}."
            else:
                return f"Demonstrates structured analytical discipline and commercial acumen required in {deg_name}."

        # 8. Business, Management & Hospitality
        elif deg_id in ["bba_analytics", "bhm_hospitality"]:
            if any(k in q_low for k in ["business", "startup", "venture", "market", "trade"]):
                return f"Directly reflects commercial strategy, venture creation, and entrepreneurial execution in {deg_name}."
            elif any(k in q_low for k in ["customer", "client", "user", "people"]):
                return f"Highlights customer-centric commercial problem solving, a vital competency for {deg_name}."
            elif any(k in q_low for k in ["lead", "manage", "team", "organize"]):
                return f"Demonstrates organizational leadership and team coordination expected in {deg_name}."
            else:
                return f"Demonstrates the strategic commercial mindset and client leadership needed in {deg_name}."

        # 9. Design, UI/UX, Fashion & Animation
        elif deg_id in ["bdes_uiux", "bdes_fashion", "bsc_animation_vfx"]:
            if any(k in q_low for k in ["user side", "user experience", "experience", "relate", "shoes", "customer", "people"]):
                return f"Directly reflects Human-Computer Interaction (HCI), user empathy mapping, and design thinking in {deg_name}."
            elif any(k in q_low for k in ["visual", "screen", "interface", "design", "fashion", "animation"]):
                return f"Matches studio coursework in digital wireframing, creative styling, and aesthetic visual communication."
            else:
                return f"Demonstrates the creative prototyping and aesthetic design thinking demanded by {deg_name}."

        # 10. Law, Media, Psychology & Economics
        elif deg_id in ["ballb_law", "bjmc_media", "ba_psychology", "ba_economics", "ba_public_admin"]:
            if any(k in q_low for k in ["law", "legal", "debate", "argument", "justice"]):
                return f"Reflects analytical jurisprudence, persuasive reasoning, and corporate advocacy central to {deg_name}."
            elif any(k in q_low for k in ["people", "mind", "behavior", "empathy", "relate"]):
                return f"Directly aligns with behavioral psychology, empathetic listening, and cognitive counseling in {deg_name}."
            else:
                return f"Demonstrates the critical inquiry, communicative clarity, and social impact required in {deg_name}."

        else:
            return f"Grounded in your specific answer, matching core academic expectations for {deg_name} ({stream_name})."

    selected_items = []
    if len(valid) >= 2:
        ranked_quotes = sorted(valid, key=score_quote_for_degree, reverse=True)
        q1 = ranked_quotes[0]
        q2 = None
        for cand in ranked_quotes[1:]:
            if cand.lower() != q1.lower() and cand.lower() not in q1.lower() and q1.lower() not in cand.lower():
                q2 = cand
                break
        if not q2 and len(ranked_quotes) > 1:
            q2 = ranked_quotes[1]

        selected_items.append({
            "quote": q1 if len(q1) <= 90 else q1[:87] + "...",
            "explanation": generate_degree_explanation(q1)
        })
        if q2:
            selected_items.append({
                "quote": q2 if len(q2) <= 90 else q2[:87] + "...",
                "explanation": generate_degree_explanation(q2)
            })
        else:
            selected_items.append(get_fallback_card(2))
    elif len(valid) == 1:
        q1 = valid[0]
        selected_items.append({
            "quote": q1 if len(q1) <= 90 else q1[:87] + "...",
            "explanation": generate_degree_explanation(q1)
        })
        selected_items.append(get_fallback_card(2))
    else:
        selected_items.append(get_fallback_card(1))
        selected_items.append(get_fallback_card(2))

    return selected_items
        
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

# ─────────────────────────────────────────────────────────────────────────────
# 6. Academic Score & Competitive Result Linguistic Parser
# ─────────────────────────────────────────────────────────────────────────────

def extract_academic_scores(text: str) -> Optional[Dict]:
    """
    Parses academic score indicators from student messages (e.g. 10th/12th percentages, JEE/NEET percentiles).
    Maps into empathetic tiers: 'high' (85%+), 'competitive' (70-85%), 'moderate' (55-70%), 'foundational' (<55%).
    """
    if not text:
        return None
        
    t_low = text.lower()
    
    # 1. Explicit percentage mentions (e.g., "got 68%", "scored 75 %", "62% in 10th", "88 percent")
    pct_match = re.search(r'(\b\d{1,2}(?:\.\d{1,2})?)\s*(?:%|percent)\b', t_low)
    if pct_match:
        val = float(pct_match.group(1))
        tier = "high" if val >= 85.0 else ("competitive" if val >= 70.0 else ("moderate" if val >= 55.0 else "foundational"))
        return {
            "raw_score": f"{val}%",
            "val": val,
            "type": "percentage",
            "tier": tier
        }
        
    # 2. Competitive exam percentiles (e.g., "jee 84 percentile", "cet 78", "neet 480")
    jee_match = re.search(r'\b(?:jee|cet|cuet)\s*(?:percentile|score|rank|mains?)?[\s:=]*(\d{1,2}(?:\.\d{1,2})?)\b', t_low)
    if jee_match:
        val = float(jee_match.group(1))
        tier = "high" if val >= 92.0 else ("competitive" if val >= 75.0 else ("moderate" if val >= 55.0 else "foundational"))
        return {
            "raw_score": f"{val} %ile",
            "val": val,
            "type": "competitive_percentile",
            "tier": tier
        }
        
    neet_match = re.search(r'\bneet\s*(?:score|marks|rank)?[\s:=]*(\d{2,3})\b', t_low)
    if neet_match:
        val = float(neet_match.group(1))
        tier = "high" if val >= 580 else ("competitive" if val >= 420 else ("moderate" if val >= 250 else "foundational"))
        return {
            "raw_score": f"{int(val)}/720 NEET",
            "val": val,
            "type": "neet_score",
            "tier": tier
        }
        
    # 3. Quick chip phrases
    if any(k in t_low for k in ["above 85", "85%+", "tier 1", "tier-1"]):
        return {"raw_score": "85%+", "val": 90.0, "type": "band", "tier": "high"}
    elif any(k in t_low for k in ["70% – 85%", "70-85", "competitive"]):
        return {"raw_score": "70% – 85%", "val": 78.0, "type": "band", "tier": "competitive"}
    elif any(k in t_low for k in ["55% – 70%", "50% – 70%", "55-70", "moderate"]):
        return {"raw_score": "55% – 70%", "val": 63.0, "type": "band", "tier": "moderate"}
    elif any(k in t_low for k in ["under 55", "under 50", "below 55"]):
        return {"raw_score": "<55%", "val": 50.0, "type": "band", "tier": "foundational"}
    elif any(k in t_low for k in ["prefer not to share", "passion first", "passion-first", "skip score"]):
        return {"raw_score": "Aptitude-First", "val": 0.0, "type": "skip", "tier": "passion_first"}
        
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 7. Conversational Signal & Information Theory Convergence Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def assess_conversational_signal(history: List[Dict[str, str]], riasec_scores: Dict[str, float]) -> Dict:
    """
    Evaluates whether the student has provided sufficient linguistic signal and
    psychometric differentiation to converge on a high-confidence recommendation,
    or whether the student remains in an ambiguous / vague / exploratory state.
    """
    user_msgs = [m["content"].strip() for m in history if m["role"] == "user"]
    total_turns = len(user_msgs)
    
    if total_turns == 0:
        return {
            "is_converged": False,
            "is_persistently_vague": False,
            "substantive_word_count": 0,
            "distinct_anchors": 0,
            "vague_count": 0,
            "total_turns": 0,
            "entropy": 1.0,
            "margin": 0.0,
            "confidence_level": "exploratory"
        }
        
    vague_patterns = [
        r'\b(?:idk|no\s*idea|dont\s*know|don\'t\s*know|not\s*sure|never\s*thought|nope|nah|nothing|whatever|fine|maybe|not\s*much|clueless|meh)\b',
        r'\b(?:neither|none|both|you\s*tell\s*me|tell\s*me|what\s*is\s*best|what\s*is\s*good|best\s*for\s*me|suggest\s*me|i\s*want\s*to\s*know\s*what\s*is\s*best)\b',
        r'^(?:ok|okay|cool|\?+)$'
    ]
    
    vague_turns = 0
    substantive_words = 0
    domain_hits = set()
    
    domain_anchors = {
        "tech": ["code", "coding", "software", "python", "java", "developer", "app", "apps", "ai", "algorithm", "data", "web", "computer", "tech", "tools"],
        "hardware": ["robot", "robotics", "machine", "mechanical", "electronics", "circuit", "engine", "iot", "hardware", "aerospace", "car"],
        "healthcare": ["doctor", "medicine", "biology", "hospital", "patient", "clinic", "biotech", "pharma", "cure", "health"],
        "business": ["business", "finance", "money", "startup", "stock", "market", "marketing", "trade", "accounting", "ca", "sales", "commerce"],
        "design": ["design", "ui", "ux", "art", "drawing", "sketch", "graphics", "visual", "creative", "media", "animation", "vfx", "story", "stories", "game", "gaming"],
        "social": ["psychology", "law", "lawyer", "advocate", "journalism", "news", "teaching", "public", "people", "counseling"]
    }
    
    for msg in user_msgs:
        m_lower = msg.lower().strip()
        # Direct choices like 'solo', 'team', 'stories', 'games', 'coding' are informative, not vague
        has_anchor = any(kw in m_lower for kw_list in domain_anchors.values() for kw in kw_list) or any(w in m_lower for w in ["solo", "team", "group", "lead", "story", "stories", "visual", "visuals", "fiction", "books"])
        is_evasive = any(re.search(pat, m_lower) for pat in vague_patterns)
        
        if is_evasive and not has_anchor:
            vague_turns += 1
            
        doc = nlp(msg)
        meaningful_tokens = [
            token.lemma_.lower() for token in doc 
            if not token.is_stop and not token.is_punct and len(token.text) > 2 and not token.like_num
        ]
        substantive_words += len(meaningful_tokens)
        
        for domain, kw_list in domain_anchors.items():
            if any(kw in m_lower for kw in kw_list):
                domain_hits.add(domain)

    # Compute Normalized Shannon Entropy
    total_score = sum(riasec_scores.values()) or 1.0
    p = [riasec_scores.get(c, 0.0) / total_score for c in ["R", "I", "A", "S", "E", "C"]]
    h = 0.0
    for prob in p:
        if prob > 1e-6:
            h -= prob * math.log2(prob)
    max_h = math.log2(6.0) # ~2.585
    h_norm = float(round(h / max_h, 3))
    
    # Margin check between top 2 RIASEC traits
    sorted_vals = sorted(riasec_scores.values(), reverse=True)
    margin = float(round(sorted_vals[0] - sorted_vals[1], 1)) if len(sorted_vals) >= 2 else 0.0
    top_score = sorted_vals[0] if sorted_vals else 16.7
    
    # Robust Convergence Criteria:
    # Requires genuine domain interest signals, not just turn counting!
    vague_ratio = vague_turns / total_turns
    has_genuine_domain_signal = len(domain_hits) >= 1 or substantive_words >= 10
    
    is_converged = (
        total_turns >= 3 and
        has_genuine_domain_signal and
        vague_ratio < 0.40 and
        (margin >= 3.5 or top_score >= 21.0 or h_norm <= 0.85)
    )
    
    is_persistently_vague = (
        total_turns >= 3 and
        (vague_ratio >= 0.50 or len(domain_hits) == 0)
    )
    
    confidence_level = "high" if is_converged else ("moderate" if (total_turns >= 3 and not is_persistently_vague) else "exploratory")
    
    return {
        "is_converged": is_converged,
        "is_persistently_vague": is_persistently_vague,
        "substantive_word_count": substantive_words,
        "distinct_anchors": len(domain_hits),
        "vague_count": vague_turns,
        "total_turns": total_turns,
        "entropy": h_norm,
        "margin": margin,
        "confidence_level": confidence_level
    }

