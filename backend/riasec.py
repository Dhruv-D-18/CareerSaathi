"""
Holland's RIASEC Psychometric Engine & Indian Educational Ontology Matrix
CSE708 NLP Lab Practical — Syllabus Sessions 26–27 (BERT, Semantic Embeddings, Vector Spaces)
"""

import numpy as np
import math
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────────────────────────────────────
# 1. Official Holland RIASEC Codes & Behavioral Anchor Corpora
# ─────────────────────────────────────────────────────────────────────────────
RIASEC_CODES = {
    "R": {
        "name": "Realistic",
        "title": "Practical & Hands-On (Doers)",
        "emoji": "🛠️",
        "color": "#fb923c",  # Warm Orange
        "anchors": [
            "I enjoy building physical things, working with tools, machinery, vehicle engines, hardware, and assembling prototypes.",
            "Tinkering with physical devices, robotics mechanisms, metal chassis, physical electronics, and practical construction.",
            "Hands-on mechanical repair, physical assembly, woodworking, electrical wiring, vehicle mechanics, and structural modeling.",
            "Operating precision machines, 3D printing physical parts, soldering hardware circuits, and structural fabrication."
        ]
    },
    "I": {
        "name": "Investigative",
        "title": "Analytical & Scientific (Thinkers)",
        "emoji": "🔬",
        "color": "#38bdf8",  # Cyan Blue
        "anchors": [
            "I love coding software, writing algorithms, programming applications, web backend, and analyzing datasets.",
            "Computer programming, solving mathematical logic problems, developing software systems, and data science.",
            "Scientific curiosity, investigating physics laws, chemical reactions, laboratory biological research, and mathematical proofs.",
            "Coding in Python, building web applications, database schemas, computational models, and diagnosing complex logic bugs."
        ]
    },
    "A": {
        "name": "Artistic",
        "title": "Creative & Expressive (Creators)",
        "emoji": "🎨",
        "color": "#c084fc",  # Soft Purple
        "anchors": [
            "I love creative writing, storytelling, poetry, literature, sketching, UI UX visual design, and original artistic expression.",
            "Visual aesthetics, graphic design, illustration, digital arts, media branding, creative writing, and architectural design.",
            "Original design thinking, fashion, creative photography, multimedia content, theater, and artistic storytelling.",
            "Expressing personal imagination, typography, visual user interfaces, creative essays, and aesthetic design."
        ]
    },
    "S": {
        "name": "Social",
        "title": "Empathetic & Helping (Helpers)",
        "emoji": "🤝",
        "color": "#34d399",  # Emerald Green
        "anchors": [
            "I enjoy helping people, teaching, patient healthcare, counseling others, community welfare, and empathetic mentoring.",
            "Studying human psychology, mental health therapy, doctor consultations, nursing, and clinical healthcare care.",
            "Collaborating with people, understanding emotions, voluntary social work, community development, and guiding youth."
        ]
    },
    "E": {
        "name": "Enterprising",
        "title": "Persuasive & Leadership (Persuaders)",
        "emoji": "📢",
        "color": "#f59e0b",  # Amber Gold
        "anchors": [
            "I love business leadership, launching commercial startups, pitching ideas to investors, sales marketing, and negotiation.",
            "Corporate strategy, executive management, commercial operations, marketing campaigns, public speaking, and business ventures.",
            "Negotiating deals, leading business teams, closing client sales, commercial contracts, and entrepreneurial leadership."
        ]
    },
    "C": {
        "name": "Conventional",
        "title": "Organized & Structured (Organizers)",
        "emoji": "📊",
        "color": "#f43f5e",  # Rose Coral
        "anchors": [
            "I enjoy structured data records, accounting balance sheets, tax audits, banking compliance, and financial records.",
            "Organizing systematic administrative filing, statutory compliance, financial spreadsheets, and orderly bookkeeping.",
            "Methodical numerical calculations, financial auditing, standard operating procedures, and statistical documentation."
        ]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dense Semantic Embedding Layer (Sentence-BERT all-MiniLM-L6-v2)
# ─────────────────────────────────────────────────────────────────────────────
print("  [RIASEC Engine] Initializing Sentence-BERT (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Compute class prototype centroids \mu_k for each of the 6 RIASEC codes
RIASEC_CENTROIDS = {}
for code, meta in RIASEC_CODES.items():
    anchors = meta["anchors"]
    # Encode anchor sentences into R^384
    vecs = embedder.encode(anchors, convert_to_tensor=True)
    # Average across anchors to get the unit centroid prototype
    centroid = vecs.mean(dim=0)
    # L2 normalize
    centroid = centroid / centroid.norm()
    RIASEC_CENTROIDS[code] = centroid

print("  [RIASEC Engine] [OK] 6 Holland RIASEC Centroid Prototypes Pre-computed in R^384!")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Psychometric Inference & Bayesian Belief Updating
# ─────────────────────────────────────────────────────────────────────────────
def compute_riasec_scores(user_text: str, temperature: float = 10.0) -> Dict[str, float]:
    """
    Encodes user text into dense vector space, calculates Cosine Similarities against
    the 6 Holland centroids, and scales with Temperature Softmax.
    """
    if not user_text or not user_text.strip():
        # Flat uniform prior: 16.67% each
        return {code: 16.67 for code in RIASEC_CODES.keys()}
    
    # 1. Encode user text
    user_emb = embedder.encode(user_text, convert_to_tensor=True)
    user_emb = user_emb / user_emb.norm()
    
    # 2. Compute Cosine Similarities
    raw_sims = {}
    for code, centroid in RIASEC_CENTROIDS.items():
        sim = float(util.cos_sim(user_emb, centroid).item())
        raw_sims[code] = sim
        
    # 3. Temperature-Scaled Softmax
    codes = list(RIASEC_CODES.keys())
    scores_array = np.array([raw_sims[c] for c in codes], dtype=np.float32)
    exp_scores = np.exp(scores_array * temperature)
    probs = exp_scores / np.sum(exp_scores)
    
    result = {}
    for i, code in enumerate(codes):
        result[code] = float(round(probs[i] * 100.0, 1))
        
    return result

def update_belief_state(previous_state: Dict[str, float], turn_scores: Dict[str, float], alpha: float = 0.55) -> Dict[str, float]:
    """
    Momentum Bayesian Belief Update: S_t = alpha * S_{t-1} + (1 - alpha) * a_t
    Prevents single-utterance instability while smoothly accumulating user profile.
    """
    if not previous_state:
        return turn_scores
        
    updated = {}
    total = 0.0
    for code in RIASEC_CODES.keys():
        prev_val = previous_state.get(code, 16.67)
        curr_val = turn_scores.get(code, 16.67)
        val = alpha * prev_val + (1.0 - alpha) * curr_val
        updated[code] = val
        total += val
        
    # Re-normalize to sum to 100%
    normalized = {}
    for code in RIASEC_CODES.keys():
        normalized[code] = float(round((updated[code] / total) * 100.0, 1))
        
    return normalized

def calculate_shannon_entropy(scores: Dict[str, float]) -> Tuple[float, bool]:
    """
    Computes Normalized Shannon Entropy:
    H_norm = - (1 / log2 6) * sum(p_k * log2 p_k)
    Returns: (entropy_val, is_ambiguous)
    Where is_ambiguous is True if H_norm > 0.85 (triggers PROBING mode).
    """
    total = sum(scores.values())
    if total <= 0:
        return 1.0, True
        
    p = [scores[c] / total for c in RIASEC_CODES.keys()]
    
    # Calculate Shannon Entropy
    h = 0.0
    for prob in p:
        if prob > 1e-6:
            h -= prob * math.log2(prob)
            
    max_h = math.log2(6.0) # log2(6) ≈ 2.585
    h_norm = float(round(h / max_h, 3))
    
    # If entropy is high (> 0.88) or top two scores are within 3%, distribution is ambiguous
    sorted_vals = sorted(scores.values(), reverse=True)
    margin = sorted_vals[0] - sorted_vals[1]
    
    is_ambiguous = (h_norm >= 0.86) or (margin < 4.0)
    return h_norm, is_ambiguous

# ─────────────────────────────────────────────────────────────────────────────
# 4. Indian Educational Mapping Matrix & Knowledge Ontology
# ─────────────────────────────────────────────────────────────────────────────

# Weights matrix W_Stream in R^(4 x 6) mapping RIASEC (R, I, A, S, E, C) to Indian 10th streams
STREAM_WEIGHTS = {
    "Science (PCM)": {
        "R": 0.40, "I": 0.45, "A": 0.05, "S": 0.02, "E": 0.05, "C": 0.03,
        "description": "Physics, Chemistry & Mathematics — Pathway to Engineering, Tech, Architecture & Pure Sciences."
    },
    "Science (PCB)": {
        "R": 0.10, "I": 0.50, "A": 0.05, "S": 0.30, "E": 0.02, "C": 0.03,
        "description": "Physics, Chemistry & Biology — Pathway to Medicine, Biotech, Healthcare, Psychology & Life Sciences."
    },
    "Commerce & Finance": {
        "R": 0.03, "I": 0.15, "A": 0.02, "S": 0.10, "E": 0.45, "C": 0.25,
        "description": "Commerce, Accounts & Economics — Pathway to CA, B.Com, Investment Banking, FinTech & Corporate Business."
    },
    "Liberal Studies & Arts": {
        "R": 0.05, "I": 0.20, "A": 0.45, "S": 0.20, "E": 0.07, "C": 0.03,
        "description": "Humanities, Design, Psychology & Literature — Pathway to Law, Journalism, Civil Services, Design & Economics."
    }
}

# Specialization Degrees mapped to RIASEC poles and Indian Stream
DEGREE_TAXONOMY = [
    {
        "id": "btech_cs",
        "degree": "B.Tech in Computer Science & Artificial Intelligence",
        "stream": "Science (PCM)",
        "riasec_primary": "I",
        "riasec_secondary": "R",
        "exams": ["JEE Mains", "JEE Advanced", "GUJCET", "NUV-DET"],
        "profile": "Algorithmic thinking, cloud architecture, machine learning systems, and software engineering.",
        "keywords": ["code", "coding", "software", "algorithm", "computer", "programming", "python", "tech", "developer", "ai"]
    },
    {
        "id": "btech_mech_robotics",
        "degree": "B.Tech in Robotics & Mechanical Automation",
        "stream": "Science (PCM)",
        "riasec_primary": "R",
        "riasec_secondary": "I",
        "exams": ["JEE Mains", "GUJCET", "State CET"],
        "profile": "Kinematics, automotive dynamics, mechatronics, CAD manufacturing, and industrial assembly.",
        "keywords": ["hardware", "robot", "robotics", "machine", "mechanical", "assemble", "tools", "engine", "physical"]
    },
    {
        "id": "bsc_datascience",
        "degree": "B.Sc (Hons) in Applied Mathematics & Data Science",
        "stream": "Science (PCM)",
        "riasec_primary": "I",
        "riasec_secondary": "C",
        "exams": ["CUET UG", "IISER IAT", "NEST"],
        "profile": "Theoretical scientific investigation, statistical modeling, probability theory, and data analytics.",
        "keywords": ["math", "mathematics", "data", "statistics", "analysis", "probability", "logic", "numbers"]
    },
    {
        "id": "mbbs_healthcare",
        "degree": "MBBS / Clinical Healthcare & Medicine",
        "stream": "Science (PCB)",
        "riasec_primary": "I",
        "riasec_secondary": "S",
        "exams": ["NEET UG"],
        "profile": "Human physiological anatomy, cellular immunology, diagnostic therapy, and patient healing.",
        "keywords": ["doctor", "medicine", "patient", "hospital", "clinic", "biology", "health", "cure", "treatment"]
    },
    {
        "id": "bsc_lifesciences",
        "degree": "B.Sc in Life Sciences & Biotechnology",
        "stream": "Science (PCB)",
        "riasec_primary": "I",
        "riasec_secondary": "R",
        "exams": ["CUET UG", "NEET UG", "GAT-B"],
        "profile": "Genetic synthesis, bio-informatics, drug delivery nanotech, and pharmaceutical laboratory research.",
        "keywords": ["biology", "lab", "genetics", "biotech", "cells", "experiment", "microbiology", "nature"]
    },
    {
        "id": "ca_bcom",
        "degree": "Chartered Accountancy (CA) & B.Com (Professional)",
        "stream": "Commerce & Finance",
        "riasec_primary": "C",
        "riasec_secondary": "E",
        "exams": ["ICAI CA Foundation", "CUET UG"],
        "profile": "Statutory auditing, taxation governance, financial reporting balance sheets, and corporate accounting.",
        "keywords": ["accounting", "finance", "audit", "tax", "ca", "spreadsheet", "balance sheet", "money", "banking"]
    },
    {
        "id": "bba_analytics",
        "degree": "BBA in Business Analytics & Entrepreneurship",
        "stream": "Commerce & Finance",
        "riasec_primary": "E",
        "riasec_secondary": "I",
        "exams": ["IPMAT", "SET", "CUET UG", "NUV-DET"],
        "profile": "Startup leadership, corporate valuation, commercial operations, marketing strategy, and investment pitching.",
        "keywords": ["business", "startup", "management", "pitch", "leadership", "marketing", "sales", "strategy", "enterprise"]
    },
    {
        "id": "bdes_uiux",
        "degree": "B.Des in Product Design & Digital UI/UX",
        "stream": "Liberal Studies & Arts",
        "riasec_primary": "A",
        "riasec_secondary": "R",
        "exams": ["UCEED", "NID DAT", "NIFT"],
        "profile": "User interface design, digital product styling, design thinking, visual aesthetics, and creative prototyping.",
        "keywords": ["design", "ui", "ux", "visual", "art", "creative", "sketch", "graphics", "aesthetic", "prototype"]
    },
    {
        "id": "ba_psychology",
        "degree": "B.A. (Hons) in Psychology & Behavioral Science",
        "stream": "Liberal Studies & Arts",
        "riasec_primary": "S",
        "riasec_secondary": "I",
        "exams": ["CUET UG"],
        "profile": "Behavioral psychology, human cognition, counseling therapy, social dynamics, and neuro-psychology.",
        "keywords": ["psychology", "mind", "behavior", "counseling", "people", "mental health", "empathy", "listening"]
    },
    {
        "id": "ba_economics",
        "degree": "B.A. (Hons) in Economics & Public Policy",
        "stream": "Liberal Studies & Arts",
        "riasec_primary": "I",
        "riasec_secondary": "E",
        "exams": ["CUET UG"],
        "profile": "Macroeconomic econometrics, government policy analysis, financial market trends, and developmental sociology.",
        "keywords": ["economics", "policy", "market", "society", "inflation", "governance", "trade"]
    },
    {
        "id": "ballb_law",
        "degree": "B.A. LL.B (Hons) in Corporate Law & Advocacy",
        "stream": "Liberal Studies & Arts",
        "riasec_primary": "E",
        "riasec_secondary": "A",
        "exams": ["CLAT (NLUs)", "AILET"],
        "profile": "Legal argumentation, constitutional jurisprudence, corporate intellectual property, and persuasive court advocacy.",
        "keywords": ["law", "advocate", "court", "legal", "justice", "argument", "debate", "rights"]
    }
]

def map_riasec_to_indian_streams(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Computes percentage affinity fit for the 4 Indian 10th/12th academic streams
    using our expert knowledge matrix W_Stream * S.
    """
    stream_scores = {}
    for stream, meta in STREAM_WEIGHTS.items():
        score = sum(meta[code] * scores.get(code, 0.0) for code in RIASEC_CODES.keys())
        stream_scores[stream] = score
        
    # Normalize to 100%
    tot = sum(stream_scores.values()) or 1.0
    normalized = {}
    for stream, sc in stream_scores.items():
        normalized[stream] = float(round((sc / tot) * 100.0, 1))
        
    return dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True))

def apply_negative_feedback(current_scores: Dict[str, float], rejection_text: str, rejected_degree: str = "") -> Tuple[Dict[str, float], List[str]]:
    """
    Active Learning Constraint Update (Negative Evidence Incorporation):
    When a student dislikes a suggested career or discipline, downweight the
    corresponding RIASEC dimensions and compile a blacklist of excluded career IDs.
    """
    updated = dict(current_scores)
    text_lower = (rejection_text + " " + rejected_degree).lower()
    excluded_ids = []
    
    # 1. Negative rules for Commerce / Finance / Accounting
    if any(k in text_lower for k in ["ca", "account", "commerce", "finance", "audit", "tax", "spreadsheet", "balance sheet"]):
        updated["C"] = max(2.0, updated.get("C", 16.7) * 0.3)
        updated["E"] = max(4.0, updated.get("E", 16.7) * 0.6)
        excluded_ids.extend(["ca_bcom", "bba_analytics"])
        
    # 2. Negative rules for Biology / Medical / Healthcare
    if any(k in text_lower for k in ["bio", "biology", "medical", "doctor", "hospital", "patient", "clinic", "life science"]):
        updated["S"] = max(4.0, updated.get("S", 16.7) * 0.6)
        # Only dampen bio aspects of I, don't kill I completely if they like tech
        excluded_ids.extend(["mbbs_healthcare", "bsc_lifesciences"])
        
    # 3. Negative rules for Coding / Software / Engineering
    if any(k in text_lower for k in ["coding", "code", "software", "programming", "screen all day", "sit at desk"]):
        updated["I"] = max(4.0, updated.get("I", 16.7) * 0.4)
        excluded_ids.extend(["btech_cs", "bsc_datascience"])
        
    # 4. Negative rules for Heavy Hardware / Mechanical
    if any(k in text_lower for k in ["hardware", "mechanic", "tools", "machine", "grease", "factory"]):
        updated["R"] = max(4.0, updated.get("R", 16.7) * 0.3)
        excluded_ids.extend(["btech_mech_robotics"])
        
    # 5. Negative rules for Pure Arts / Writing
    if any(k in text_lower for k in ["art", "sketch", "drawing", "writing", "literature", "history"]):
        updated["A"] = max(3.0, updated.get("A", 16.7) * 0.3)
        excluded_ids.extend(["bdes_uiux", "ba_psychology", "ba_economics", "ballb_law"])

    # Re-normalize to 100%
    total = sum(updated.values()) or 1.0
    normalized = {k: float(round((v / total) * 100.0, 1)) for k, v in updated.items()}
    return normalized, list(set(excluded_ids))


def get_career_recommendations(scores: Dict[str, float], stream_filter: str = None, excluded_ids: List[str] = None, is_10th: bool = False, user_text: str = "") -> Dict:
    """
    Ranks Indian degrees based on RIASEC Euclidean proximity and returns
    Primary Match, Secondary Alternative, Stream affinity, and Entrance Exam roadmap.
    Supports negative exclusion filtering, explicit keyword boosting, and 10th vs 12th differentiation.
    """
    excluded_ids = excluded_ids or []
    user_text_lower = user_text.lower() if user_text else ""
    
    # Sort RIASEC traits
    sorted_riasec = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_code = sorted_riasec[0][0]
    second_code = sorted_riasec[1][0]
    
    stream_affinities = map_riasec_to_indian_streams(scores)
    top_stream = list(stream_affinities.keys())[0]
    
    # Calculate fit for each degree
    ranked_degrees = []
    for deg in DEGREE_TAXONOMY:
        deg_id = deg.get("id", deg["degree"].lower())
        if deg_id in excluded_ids:
            continue
            
        # Degree alignment score = primary match weight * 0.65 + secondary match weight * 0.35
        p_weight = scores.get(deg["riasec_primary"], 0.0)
        s_weight = scores.get(deg["riasec_secondary"], 0.0)
        fit = (p_weight * 0.65) + (s_weight * 0.35)
        
        # Boost if stream matches student's 12th stream
        if stream_filter and stream_filter.lower() in deg["stream"].lower():
            fit *= 1.15

        # Surface Keyword Attribution Boost:
        # If student explicitly mentioned mobile apps, software, code, algorithms -> strong domain affinity
        keywords = deg.get("keywords", [])
        matched_kws = [kw for kw in keywords if kw in user_text_lower]
        if matched_kws:
            kw_multiplier = min(1.35, 1.0 + (len(matched_kws) * 0.07))
            fit *= kw_multiplier
            
        ranked_degrees.append({
            "id": deg_id,
            "degree": deg["degree"],
            "stream": deg["stream"],
            "fit_score": float(round(fit, 1)),
            "exams": deg["exams"],
            "profile": deg["profile"],
            "primary_code": deg["riasec_primary"],
            "secondary_code": deg["riasec_secondary"]
        })
        
    ranked_degrees.sort(key=lambda x: x["fit_score"], reverse=True)
    
    # Fallback if all excluded
    if not ranked_degrees:
        ranked_degrees = [{
            "id": "general_bdes",
            "degree": "B.Des in Product Design & Digital UI/UX",
            "stream": "Liberal Studies & Arts",
            "fit_score": 75.0,
            "exams": ["UCEED", "NID DAT"],
            "profile": "Multidisciplinary creative problem solving.",
            "primary_code": "A",
            "secondary_code": "R"
        }]
        
    primary = ranked_degrees[0]
    secondary = ranked_degrees[1] if len(ranked_degrees) > 1 else ranked_degrees[0]
    
    # If student is 10th standard, adapt primary presentation to highlight STREAM first
    if is_10th:
        # Ensure primary stream matches the top degree's required stream
        recommended_title = f"{primary['stream']} Stream (Pathway to {primary['degree']})"
    else:
        recommended_title = primary["degree"]
        
    # Also worth a look list (excluding top 1)
    alternatives = [d["degree"] for d in ranked_degrees[1:4]]
    
    return {
        "primary": primary,
        "secondary": secondary,
        "recommended_title": recommended_title,
        "top_stream": top_stream,
        "stream_affinities": stream_affinities,
        "alternatives": alternatives,
        "is_10th": is_10th,
        "top_riasec": [
            {"code": top_code, "name": RIASEC_CODES[top_code]["name"], "score": scores[top_code]},
            {"code": second_code, "name": RIASEC_CODES[second_code]["name"], "score": scores[second_code]}
        ],
        "all_ranked": ranked_degrees[:5]
    }
