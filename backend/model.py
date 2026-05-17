import os
# Mute Hugging Face hub symlinks caching warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import re

# Try to import sentence-transformers for Option A (Sentence-BERT Embeddings)
try:
    from sentence_transformers import SentenceTransformer, util
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# 8 Target Classes
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = [
    "Computer Science",
    "Mechanical Engineering",
    "Electrical Engineering",
    "Civil Engineering",
    "Biology",
    "Chemistry",
    "Physics",
    "Management"
]

# ─────────────────────────────────────────────────────────────────────────────
# High-Quality Synthetic Dataset (Representative student interest statements)
# ─────────────────────────────────────────────────────────────────────────────
DATASET = {
    "Computer Science": [
        "I love coding and building apps in python or javascript.",
        "Writing software and designing complex algorithms is my passion.",
        "I want to build web pages, databases, and secure servers.",
        "Developing software, hacking systems, and coding in java.",
        "Setting up database schemas and building cloud applications.",
        "I enjoy programming puzzles and automating daily tasks with scripts.",
        "Artificial intelligence, neural networks, and machine learning excite me.",
        "Building a mobile app or a computer game from scratch.",
        "Understanding how operating systems, cybersecurity, and computer networks work.",
        "I like designing clean UI interfaces and writing robust backend systems.",
        "Solving logical coding challenges and debugging code issues.",
        "I want to write software for servers, create websites, and manage data.",
        "Coding, automation, software testing, and fullstack engineering.",
        "Developing games, configuring local servers, and writing scripting files.",
        "Using python, visual studio, git, and deploying code to cloud databases."
    ],
    "Mechanical Engineering": [
        "I love designing engines, mechanical systems, and cars.",
        "Tinkering with machines, gears, and robotic joints is fun.",
        "I want to design CAD models of aerodynamic vehicles and wings.",
        "Understanding thermodynamics, manufacturing engines, and mechanical CAD structures.",
        "Building physical machines, mechanical gearboxes, and automotive engines.",
        "I like building physical drone frames, landing gear, and mechanical wings.",
        "Working in a workshop, designing tools, and assembling mechanical parts.",
        "Analyzing fluid mechanics, engine efficiency, and thermodynamic heat cycles.",
        "Designing cars, motorcycles, industrial turbines, and gear mechanisms.",
        "I love mechanics, manufacturing, mechanical design, and assembly lines.",
        "Tinkering with hardware, motors, robotic arms, and mechanical fixtures.",
        "Drawing 3D CAD prints and stress-testing mechanical parts.",
        "Assembling motors, vehicle chassis, wheel axles, and transmission systems."
    ],
    "Electrical Engineering": [
        "I enjoy wiring circuits, soldering breadboards, and electronic microchips.",
        "Designing electrical circuit boards, transistors, and wiring battery systems.",
        "Working with microcontrollers, Arduino, Raspberry Pi, and circuit logic.",
        "I like sensors, electronic components, resistors, and capacitors.",
        "Configuring voltage, currents, electricity wiring, and solar panels.",
        "Designing battery management systems, logic gates, and hardware controllers.",
        "I enjoy connecting wires, soldering LEDs, and debugging hardware circuits.",
        "Understanding electromagnetism, power grids, transformers, and electronic systems.",
        "Assembling circuit boards, wiring battery cells, and measuring voltage currents.",
        "Working with logic wiring, power electronics, sensors, and electrical components.",
        "Tinkering with microchips, soldering wires, and testing electrical signals.",
        "Installing solar panel arrays, smart home wiring, and LED driver systems."
    ],
    "Civil Engineering": [
        "I love structural architecture, drawing blueprints, and building bridges.",
        "Designing skyscrapers, concrete foundations, and surveying road designs.",
        "Working on building sites, analyzing concrete, brick strength, and dams.",
        "I want to construct highways, bridges, tunnels, and massive architectural frames.",
        "Drawing blueprints of houses, structural engineering, and soil foundations.",
        "Calculating structural loads, concrete reinforcements, and highway systems.",
        "Building smart cities, bridges, dams, canals, and structural layouts.",
        "I like brickwork, structural design, construction materials, and civil design.",
        "Analyzing architectural maps, concrete curing, and foundation load limits.",
        "Designing water dams, city sewage networks, and highway flyovers.",
        "Drafting structural blueprints and planning building layouts."
    ],
    "Biology": [
        "I love looking through a microscope to study cells and tissues.",
        "Understanding human anatomy, genes, DNA, and molecular biology.",
        "Studying diseases, vaccines, immunology, and curing infections.",
        "Working in a hospital, assisting doctors, and studying cellular genetics.",
        "I am fascinated by botany, plants, zoology, wildlife, and natural ecosystems.",
        "Analyzing cell structures, DNA sequencing, genetics, and biotechnology.",
        "Learning about human health, pharmaceutical drugs, and biology lab experiments.",
        "I want to be a biologist, study medicine, or research genetic cloning.",
        "Ecosystems, marine biology, bacterial micro-organisms, and animal health.",
        "Studying plant biology, genetic mutations, and cell mitosis.",
        "Analyzing blood samples, looking at immune cells, and medical microbiology."
    ],
    "Chemistry": [
        "I enjoy mixing chemicals in a lab, testing pH levels and reactions.",
        "Analyzing organic compounds, molecules, and chemical solutions.",
        "Fascinated by the periodic table, catalysts, and molecular synthesis.",
        "Testing acids, bases, chemical solvents, and polymer chemistry.",
        "Doing titrations, chemical assays, and synthesizing molecules in test tubes.",
        "Understanding metallurgy, chemical reactions, and compound structures.",
        "Working in a chemistry lab, analyzing chemical formulas and molecular bonds.",
        "I like studying chemical properties, polymers, petrochemicals, and formulas.",
        "Measuring chemical concentrations, solution reactions, and solvent safety.",
        "Synthesizing compounds, plastic crystallization, and biochemistry experiments.",
        "Mixing catalysts, watching solutions change colors, and chemistry reactions."
    ],
    "Physics": [
        "I love studying space, black holes, astrophysics, and gravity.",
        "Understanding mechanics, light wave refraction, lenses, and laser optics.",
        "Fascinated by quantum physics, nuclear energy, and thermodynamics.",
        "Analyzing forces, friction, velocity, motion, and relativity theories.",
        "I like optics, laser beams, magnetic waves, and electromagnetic physics.",
        "Understanding sound propagation, heat transfers, and kinetic energy.",
        "Running physics experiments with pendulums, gravity meters, and lenses.",
        "Studying planetary orbits, astronomical objects, and quantum particles.",
        "Analyzing wave optics, speed of light, nuclear fusion, and space physics.",
        "Fascinated by mechanical forces, gravity, quantum mechanics, and cosmology.",
        "Measuring speed, force acceleration, light refraction, and lens focal lengths."
    ],
    "Management": [
        "I love business, marketing campaigns, and launching startups.",
        "Managing budgets, writing balance sheets, and analyzing financial data.",
        "Leading teams, negotiating prices, and pitching ideas to investors.",
        "I want to run a company, manage human resources, and track sales metrics.",
        "Understanding corporate finance, stocks, market research, and products.",
        "I enjoy sales pitches, branding strategy, recruiting employees, and operations.",
        "Managing canteens, calculating profit margins, and supply chain routing.",
        "I want to do BBA or MBA, build a business empire, and do marketing.",
        "Negotiating supplier deals, planning budgets, and managing customer relations.",
        "Analyzing target markets, pitching startup models, and HR coordinating.",
        "Managing retail stores, stock inventory, business strategies, and leadership."
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Traditional ML Pipeline Training (Used for Explainable AI Dashboard splits & tags)
# ─────────────────────────────────────────────────────────────────────────────
X_train = []
y_train = []

for category, sentences in DATASET.items():
    for sentence in sentences:
        X_train.append(sentence.lower())
        y_train.append(category)
        
    # Generate synthetic mixtures to robustify the vectorizer
    for i in range(len(sentences) - 1):
        mixed = sentences[i] + " Also, " + sentences[i+1]
        X_train.append(mixed.lower())
        y_train.append(category)

vectorizer = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

# Random Forest for smooth, traditional explanations
rf_classifier = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf_classifier.fit(X_train_vec, y_train)

# Decision Tree for clean visual explanations (XAI splits)
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train_vec, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# Option A: Pre-trained Sentence Transformers Initialisation (Semantic Layer)
# ─────────────────────────────────────────────────────────────────────────────
embedding_model = None
CLASS_EMBEDDINGS = {}
ST_ENABLED = False

if ST_AVAILABLE:
    try:
        print("  [ML Pipeline] [Sentence-Transformers] Loading all-MiniLM-L6-v2 semantic model...")
        # Loads pre-trained model (downloads on first run, ~90MB)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("  [ML Pipeline] [Sentence-Transformers] Pre-computing semantic class prototype embeddings...")
        for category, sentences in DATASET.items():
            # Calculate dense embedding vectors for reference sentences in each category
            vecs = embedding_model.encode(sentences, convert_to_tensor=True)
            # Calculate the average vector (centroid profile) for the category
            CLASS_EMBEDDINGS[category] = vecs.mean(dim=0)
            
        ST_ENABLED = True
        print("  [ML Pipeline] [Sentence-Transformers] [OK] Semantic class prototype embeddings ready!")
    except Exception as e:
        print(f"  [ML Pipeline] [Sentence-Transformers] [ERROR] Failed to load model: {e}. Falling back to Random Forest.")
        ST_ENABLED = False
else:
    print("  [ML Pipeline] [Sentence-Transformers] Package not yet ready or installed. Running in classical fallback mode.")

# ─────────────────────────────────────────────────────────────────────────────
# Model Inference Helpers
# ─────────────────────────────────────────────────────────────────────────────

def predict_interest(text: str) -> dict:
    """
    Returns prediction probabilities for all 8 categories.
    Uses pre-trained Sentence-Transformers if available, otherwise falls back to Random Forest.
    """
    if not text.strip():
        # Default flat probability if empty
        return {cat: 12.5 for cat in CLASSES}
    
    # ── Option A: Semantic BERT Classifier ──
    if ST_ENABLED and embedding_model is not None:
        try:
            # 1. Encode user text into dense vector space
            user_emb = embedding_model.encode(text, convert_to_tensor=True)
            
            # 2. Calculate Cosine Similarity with all 8 class prototype vectors
            similarities = {}
            for category, proto_emb in CLASS_EMBEDDINGS.items():
                sim = util.cos_sim(user_emb, proto_emb).item()
                similarities[category] = float(sim)
                
            # 3. Apply Softmax with temperature scaling to expand probability contrast
            raw_scores = np.array([similarities[cat] for cat in CLASSES])
            
            # Temperature=12.0 provides beautiful, realistic probability distributions
            temp = 12.0
            exp_scores = np.exp(raw_scores * temp)
            probs = exp_scores / np.sum(exp_scores)
            
            # Map classes to their probabilities (scaled to %)
            result = {}
            for i, cat in enumerate(CLASSES):
                result[str(cat)] = float(round(probs[i] * 100, 1))
                
            # Sort descending
            return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        except Exception as e:
            print(f"  [Semantic Prediction Error] {e}. Falling back to Random Forest.")
            
    # ── Fallback: Classical Random Forest TF-IDF ──
    vec = vectorizer.transform([text.lower()])
    probs = rf_classifier.predict_proba(vec)[0]
    
    # Map classes to their probabilities (scaled to %)
    result = {}
    for i, cls in enumerate(rf_classifier.classes_):
        result[str(cls)] = float(round(probs[i] * 100, 1))
        
    # Sort descending
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))

def get_highlighted_words(text: str) -> list:
    """
    Identifies which words in the user's text are active TF-IDF features and
    maps them to the categories they represent. Used in Explainable AI Cloud.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    feature_names = vectorizer.get_feature_names_out()
    feature_set = set(feature_names)
    
    highlights = []
    for word in words:
        if word in feature_set:
            word_vec = vectorizer.transform([word])
            class_idx = rf_classifier.predict(word_vec)[0]
            highlights.append({
                "word": str(word),
                "category": str(class_idx)
            })
    return highlights

def get_decision_path(text: str) -> list:
    """
    Traces the decision tree path for a given text input and returns the nodes traversed.
    Used in Explainable AI Node splits.
    """
    vec = vectorizer.transform([text.lower()]).toarray()[0]
    tree = dt_classifier.tree_
    feature_names = vectorizer.get_feature_names_out()
    
    path = []
    node_id = 0
    
    # Traverse until a leaf node is hit
    while tree.children_left[node_id] != tree.children_right[node_id]: # not leaf
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_idx]
        
        # Check user's value
        user_val = vec[feature_idx]
        is_greater = user_val > threshold
        
        node_description = f"Does the response contain '{feature_name}' keyword?"
        decision = f"Yes (value {float(user_val):.2f} > {float(threshold):.2f})" if is_greater else f"No (value {float(user_val):.2f} <= {float(threshold):.2f})"
        
        path.append({
            "node_id": int(node_id),
            "description": node_description,
            "decision": decision,
            "outcome": "left" if not is_greater else "right"
        })
        
        if is_greater:
            node_id = tree.children_right[node_id]
        else:
            node_id = tree.children_left[node_id]
            
    # Leaf node reached
    leaf_value = tree.value[node_id][0]
    class_idx = np.argmax(leaf_value)
    predicted_class = dt_classifier.classes_[class_idx]
    
    path.append({
        "node_id": int(node_id),
        "description": "Leaf Node reached",
        "decision": f"Recommendation: {str(predicted_class)}",
        "outcome": "leaf"
    })
    
    return path

# ─────────────────────────────────────────────────────────────────────────────
# Dry Run Verification Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_text = "I love coding software and setting up databases on cloud servers."
    print("Testing Hybrid ML Inference...")
    print("Probabilities:", predict_interest(test_text))
    print("\nLinguistic highlights:", get_highlighted_words(test_text))
    print("\nDecision Tree Path:")
    for step in get_decision_path(test_text):
        print(f"Node {step['node_id']}: {step['description']} -> {step['decision']}")
