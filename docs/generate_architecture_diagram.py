import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up figure
fig = plt.figure(figsize=(22, 14), facecolor='#0b0f19', dpi=300)
ax = fig.add_subplot(111, facecolor='#0b0f19')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title block
ax.text(50, 96.8, 'CAREERSAATHI : HYBRID NLP ARCHITECTURE & INFERENCE PIPELINE', 
        ha='center', va='center', fontsize=20, fontweight='bold', color='#ffffff', fontfamily='sans-serif')
ax.text(50, 93.8, 'Cognitive Career Advisory Suite | 100% On-Device Local Acceleration (NVIDIA RTX 4060 + Local Gemma-2B)', 
        ha='center', va='center', fontsize=12, color='#38bdf8', fontfamily='sans-serif')

# Helper to draw rounded cards
def draw_card(x, y, w, h, title, subtitle, color, bg='#131c31', border='#223154'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.8,rounding_size=1.2', 
                                  linewidth=1.5, edgecolor=border, facecolor=bg)
    ax.add_patch(rect)
    if title:
        t_box = patches.FancyBboxPatch((x + 1.2, y + h - 3.8), w - 2.4, 3.2, boxstyle='round,pad=0.3,rounding_size=0.6',
                                       linewidth=1, edgecolor=color, facecolor=color, alpha=0.15)
        ax.add_patch(t_box)
        ax.text(x + 2.5, y + h - 2.2, title, fontsize=11, fontweight='bold', color=color, va='center')
    if subtitle:
        ax.text(x + 2.5, y + h - 5.5, subtitle, fontsize=8.5, color='#94a3b8', va='center', style='italic')

# Helper for flow arrows
def draw_arrow(x1, y1, x2, y2, color='#38bdf8', style='->', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=4, shrinkB=4))

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLIENT LAYER (Left Column)
# ─────────────────────────────────────────────────────────────────────────────
draw_card(3, 52, 26, 38, 'CLIENT LAYER (Glassmorphism SPA)', 'Interactive UI & Dual Input Modality', '#38bdf8')
ax.text(5, 83.5, 'Input Modality 1: Custom Text Input', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(6.5, 80.5, 'Supports English + Hinglish transliteration\n(e.g., "coding pasand hai but maths me doubt")\nDefends against one-word or troll inputs.', fontsize=8, color='#cbd5e1')

ax.text(5, 73.5, 'Input Modality 2: Voice Mic (Web Speech API)', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(6.5, 70.8, 'Direct browser-native speech-to-text conversion.\nAllows student to speak aloud in viva.', fontsize=8, color='#cbd5e1')

ax.text(5, 64.5, 'Dynamic Progress State-Machine', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(6.5, 62.0, 'Auto-adapts Turn Denominator (Step 1/6 for 12th,\nStep 1/5 for 10th) based on path taken.', fontsize=8, color='#cbd5e1')

ax.text(5, 56.5, 'Fast Asynchronous Fetch Protocol', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(6.5, 54.0, 'POST /api/chat passing complete conversation\nbelief state to stateless FastAPI backend.', fontsize=8, color='#cbd5e1')

# Real-time Telemetry Dashboard (Client Output)
draw_card(3, 8, 26, 40, 'REAL-TIME TELEMETRY & XAI OUTPUT', 'Live Cognitive Visualizations on Client DOM', '#34d399')
ax.text(5, 41.5, 'Live 6D RIASEC Radar Chart (SVG Polygon)', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(6.5, 38.8, 'Real-time polygon shifting between R, I, A, S, E, C.\nVisually illustrates psychometric convergence.', fontsize=8, color='#cbd5e1')

ax.text(5, 33.0, 'Indian Stream Alignment Gauges', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(6.5, 30.5, 'Live progress bars for Science PCM, PCB, Commerce\nand Liberal Studies / Arts.', fontsize=8, color='#cbd5e1')

ax.text(5, 24.8, 'Linguistic Attribution Cloud (TF-IDF XAI)', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(6.5, 22.3, 'Active words tagged with RIASEC color codes\n(e.g., algorithms [I], sketches [A], startups [E]).', fontsize=8, color='#cbd5e1')

ax.text(5, 16.5, 'Final Comprehensive Career Dossier', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(6.5, 12.5, 'Scorecard overlay displaying Primary/Secondary match,\nstudent quote justifications, and Indian entrance\nexam roadmap (JEE, CUET, CLAT, NIFT, IPMAT).', fontsize=8, color='#cbd5e1')

# ─────────────────────────────────────────────────────────────────────────────
# 2. NLP BACKEND PIPELINE (Center Column: Core Cognitive Architecture)
# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Preprocessing
draw_card(33, 73, 34, 17, 'STAGE 1: LINGUISTIC PREPROCESSING', 'Lexemes, Morphemes, POS Tagging & Indian Normalization', '#a78bfa')
ax.text(35, 84.5, '• Unicode Normalization & Hinglish Transliteration Regex', fontsize=8.5, color='#e2e8f0')
ax.text(35, 81.8, '• Tokenization & Stopword Filtering (NLTK / SpaCy en_core_web_sm)', fontsize=8.5, color='#e2e8f0')
ax.text(35, 79.1, '• Lemmatization & Morphological Analysis (Resolving base morphemes)', fontsize=8.5, color='#e2e8f0')
ax.text(35, 76.4, '• Syntactic POS Tagging: Extracts Action Verbs & Noun Chunks', fontsize=8.5, color='#e2e8f0')

# STAGE 2: Deep Dense Embeddings
draw_card(33, 49, 34, 21, 'STAGE 2: PSYCHOMETRIC VECTOR SPACE', 'Deep Dense Representations & Centroid Cosine Geometry', '#38bdf8')
ax.text(35, 63.8, '• Sentence-BERT Encoder (all-MiniLM-L6-v2) on PyTorch CUDA', fontsize=8.5, color='#e2e8f0')
ax.text(35, 61.1, '  Maps student text into 384-dimensional dense vector v_user in R^384', fontsize=8, color='#94a3b8')
ax.text(35, 58.2, '• 6 Holland RIASEC Class Centroid Prototypes: mu_R, ..., mu_C', fontsize=8.5, color='#e2e8f0')
ax.text(35, 55.5, '• Cosine Similarity Mapping: Sim(v_user, mu_k) = (v . mu) / (||v|| ||mu||)', fontsize=8.5, color='#38bdf8', fontweight='bold')
ax.text(35, 52.8, '• Bayesian Momentum State Update: S_t = alpha * S_{t-1} + (1 - alpha) * a_t', fontsize=8.5, color='#f8fafc', fontweight='bold')

# STAGE 3: Sparse Vectorization
draw_card(33, 26, 34, 20, 'STAGE 3: SPARSE VECTORIZATION & XAI', 'N-Grams, TF-IDF Matrices & Tree-Based Attribution', '#fb923c')
ax.text(35, 40.0, '• TF-IDF Vectorizer with unigrams & bigrams (ngram_range=(1,2))', fontsize=8.5, color='#e2e8f0')
ax.text(35, 37.3, '  Sparse feature matrix dimension V ~ 1,105 tokens, sublinear TF', fontsize=8, color='#94a3b8')
ax.text(35, 34.5, '• Random Forest Classifier (150 Tree Ensembles) + Decision Tree', fontsize=8.5, color='#e2e8f0')
ax.text(35, 31.8, '• Single-token Attribution Pass: Maps active keywords to RIASEC poles', fontsize=8.5, color='#e2e8f0')
ax.text(35, 29.0, '• Decision Tree Split Logging: Traces exact node threshold routing', fontsize=8.5, color='#fb923c', fontweight='bold')

# STAGE 4: Ambiguity Math
draw_card(33, 4, 34, 19, 'STAGE 4: AMBIGUITY MATH & DIALOGUE CONTROLLER', 'Information Theory & Finite-State Dialogue Policy', '#f43f5e')
ax.text(35, 17.0, '• Normalized Shannon Entropy: H(P) = - (1 / log2 6) * sum(P_k * log2 P_k)', fontsize=9, color='#f43f5e', fontweight='bold')
ax.text(35, 14.0, '• If H > 0.85 (High Ambiguity / Vague Response):', fontsize=8.5, color='#f8fafc', fontweight='bold')
ax.text(36.5, 11.8, '  Trigger PROBING MODE: Formulate dilemma between top 2 competing traits', fontsize=8, color='#cbd5e1')
ax.text(35, 9.2, '• If H <= 0.85 (Low Ambiguity / Converged Preference):', fontsize=8.5, color='#f8fafc', fontweight='bold')
ax.text(36.5, 7.0, '  Trigger NARROWING MODE: Target specific Indian Stream & Degree domain', fontsize=8, color='#cbd5e1')

# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATION & ONTOLOGY (Right Column)
# ─────────────────────────────────────────────────────────────────────────────
draw_card(71, 52, 26, 38, 'STAGE 5: LOCAL ON-DEVICE SLM', 'Real-Time NLG (Zero Cloud API / Zero Paywall)', '#e879f9')
ax.text(73, 83.5, 'Local Ollama Engine on localhost:11434', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(74.5, 81.0, 'Connected via async HTTP request from FastAPI.\nCompletely offline with zero external network lag.', fontsize=8, color='#cbd5e1')

ax.text(73, 74.5, 'Google Gemma-2B Model (gemma2:2b)', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(74.5, 72.0, 'Lightweight 2-Billion parameter open-weights SLM.\nUnder 1.6 GB RAM. Exceptionally natural dialogue.', fontsize=8, color='#cbd5e1')

ax.text(73, 65.5, 'NVIDIA RTX 4060 GPU Acceleration', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(74.5, 63.0, '100% CUDA layer offload via 8GB GDDR6 VRAM.\nBlazing throughput: 80 - 110 tokens/second (<300ms)!', fontsize=8, color='#38bdf8', fontweight='bold')

ax.text(73, 56.5, 'Psychometric Context Conditioning', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(74.5, 53.8, 'Dynamically injects student conversation context,\ntarget RIASEC dilemma, and Indian framing into prompt.', fontsize=8, color='#cbd5e1')

# Indian Knowledge Ontology Matrix
draw_card(71, 8, 26, 40, 'STAGE 6: INDIAN EDUCATION ONTOLOGY', 'Stream Mapping, Degree Specializations & Entrance Exams', '#eab308')
ax.text(73, 41.5, '10th Standard Stream Mapping Matrix', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(74.5, 39.0, 'Score_Streams = W_Stream * S_final in R^4\n- Science (PCM): Realistic + Investigative\n- Science (PCB): Investigative + Social\n- Commerce: Enterprising + Conventional\n- Liberal Studies / Arts: Artistic + Social', fontsize=8, color='#cbd5e1')

ax.text(73, 27.5, '12th Standard Degree Specializations', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(74.5, 25.0, 'Engineering (B.Tech CS, Mech, Civil, Biotech),\nPure Sciences (B.Sc Physics, Chem, Math),\nCommerce (CA, B.Com, Finance, FinTech),\nManagement (BBA, Analytics), Arts (Psychology, Lit).', fontsize=8, color='#cbd5e1')

ax.text(73, 16.5, 'Competitive Entrance Exam Roadmap', fontsize=9.5, fontweight='bold', color='#f8fafc')
ax.text(74.5, 12.5, 'Provides targeted testing paths:\n- Engineering: JEE Mains & Advanced\n- Commerce/Management: IPMAT, CA Foundation\n- Arts & Sciences: CUET UG, CLAT, NIFT / UCEED', fontsize=8, color='#cbd5e1')

# ─────────────────────────────────────────────────────────────────────────────
# FLOW ARROWS & DATA BUS
# ─────────────────────────────────────────────────────────────────────────────
draw_arrow(29, 71, 33, 81.5, '#38bdf8', lw=2.2)      # Client input -> Preprocessing
draw_arrow(50, 73, 50, 70, '#a78bfa', lw=2.2)        # Preprocessing -> Stage 2
draw_arrow(50, 49, 50, 46, '#38bdf8', lw=2.2)        # Stage 2 -> Stage 3
draw_arrow(50, 26, 50, 23, '#fb923c', lw=2.2)        # Stage 3 -> Stage 4

draw_arrow(67, 13.5, 71, 71, '#e879f9', lw=2.5)      # Stage 4 Entropy -> Local Gemma SLM
draw_arrow(71, 68, 29, 68, '#38bdf8', lw=2.5)        # Local Gemma Dynamic Q -> Client UI

draw_arrow(67, 59.5, 71, 28, '#eab308', lw=2.2)      # Stage 2 S_final -> Indian Ontology
draw_arrow(71, 28, 29, 28, '#34d399', lw=2.5)        # Indian Ontology -> Client Telemetry

# Save figure in docs and brain
out_path_1 = os.path.join(os.path.dirname(__file__), 'system_architecture_pipeline.png')
out_path_2 = r'C:\Users\dhruv\.gemini\antigravity\brain\37ff4593-e160-48e3-aea1-df1ce3769575\system_architecture_pipeline.png'

plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
fig.savefig(out_path_1, facecolor='#0b0f19', edgecolor='none')
fig.savefig(out_path_2, facecolor='#0b0f19', edgecolor='none')
plt.close(fig)

print("SUCCESS: Image generated successfully at:")
print("Path 1:", out_path_1)
print("Path 2:", out_path_2)
