# 🌌 AuraPath AI: Explainable Adaptive Career Guidance & Cognitive Admissions Suite

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--BERT-34D058?style=for-the-badge&logo=pytorch&logoColor=white)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Vanilla CSS](https://img.shields.io/badge/Vanilla_CSS-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**AuraPath AI** is a state-of-the-art, interactive career counselling and cognitive university admissions system. By utilizing an adaptive dialogue model and an explainable machine learning backend, it replaces rigid, traditional rule-based questionnaires with an intelligent, conversational interviewer. Through indirect, scenario-based questioning, it uncovers a student’s latent interests, projects their aptitude across eight foundational academic disciplines, and visualizes the underlying decision-making patterns in real-time.

---

## 🚀 Interactive Showcase Preview

### 1. Adaptive Profiling Start (`Step: 1 / 6`)
An elegant initial conversation card sets up the student's background (Education and Stream profiling) before dynamically branching into custom interest-matching tracks.
*   **Step Indicator**: `Step: 1 / 6`
*   **Prior Probability Distribution**: Balanced flat priors at $12.5\%$ for all disciplines.

### 2. Live Explainable ML Telemetry
As the student responds to the interactive scenario cards, the right-hand dashboard animates dynamically:
*   **Random Forest / S-BERT Probabilities**: Real-time progress bars showing fit strength.
*   **TF-IDF Linguistic Features**: Active terms highlighted and tagged (e.g., `coding (CSE)`).
*   **Decision Tree splits**: The program's mathematical path highlighted step-by-step.

### 3. Ultimate Scoring Report Overlay
Once the 6-turn chat concludes, a stunning glassmorphism scorecard overlays the system, providing personalized career matching, secondary backup tracks, and explainable logic reports.

---

## 🌟 Key Product Features

*   **🎭 Conversational Scenario Interviewer**: Engages the student through fun, indirect, scenario-based questions (such as assembling a go-kart, curing a strange plant, or launching a recycled t-shirt brand) to extract underlying motivations rather than asking direct, boring questions.
*   **🧠 Hybrid Dual-Engine ML Classifier**:
    *   *Primary Engine*: **Sentence-BERT (`all-MiniLM-L6-v2`)** dense vector space embeddings calculating Cosine Similarity against class prototype centroids, scaled via a Temperature Softmax ($T = 12.0$) for realistic probability spreads.
    *   *Secondary Engine*: **TF-IDF + Random Forest Classifier (150 Ensembles)** running in parallel for extremely responsive, reliable backup inferences.
*   **👁️ True Explainable AI (XAI) Panel**:
    *   *Linguistic Attributions*: TF-IDF tokens are extracted in real-time from open text fields, displaying direct term mappings to specific target branches.
    *   *Logic Path Traversal*: The program records the exact node-by-node traversal down an explainable Decision Tree, demonstrating how the model evaluates answers.
*   **💎 High-Fidelity Glassmorphism Interface**: Sleek, modern design featuring subtle backdrop blur, radial ambient glow, HSL color-coded telemetry bars, dynamic micro-animations, and full mobile responsiveness.
*   **📊 Dynamic Progress Telemetry**: An intelligent state-machine tracking the exact conversation history to render step counts dynamically (adapting to `Step: X / 6` or `Step: X / 5` depending on selected profiling answers).

---

## 📂 Codebase File Tree

```directory
AuraPath-AI/
│
├── backend/
│   ├── __init__.py          # Python package initializer
│   ├── main.py              # FastAPI server routes, session logic, and API handlers
│   ├── model.py             # Dual-engine ML pipeline, dataset generation, and XAI splits
│   └── questions.py         # Scenario database, bridging statements, and seed questions
│
├── frontend/
│   ├── index.html           # Structure for chat panel, XAI dashboards, and scorecard overlays
│   ├── script.js            # Client state-machine, Server API communications, and SVG chart animations
│   └── style.css            # Custom CSS styles, Glassmorphism variables, dark themes, and animations
│
├── docs/
│   ├── architecture.md      # Detailed system lifecycle and Mermaid block designs
│   ├── algorithms_math.md   # Mathematical details of BERT, Softmax, Random Forest, & XAI traversing
│   └── design_system.md     # Styling guides, UI components, animations, and responsive grids
│
├── run.py                   # Automated server compiler, bootstrapping manager, and browser launcher
└── LICENSE                  # Project license (MIT)
```

---

## 🛠️ Technology Stack

| Component | Framework / Library | Primary Purpose |
| :--- | :--- | :--- |
| **Backend Engine** | Python 3.10+ / FastAPI | Core API hosting, fast session processing, and JSON static routing. |
| **Deep Learning** | Sentence-Transformers | Dense vector space representation and Cosine Similarity mapping. |
| **Machine Learning**| Scikit-Learn | Random Forest models, Decision Trees, and TF-IDF feature matrices. |
| **Frontend UI** | HTML5 / CSS3 (Vanilla) | Immersive Glassmorphism, animations, layouts, and responsive grids. |
| **Client Control** | Javascript (ES6+) | State machines, event handlers, and SVG dynamic bars. |
| **Execution Tool** | Uvicorn / Webbrowser | Automatic pipeline compilation and direct local browser hosting. |

---

## ⚡ Developer Installation & Quickstart

To run the full suite locally, follow these simple developer steps:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/aurapath-ai.git
cd aurapath-ai
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy scikit-learn fastapi uvicorn sentence-transformers
```
> **Note**: On the first run, the system will automatically download the lightweight Sentence-Transformers model `all-MiniLM-L6-v2` (~90MB). If PyTorch/Sentence-Transformers are not installed, the application will automatically fall back to the classical high-performance Random Forest TF-IDF mode, ensuring uninterrupted operation!

### 4. Run the Integrated Launcher
Simply execute the root launcher script. This will compile the dataset, train the classifiers in a split-second, spin up the local server, and automatically open your default browser to the web app:
```bash
python run.py
```
Your console will output:
```text
===========================================================================
  TRADITIONAL ML CAREER COUNSELLOR - UNIVERSITY LAB PRACTICAL
===========================================================================
  [ML Pipeline] Bootstrapping Synthetic Dataset & Training...
  [ML Pipeline] [Sentence-Transformers] Loading all-MiniLM-L6-v2 semantic model...
  [ML Pipeline] [Sentence-Transformers] Pre-computing semantic class prototype embeddings...
  [ML Pipeline] [Sentence-Transformers] [OK] Semantic class prototype embeddings ready!
  [ML Pipeline] [OK] Successfully trained classifiers in 16.6 seconds!
  [ML Pipeline] [OK] Training samples generated: 182
  [ML Pipeline] [OK] TF-IDF Vocabulary dimension: 1105
  [ML Pipeline] [OK] Random Forest Ensembles: 150 estimators
  [ML Pipeline] [OK] Decision Tree depth restriction: max_depth=5
---------------------------------------------------------------------------
  [Server] Initializing FastAPI Web App & Static Files mount...
  [Server] Running locally at: http://127.0.0.1:8080
===========================================================================
[Launcher] Opening browser automatically to http://127.0.0.1:8080 ...
```

---


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
