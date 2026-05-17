# 🏗️ AuraPath AI: Integrated System Architecture & Lifecycle

This document provides a highly detailed explanation of the **AuraPath AI** system architecture, system sequence flows, state machine specifications, client-server data synchronization contracts, and backend routing endpoints. It is designed to help team members quickly understand the inner engineering operations of the codebase.

---

## 📌 Architectural Blueprint

AuraPath AI is constructed as a decoupled **Client-Server Single-Page Web Application (SPA)** utilizing a unified launcher. It blends high-performance web frameworks with state-of-the-art Python machine learning and deep learning dependencies.

```mermaid
graph TD
    subgraph Launcher [Integrated Launcher]
        run[run.py launcher]
    end

    subgraph Client [Client UI - Browser SPA]
        html[index.html Layout]
        css[style.css Theme / CSS Grid]
        js[script.js State Machine]
    end

    subgraph Server [Backend Server - FastAPI]
        api[main.py Routing & Session Manager]
        questions[questions.py Adaptive Scenarios]
        subgraph ML_Engine [Dual-Engine Classifier]
            model[model.py Core Pipeline]
            bert[Sentence-BERT Dense Vector Centroids]
            rf[TF-IDF Random Forest Ensemble]
            dt[Decision Tree XAI Traversal]
        end
    end

    %% Execution Flows
    run -->|Instantiates| api
    run -->|Spawns| Client
    js -->|JSON Payload State Sync| api
    api -->|Consults Pool| questions
    api -->|Runs Inference| model
    model -->|Primary Cosine Similarity| bert
    model -->|Secondary Backup & Highlight| rf
    model -->|Node split mapping| dt
    api -->|JSON Response State Sync| js
    js -->|Renders DOM & Charts| html
```

---

## 🔄 Client-Server Sequence Flow & Lifecycle

The application operates on a strict **State-Passing Protocol**. The client acts as the single source of truth for the conversation history and configuration variables, passing the complete state to the server on every request. The server functions as a stateless mathematical computer, updating the state variables and returning them alongside the classification predictions.

```mermaid
sequenceDiagram
    autonumber
    actor Student as Student
    participant UI as Browser DOM (HTML/CSS)
    participant Client as Client Engine (script.js)
    participant Server as FastAPI Server (main.py)
    participant Model as ML Engine (model.py)

    Note over Student, Server: 1. System Boot & Initialization
    Student->>UI: Opens page / Click Restart
    Client->>Client: initChat() - Resets chatState to Turn 0
    Client->>Server: POST /api/chat {chatState (empty latest_message)}
    Server->>Server: Turn 0: empty message check
    Server-->>Client: Returns initial Profiling Question (Education Level)
    Client->>UI: Renders Education Question & Chips, sets badge to Step 1/6

    Note over Student, Server: 2. Background Profiling
    Student->>UI: Selects "🎓 12th Standard Completed"
    Client->>Server: POST /api/chat {chatState, latest_message: "12th Standard"}
    Server->>Server: Records education level. Detects "12" -> branch to stream
    Server-->>Client: Returns stream selection question
    Client->>UI: Renders Stream question & chips, sets badge to Step 2/6

    Student->>UI: Selects "📐 Science with Maths (PCM)"
    Client->>Server: POST /api/chat {chatState, latest_message: "PCM"}
    Server->>Server: Records stream. Branches to scenario, appends bridge message
    Server-->>Client: Returns bridge message + 1st Scenario Question
    Client->>UI: Renders Scenario question, sets badge to Step 3/6

    Note over Student, Server: 3. Real-Time Explainable ML Assessment Loop
    loop Scenario Questions (Steps 3/6 to 6/6)
        Student->>UI: Clicks option OR types custom text response
        Client->>Server: POST /api/chat {chatState, latest_message: "Response"}
        Server->>Model: Passes user response for inference
        Model->>Model: Calculates Sentence-BERT similarities / TF-IDF Random Forest probabilities
        Model->>Model: Traverses Decision Tree to extract node split XAI
        Model->>Model: Extracts TF-IDF word attributions (highlights)
        Server->>Server: Turn count += 1, updates history logs
        Server-->>Client: Returns next Scenario Question + ML telemetry payload
        Client->>UI: Animates charts, updates highlighted keywords, updates Decision Tree steps
        Client->>UI: Renders next scenario question card, increments step badge
    end

    Note over Student, Server: 4. Final Evaluation & Report Render
    Student->>UI: Answers 4th (final) Scenario Question (Step 6/6)
    Client->>Server: POST /api/chat {chatState, latest_message: "Final Response"}
    Server->>Model: Final dual-engine inference run
    Server->>Server: Detects scenario turns completed (4 scenario questions) -> phase = "finished"
    Server-->>Client: Returns phase = "finished" + final ML predictions
    Client->>UI: Renders Personalized Career Report Card Overlay (Matches, Backup, Justification details)
```

---

## 📊 Client-Server API State Contracts

### 1. Request Protocol: `POST /api/chat`
The request must contain the entire current state serialised inside the JSON body:

```json
{
  "turn_count": 2,
  "phase": "scenario",
  "education_level": "12th Standard",
  "stream": "PCM",
  "asked_question_ids": ["general_start"],
  "history": [
    {"role": "assistant", "content": "First, what is your current education level?"},
    {"role": "user", "content": "12th Standard"},
    {"role": "assistant", "content": "Which academic stream did you study?"},
    {"role": "user", "content": "PCM"},
    {"role": "assistant", "content": "Great, PCM stream!... Imagine your school or colony is organizing..."}
  ],
  "latest_message": "The Tech Lead: Building the registration website, databases..."
}
```

### 2. Response Protocol
The server processes the payload and returns the next conversation block along with full machine learning telemetry:

```json
{
  "type": "question",
  "question": "Imagine you are designing a high-tech smart lock system for a secure vault. What is the most satisfying part of the setup?",
  "options": [
    {"text": "💻 Writing the software algorithm that encrypts credentials", "value": "I love coding software algorithms..."},
    {"text": "🔌 Wiring up the electronic locks, batteries, and sensors", "value": "I want to wire electronic microchips..."},
    {"text": "🛠️ Assembling the physical steel frame and mechanical hinges", "value": "Tinkering with heavy mechanical locks..."},
    {"text": "📊 Sourcing parts under budget and selling to banks", "value": "Negotiating with suppliers, pitching business..."}
  ],
  "state": {
    "turn_count": 3,
    "phase": "scenario",
    "education_level": "12th Standard",
    "stream": "PCM",
    "asked_question_ids": ["general_start", "cs_vault"],
    "history": [
      "...[all previous history]...",
      {"role": "user", "content": "The Tech Lead: Building the registration website, databases..."},
      {"role": "assistant", "content": "Imagine you are designing a high-tech smart lock..."}
    ],
    "latest_message": ""
  },
  "ml_data": {
    "probabilities": {
      "Computer Science": 82.4,
      "Management": 8.2,
      "Electrical Engineering": 4.1,
      "Mechanical Engineering": 2.1,
      "Civil Engineering": 1.0,
      "Biology": 0.8,
      "Chemistry": 0.8,
      "Physics": 0.6
    },
    "highlights": [
      {"word": "building", "category": "Computer Science"},
      {"word": "databases", "category": "Computer Science"},
      {"word": "website", "category": "Computer Science"}
    ],
    "decision_path": [
      {
        "node_id": 0,
        "description": "Does the response contain 'databases' keyword?",
        "decision": "Yes (value 1.00 > 0.05)",
        "outcome": "right"
      },
      {
        "node_id": 4,
        "description": "Does the response contain 'coding' keyword?",
        "decision": "No (value 0.00 <= 0.05)",
        "outcome": "left"
      },
      {
        "node_id": 9,
        "description": "Leaf Node reached",
        "decision": "Recommendation: Computer Science",
        "outcome": "leaf"
      }
    ]
  }
}
```

---

## 🏛️ Component Specifications

### 1. `backend/main.py`
Serves as the high-speed middleware router:
*   Imports configuration matrices from `questions.py` and predictions from `model.py`.
*   Directly mounts static assets at `/` via standard HTML/CSS files routing.
*   Implements the chat API handler `/api/chat`, running logical branching based on dynamic session rules.

### 2. `backend/questions.py`
The absolute database for adaptive conversations:
*   Maintains the `QUESTION_POOL` nested dictionary containing scenarios mapped to target career paths.
*   Houses the `SEED_QUESTIONS` dictionary containing the education profiling, stream profiling, and bridging text templates.
*   Implements `get_adaptive_question()`, which parses the current machine learning probabilities in descending order and dynamically fetches the highest-ranking unasked question to steer the next conversation turn!

### 3. `backend/model.py`
The predictive engine of AuraPath AI:
*   Initializes the hybrid NLP pipeline.
*   Pre-calculates centroids for dense vector embeddings or fits TF-IDF Random Forests and Decision Trees.
*   Executes high-speed inference, linguistic highlights generation, and node split traversal logs.

### 4. `frontend/script.js`
The central frontend coordinator:
*   Tracks local client state synchronization.
*   Directly queries backend REST APIs via `fetch`.
*   Dynamically manipulates DOM nodes, generates responsive animated SVG chips, compiles semantic inputs, and manages scorecard views.
