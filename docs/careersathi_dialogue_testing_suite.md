# CareerSaathi: Verified Dialogue Testing Suite & Demonstration Guide
*Navrachana University • Department of Computer Science & Engineering*

All test cases in this document have been **pre-tested and verified** against the local on-device SLM (`gemma2:2b` via Ollama) and the Sentence-BERT psychometric engine. Entering these exact prompts will produce the verified, deterministic recommendations.

---

## How to Start Your Testing Session

1. Open your terminal in the project directory (`CareerSaathi`).
2. Run the server launcher:
   ```powershell
   python run.py
   ```
3. Open your browser at: **`http://127.0.0.1:8080`**

---

## Test Scenario 1: 10th Pass Student -> Science (PCM) & Software Track

* **Goal**: Test that a 10th-grade student interested in coding and mobile apps is correctly recommended the **Science (PCM) Stream** leading to **B.Tech in Computer Science & Artificial Intelligence**, without getting sidetracked into mechanical robotics.
* **Landing Screen Selection**: Click **`AFTER 10TH: Choosing a stream`**

### Turn-by-Turn Script

| Turn | Stage on Journey Rail | What You Type / Click | What Sathi Explores |
| :---: | :--- | :--- | :--- |
| **0** | *Landing* | Click **`Start talking`** on the **After 10th** card | Sathi asks opening exhibition dilemma (Tech vs Trade vs Arts) |
| **1** | `1. Interests` | `Building software or tech tools` *(Click chip or type)* | Sathi acknowledges coding interest and asks what software you want to build |
| **2** | `1. Interests` | `I like how mobile apps work and solving computer logic` | Sathi explores how you break down complex app problems |
| **3** | `2. Thinking style` | `Like automating daily tasks for people with clean code` | Sathi moves to thinking style under problem solving |
| **4** | `2. Thinking style` | `I prefer breaking down complex algorithms step by step` | Sathi asks about your preferred work environment |
| **5** | `3. Strengths` | `Deep solo technical focus writing backend software` | Sathi confirms your focus on code execution |

### Expected Verified Output (Admissions Dossier):
* **Primary Recommendation**: **Science (PCM) Stream (Pathway to B.Tech in Computer Science & Artificial Intelligence)**
* **Dominant Cognitive Trait**: **Investigative (I) at ~39%**, followed by Realistic (R) at ~33%
* **Fit Score**: **~64%**
* **Target Indian Entrance Exams**: `JEE Mains, JEE Advanced, GUJCET, NUV-DET`
* **Evidence Quotes**: Quotes your statements on *"automating daily tasks"* and *"breaking down algorithms step by step"*.

---

## Test Scenario 2: 10th Pass Student -> Commerce & Finance Track

* **Goal**: Test that a 10th-grade student interested in business, strategy, and negotiation is guided to the **Commerce & Finance Stream** leading to **BBA / CA**.
* **Landing Screen Selection**: Click **`AFTER 10TH: Choosing a stream`**

### Turn-by-Turn Script

| Turn | Stage on Journey Rail | What You Type / Click | What Sathi Explores |
| :---: | :--- | :--- | :--- |
| **0** | *Landing* | Click **`Start talking`** on the **After 10th** card | Sathi presents opening project dilemma |
| **1** | `1. Interests` | `Business, finance or marketing` *(Click chip or type)* | Sathi asks what specifically excites you about commercial ventures |
| **2** | `1. Interests` | `I enjoy business marketing campaigns, client negotiations and launching ventures` | Sathi probes how you approach planning a commercial launch |
| **3** | `2. Thinking style` | `When a project runs into chaos I reorganize the financial budget and negotiate` | Sathi explores your crisis-handling and tactical problem solving |
| **4** | `2. Thinking style` | `Commercial market strategy, investment pitching and business growth` | Sathi asks about your collaboration and operational preferences |
| **5** | `3. Strengths` | `Collaborative leadership, corporate operations and client management` | Sathi completes the evaluation |

### Expected Verified Output (Admissions Dossier):
* **Primary Recommendation**: **Commerce & Finance Stream (Pathway to BBA in Business Analytics & Entrepreneurship)**
* **Dominant Cognitive Trait**: **Enterprising (E) at ~42%**, followed by Conventional (C)
* **Fit Score**: **~78%**
* **Target Indian Entrance Exams**: `IPMAT, SET, CUET UG, NUV-DET`
* **Alternatives**: `Chartered Accountancy (CA) & B.Com, B.A. LL.B Corporate Law`

---

## Test Scenario 3: 12th Pass Student -> Creative Digital UI/UX Design

* **Goal**: Test that a 12th-grade student who loves sketching, aesthetics, and digital apps is recommended **B.Des in Product Design & Digital UI/UX**.
* **Landing Screen Selection**: Click **`AFTER 12TH: Choosing a degree`**

### Turn-by-Turn Script

| Turn | Stage on Journey Rail | What You Type / Click | What Sathi Explores |
| :---: | :--- | :--- | :--- |
| **0** | *Landing* | Click **`Start talking`** on the **After 12th** card | Sathi presents 4-year degree immersion opening |
| **1** | `1. Interests` | `UI/UX product design and arts` *(Click chip or type)* | Sathi asks what draws you to digital arts |
| **2** | `1. Interests` | `I love visual design aesthetics, sketching user interfaces and digital prototyping` | Sathi explores how you translate ideas into visual experiences |
| **3** | `2. Thinking style` | `I make visual wireframes, user journeys and creative layouts` | Sathi probes your creative problem-solving workflow |
| **4** | `2. Thinking style` | `Typography, digital art styling, design thinking and creative storytelling` | Sathi explores your studio and collaborative style |
| **5** | `3. Strengths` | `Creative studio environment designing mobile app interfaces and graphics` | Sathi synthesizes your creative aptitude |

### Expected Verified Output (Admissions Dossier):
* **Primary Recommendation**: **B.Des in Product Design & Digital UI/UX**
* **Stream**: **Liberal Studies & Arts**
* **Dominant Cognitive Trait**: **Artistic (A) at ~46%**, followed by Realistic (R)
* **Fit Score**: **~79%**
* **Target Indian Entrance Exams**: `UCEED (IITs), NID DAT, NIFT`

---

## Test Scenario 4: 12th Pass Student -> Clinical Healthcare & Medicine (PCB)

* **Goal**: Test that a biology enthusiast is directed to **MBBS / Clinical Healthcare & Medicine**.
* **Landing Screen Selection**: Click **`AFTER 12TH: Choosing a degree`**

### Turn-by-Turn Script

| Turn | Stage on Journey Rail | What You Type / Click | What Sathi Explores |
| :---: | :--- | :--- | :--- |
| **0** | *Landing* | Click **`Start talking`** on the **After 12th** card | Sathi presents opening question |
| **1** | `1. Interests` | `Medical, biology and healthcare` *(Click chip or type)* | Sathi explores biological curiosity |
| **2** | `1. Interests` | `Investigating cellular genetics, microbiology and patient healing` | Sathi asks how you approach complex biological questions |
| **3** | `2. Thinking style` | `I carefully observe laboratory data and test biological hypotheses` | Sathi explores your scientific analytical method |
| **4** | `2. Thinking style` | `Human physiological anatomy, cellular immunology and healthcare diagnostics` | Sathi asks about your work environment |
| **5** | `3. Strengths` | `Clinical laboratory research, patient consultations and therapy` | Sathi finalizes healthcare alignment |

### Expected Verified Output (Admissions Dossier):
* **Primary Recommendation**: **MBBS / Clinical Healthcare & Medicine**
* **Stream**: **Science (PCB)**
* **Dominant Cognitive Trait**: **Investigative (I) & Social (S)**
* **Target Indian Entrance Exams**: `NEET UG`
* **Alternatives**: `B.Sc in Life Sciences & Biotechnology, B.A. (Hons) in Psychology`

---

## Test Scenario 5: Mixed-Initiative Dialogue (Student Asks a Question)

* **Goal**: Test that if you ask a question during the assessment, CareerSathi answers authoritatively **without lecturing or losing conversational state**.
* **When to test**: Any turn between Turn 1 and Turn 5.

### Sample Questions to Enter:
1. `what kind of software are there`
   * **Expected Response**: Direct, conversational explanation of consumer apps, design software, and automation scripts without preaching about exams.
2. `what is the difference between B.Tech CS and BCA?`
   * **Expected Response**: Clarifies engineering curriculum vs computer applications, then transitions smoothly back to the assessment.
3. `can I do bioinformatics if I take PCB without maths?`
   * **Expected Response**: Confirms eligibility with introductory university bridge courses, then returns to your strengths.

---

## Test Scenario 6: Human-in-the-Loop Recalibration ("I didn't like this suggestion")

* **Goal**: Test that CareerSathi can recover from student disagreement through active learning.
* **Steps**:
  1. Complete any scenario until the Dossier opens on the right.
  2. In the right pane, click: **`[ 🔄 I didn't like this suggestion — Let's refine ]`**
  3. Notice that the left chat receives a prompt:
     > *"Understood! What specifically felt off about [Recommended Degree]?"*
  4. Type in the left chat:
     `I don't like finance or spreadsheets, I want something in creative technology and digital arts`
  5. **Expected Result**:
     * The backend penalizes Conventional (C) and Enterprising (E).
     * Boosts Artistic (A) and Investigative (I).
     * The right-hand Dossier **live-updates** to **B.Des in Product Design & Digital UI/UX** or **B.Tech Computer Science & AI**!

---

## Demonstration & Evaluation Checklist

During your project walkthrough:

1. **Start from Clean Landing Page**: Show the distinction between *After 10th* (Stream selection) and *After 12th* (Degree selection).
2. **Show the Checkmarks**: Point out how the left Journey Rail marks completed stages with `✓` (`Interests` $\rightarrow$ `Thinking style` $\rightarrow$ `Strengths`).
3. **Show Claude-Style Split-View**: Point out that when the Dossier is generated, the chat is **never lost** — it sits on the left for evaluation.
4. **Click the Evidence Quotes**: In the *"Why this fits you"* section, click a quoted sentence: the left chat will smoothly scroll directly to that exact user message!
5. **Open AI Reasoning & Telemetry (XAI)**:
   * Click **`[ 🔬 AI Reasoning (XAI) ]`** at the top right of the Dossier.
   * Point out the live **Holland 6D Radar Chart** plotted in real-time SVG ($R^6$ coordinates).
   * Point out the **Normalized Shannon Entropy H** formula verifying mathematical convergence.
   * Point out the **spaCy POS Action Verbs** and **TF-IDF Vocabulary attribution**.
   * Point out the **Decision Tree splits** and deterministic trajectory traversal.
6. **Print / Export PDF**: Click **`[ 📥 Download PDF ]`** to show the print-optimized institutional report ready for Navrachana admissions!
