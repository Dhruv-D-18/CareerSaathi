# 🧮 AuraPath AI: Hybrid Classification Pipeline & Explainable AI (XAI) Mathematics

This document explains the mathematical foundations and algorithmic logic that powers **AuraPath AI**. It details the dual-engine classification model, embedding dimensions, cosine similarity formulas, temperature-scaled softmax functions, TF-IDF representations, and the explainability pipelines (attribution and node splits).

---

## 🧠 Part 1: Hybrid Dual-Engine Classifier Mathematics

AuraPath AI utilizes a **Hybrid Dual-Engine Classifier** designed to perform semantic document mapping while maintaining traditional, highly granular word-level transparency.

---

### 🌟 Engine 1: Semantic Deep Learning Model (Sentence-BERT)

The primary classifier relies on **Sentence-BERT (S-BERT)**, specifically the pre-trained `all-MiniLM-L6-v2` transformer model. It maps arbitrary sequences of natural language text into a dense vector space.

#### 1. Vector Space Representation
Let $S$ represent a sequence of characters (e.g., student text input). The S-BERT encoder maps this string into a high-dimensional dense vector space:
$$v_s = E(S) \in \mathbb{R}^{384}$$
Where:
*   $E(\cdot)$ is the Sentence-Transformers encoding function.
*   $v_s$ is a dense vector of size $384$.

#### 2. Class Prototype Centroid Computation
During server boot, AuraPath AI compiles a representative interest vocabulary (compiled inside `DATASET`). For each of the $K = 8$ career categories $C_k$, the system pre-computes a **Class Prototype Centroid Vector** $\mu_k \in \mathbb{R}^{384}$.

Let $\{S_1^{(k)}, S_2^{(k)}, \dots, S_{N_k}^{(k)}\}$ be the set of $N_k$ synthetic interest statements for category $C_k$. The centroid vector $\mu_k$ represents the average semantic profile of that category:
$$\mu_k = \frac{1}{N_k} \sum_{i=1}^{N_k} E\left(S_i^{(k)}\right)$$
Each centroid $\mu_k$ acts as a multi-dimensional semantic anchor representing the "ideal profile" of a career track (e.g., *Computer Science*).

#### 3. Cosine Similarity Mapping
When a student submits custom text $S_{\text{user}}$, the system embeds it to obtain $v_{\text{user}} = E(S_{\text{user}})$. It then calculates the angular similarity between the student’s profile and all $K = 8$ class centroids using the **Cosine Similarity** metric:
$$\text{Sim}(v_{\text{user}}, \mu_k) = \cos(\theta_k) = \frac{v_{\text{user}} \cdot \mu_k}{\|v_{\text{user}}\|_2 \|\mu_k\|_2}$$
Where:
*   $v_{\text{user}} \cdot \mu_k$ is the vector dot product: $\sum_{d=1}^{384} v_{\text{user}, d} \times \mu_k, d$
*   $\|v\|_2$ is the Euclidean $L_2$ norm: $\sqrt{\sum_{d=1}^{384} v_d^2}$

Cosine similarity bounds the semantic alignment score strictly between $[-1.0, 1.0]$. A score close to $1.0$ indicates perfect semantic alignment, while a score near $0.0$ implies absolute orthogonality.

#### 4. Temperature-Scaled Softmax Scaling
Cosine similarity scores for high-quality representations usually cluster tightly within a narrow positive range (e.g., $[0.35, 0.75]$). To transform these raw similarity metrics into distinct, high-contrast probability distributions suited for visual telemetry, AuraPath AI applies a **Softmax function with Temperature Scaling**:
$$P(C_k \mid S_{\text{user}}) = \frac{e^{\text{Sim}(v_{\text{user}}, \mu_k) \cdot T}}{\sum_{j=1}^{K} e^{\text{Sim}(v_{\text{user}}, \mu_j) \cdot T}}$$
Where:
*   $T \in \mathbb{R}^+$ is the **Temperature Coefficient**.
*   In AuraPath AI, the temperature is set to **$T = 12.0$**.

##### Why Temperature Scaling is Critical:
*   **Without Temperature ($T=1.0$)**: The differences between similarity values are small, causing the Softmax output to collapse toward a flat, high-entropy distribution (e.g., $13.5\%$ vs $12.8\%$), which is difficult for users to interpret.
*   **With Temperature ($T=12.0$)**: The temperature acts as a contrast amplifier. It scales up the differences in similarity before exponentiation, expanding the contrast:
    *   High semantic matches are amplified towards $90\%+$.
    *   Fringe or weak matches are pushed down toward $0.1\%$, mimicking the sharp, confident classification thresholds of a human counselor.

---

### ⚡ Engine 2: Classical Machine Learning Model (Fallback & Attribution Layer)

If the `sentence-transformers` library is not available, the system falls back to a traditional pipeline combining **TF-IDF Vectorization** with a **Random Forest Ensemble**.

#### 1. TF-IDF Representation
Let $D = \{S_1, S_2, \dots, S_M\}$ be our training corpus. The **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorizer transforms input documents into a sparse, term-weighted vector space:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Where:
*   $\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in } d}$
*   $\text{IDF}(t, D) = \ln\left(\frac{1 + M}{1 + \text{Document Frequency of } t}\right) + 1$

AuraPath AI utilizes unigrams and bigrams ($\text{ngram\_range}=(1, 2)$) with stop-word filtration. This generates a robust feature vocabulary dimension of approximately $V \approx 1,105$ vocabulary dimensions.

#### 2. Random Forest Ensemble Probability
Our Random Forest Classifier fits $N_E = 150$ independent decision tree estimators. Each tree is trained on a bootstrap sample of the TF-IDF feature matrix. The final probability for target class $C_k$ is computed as the average class prediction across all $150$ trees:
$$P_{\text{RF}}(C_k \mid x) = \frac{1}{N_E} \sum_{e=1}^{N_E} P_{\text{tree}_e}(C_k \mid x)$$
Where $x$ is the TF-IDF vector of the student's text. This ensemble approach prevents overfitting and ensures robust, stable fallback predictions.

---

## 👁️ Part 2: Explainable AI (XAI) Mathematics

To ensure absolute transparency, AuraPath AI provides dual-layer explainability: **Linguistic Feature Highlights** (attribution) and **Decision Tree Path Traversals**.

---

### 📝 1. Real-time Linguistic Feature Highlights (Local Word Attribution)

For any user-submitted text $S_{\text{user}}$, the system identifies which words are active features in our vocabulary and determines their individual impact.

```text
Student Input: "I love coding software and setting up databases on cloud servers."
   ↓ (Tokenization & Filtering)
Active Tokens: ["coding", "software", "databases"]
   ↓ (Individual Single-Token Vectorization)
Feature Vectors:
   "coding"    ──→ [Vector] ──→ Random Forest ──→ Match: Computer Science (Highlighted!)
   "software"  ──→ [Vector] ──→ Random Forest ──→ Match: Computer Science (Highlighted!)
   "databases" ──→ [Vector] ──→ Random Forest ──→ Match: Computer Science (Highlighted!)
```

For each word $w \in S_{\text{user}}$:
1.  Verify if $w$ exists in our TF-IDF feature vocabulary: $w \in V$.
2.  If true, generate a single-token vector $x_w \in \mathbb{R}^V$ where the index of $w$ is set to $1.0$ and all other indices are $0.0$.
3.  Perform a single-token prediction pass through the Random Forest classifier:
    $$\hat{C}_w = \arg\max_{c} P_{\text{RF}}(C = c \mid x_w)$$
4.  If the model predicts a specific class with high confidence, the UI highlights $w$ using a color matched to category $\hat{C}_w$. This shows the user exactly how the model attributes individual keywords to different academic disciplines!

---

### 🌳 2. Decision Tree Node-Traversal Path Trace

In parallel with our classification models, we train a **Decision Tree Classifier** restricted to a maximum depth of $5$. This model is used specifically to trace logic paths.

```text
                       [Node 0: 'databases'?]
                            /          \
                (Value <= 0.05)      (Value > 0.05)
                        /                  \
             [Node 1: 'engine'?]      [Node 4: 'coding'?]
                  /         \             /         \
                ...         ...         No          Yes
                                         /            \
                                [Leaf Node 9]     [Leaf Node 10]
                                 CS (82.4%)         CS (98.9%)
```

#### Traversal Algorithm:
Let $node\_id$ start at the root node ($0$).
While $node\_id$ is not a leaf node (i.e., its left child is not equal to its right child):
1.  Identify the split feature index $j$ and split threshold $\tau$ associated with the node:
    $$\text{Feature Name} = V_j, \quad \text{Threshold} = \tau$$
2.  Evaluate the student's TF-IDF feature value $x_j$ against the threshold:
    *   If $x_j > \tau$: Move to the right child node:
        $$node\_id \leftarrow \text{child\_right}[node\_id]$$
        *Record Decision*: `"Yes (value > threshold)"`
    *   If $x_j \le \tau$: Move to the left child node:
        $$node\_id \leftarrow \text{child\_left}[node\_id]$$
        *Record Decision*: `"No (value <= threshold)"`

This traversal trace compiles the exact logic path. It outputs an ordered list of decisions explaining how the model branched from the root question to a specific career recommendation (e.g., *Management* or *Physics*), giving users clear insight into the decision-making process.
