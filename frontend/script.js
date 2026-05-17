// ─────────────────────────────────────────────────────────────────────────────
// State Configuration
// ─────────────────────────────────────────────────────────────────────────────
let chatState = {
    turn_count: 0,
    phase: "profiling",
    education_level: null,
    stream: null,
    asked_question_ids: [],
    history: [],
    latest_message: ""
};

// ─────────────────────────────────────────────────────────────────────────────
// DOM References
// ─────────────────────────────────────────────────────────────────────────────
const chatContainer = document.getElementById("chat-container");
const optionsContainer = document.getElementById("options-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const loadingIndicator = document.getElementById("loading-indicator");
const turnBadge = document.getElementById("turn-badge");

// XAI References
const probabilitiesList = document.getElementById("probabilities-list");
const keywordCloud = document.getElementById("keyword-cloud");
const treePath = document.getElementById("tree-path");

// Scorecard References
const scorecardOverlay = document.getElementById("scorecard-overlay");
const primaryRecTitle = document.getElementById("primary-rec-title");
const secondaryRecTitle = document.getElementById("secondary-rec-title");
const justificationList = document.getElementById("justification-list");
const restartBtn = document.getElementById("restart-btn");

// Helper category color key mapping
const categoryClasses = {
    "Computer Science": "cs",
    "Mechanical Engineering": "mech",
    "Electrical Engineering": "elec",
    "Civil Engineering": "civil",
    "Biology": "bio",
    "Chemistry": "chem",
    "Physics": "phys",
    "Management": "mgmt"
};

// ─────────────────────────────────────────────────────────────────────────────
// Chat Execution Flows
// ─────────────────────────────────────────────────────────────────────────────

// Initialize conversation on page load
window.addEventListener("DOMContentLoaded", () => {
    initChat();
});

async function initChat() {
    // Reset local state
    chatState = {
        turn_count: 0,
        phase: "profiling",
        education_level: null,
        stream: null,
        asked_question_ids: [],
        history: [],
        latest_message: ""
    };
    
    // Clear UI
    chatContainer.innerHTML = "";
    optionsContainer.innerHTML = "";
    probabilitiesList.innerHTML = "";
    keywordCloud.innerHTML = '<span class="empty-state-text">Waiting for custom typed answers...</span>';
    treePath.innerHTML = '<span class="empty-state-text">Start replying to view the classifier path splits.</span>';
    scorecardOverlay.classList.add("hidden");
    turnBadge.innerText = "Step: 1 / 6";
    
    // Initialise 8 flat progress bars in the dashboard
    renderFlatProbabilities();
    
    // Call server to trigger the first profiling question
    await submitMessage("");
}

// Draw initial 12.5% bars
function renderFlatProbabilities() {
    probabilitiesList.innerHTML = "";
    const classes = Object.keys(categoryClasses);
    classes.forEach(cls => {
        const key = categoryClasses[cls];
        const row = document.createElement("div");
        row.className = "probability-row";
        row.innerHTML = `
            <div class="prob-labels">
                <span class="prob-name">${cls}</span>
                <span class="prob-val">12.5%</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar-fill cat-bg-${key}" style="width: 12.5%"></div>
            </div>
        `;
        probabilitiesList.appendChild(row);
    });
}

// Call API backend
async function submitMessage(messageText) {
    toggleLoading(true);
    
    // Update state payload
    chatState.latest_message = messageText;
    
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(chatState)
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Sync local state with backend response state
        chatState = data.state;
        
        // Handle message rendering
        if (messageText.trim()) {
            appendMessage("user", messageText);
        }
        
        // Brief timeout for assistant message to feel natural
        setTimeout(() => {
            toggleLoading(false);
            
            if (data.type === "question") {
                appendMessage("assistant", data.question);
                renderOptions(data.options);
                updateXaiDashboard(data.ml_data);
                updateProgressIndicators();
            } else if (data.type === "final_report") {
                appendMessage("assistant", "Excellent! I have fully analyzed your aptitudes. Opening your comprehensive career report...");
                updateXaiDashboard(data.ml_data);
                
                // Delay opening scorecard so they can see the final message bubble
                setTimeout(() => {
                    showScorecard(data.report);
                }, 1500);
            }
        }, 600);
        
    } catch (error) {
        toggleLoading(false);
        console.error("Submission failed:", error);
        appendMessage("assistant", "⚠️ Connection error. Please make sure the backend server is running and try again.");
    }
}

// Render message bubbles
function appendMessage(role, text) {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-message ${role === "user" ? "user-msg" : "assistant-msg"}`;
    
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.innerText = text;
    
    wrapper.appendChild(bubble);
    chatContainer.appendChild(wrapper);
    
    // Scroll chat to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Render option buttons
function renderOptions(options) {
    optionsContainer.innerHTML = "";
    
    if (!options || options.length === 0) {
        return;
    }
    
    options.forEach(opt => {
        const button = document.createElement("button");
        button.className = "option-chip";
        button.innerText = opt.text;
        button.addEventListener("click", () => {
            // Options send direct pre-baked string
            submitMessage(opt.value);
        });
        optionsContainer.appendChild(button);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Explanable ML Visualizer Panel
// ─────────────────────────────────────────────────────────────────────────────

function updateXaiDashboard(mlData) {
    if (!mlData) return;
    
    // 1. Update Probabilities (Random Forest)
    if (mlData.probabilities) {
        probabilitiesList.innerHTML = "";
        Object.keys(mlData.probabilities).forEach(cls => {
            const val = mlData.probabilities[cls];
            const key = categoryClasses[cls] || "cs";
            
            const row = document.createElement("div");
            row.className = "probability-row";
            row.innerHTML = `
                <div class="prob-labels">
                    <span class="prob-name">${cls}</span>
                    <span class="prob-val">${val.toFixed(1)}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill cat-bg-${key}" style="width: ${val}%"></div>
                </div>
            `;
            probabilitiesList.appendChild(row);
        });
    }
    
    // 2. Update Highlights Cloud (TF-IDF Features)
    if (mlData.highlights && mlData.highlights.length > 0) {
        keywordCloud.innerHTML = "";
        
        // Remove duplicates
        const seenWords = new Set();
        const uniqueHighlights = mlData.highlights.filter(h => {
            if (seenWords.has(h.word)) return false;
            seenWords.add(h.word);
            return true;
        });
        
        uniqueHighlights.forEach(h => {
            const key = categoryClasses[h.category] || "cs";
            const tag = document.createElement("span");
            tag.className = `highlight-chip cat-border-${key}`;
            
            // Map category names to brief abbreviations
            let abbrev = h.category;
            if (abbrev.includes("Computer Science")) abbrev = "CSE";
            else if (abbrev.includes("Mechanical")) abbrev = "ME";
            else if (abbrev.includes("Electrical")) abbrev = "EE";
            else if (abbrev.includes("Civil")) abbrev = "CE";
            else if (abbrev.includes("Biology")) abbrev = "Bio";
            else if (abbrev.includes("Chemistry")) abbrev = "Chem";
            else if (abbrev.includes("Physics")) abbrev = "Phys";
            else if (abbrev.includes("Management")) abbrev = "Biz";
            
            tag.innerText = `${h.word} (${abbrev})`;
            keywordCloud.appendChild(tag);
        });
    } else {
        keywordCloud.innerHTML = '<span class="empty-state-text">Waiting for custom typed answers...</span>';
    }
    
    // 3. Update Decision Tree Splits
    if (mlData.decision_path && mlData.decision_path.length > 0) {
        treePath.innerHTML = "";
        mlData.decision_path.forEach(step => {
            const isLeaf = step.outcome === "leaf";
            const node = document.createElement("div");
            node.className = "tree-node";
            
            let colorClass = "outcome-left";
            if (step.outcome === "right") colorClass = "outcome-right";
            else if (isLeaf) colorClass = "outcome-leaf";
            
            node.innerHTML = `
                <div class="node-icon ${isLeaf ? 'leaf' : ''}">
                    ${isLeaf ? '✓' : '⚙'}
                </div>
                <div class="node-details">
                    <span class="node-desc">${step.description}</span>
                    <span class="node-split ${colorClass}">${step.decision}</span>
                </div>
            `;
            treePath.appendChild(node);
        });
    } else {
        treePath.innerHTML = '<span class="empty-state-text">Start replying to view the classifier path splits.</span>';
    }
}

// ── Progress Helpers ──
function updateProgressIndicators() {
    // Count user's submitted replies in conversation history to calculate current step
    const userAnswersCount = chatState.history.filter(h => h.role === "user").length;
    let stepNum = userAnswersCount + 1;
    
    // Total steps is dynamically adjusted (6 if 12th profiling is taken, 5 if 10th/Other is taken)
    let totalSteps = 6;
    if (chatState.education_level && !chatState.education_level.includes("12")) {
        totalSteps = 5;
    }
    
    if (stepNum > totalSteps) stepNum = totalSteps;
    
    turnBadge.innerText = `Step: ${stepNum} / ${totalSteps}`;
}

function toggleLoading(isLoading) {
    if (isLoading) {
        loadingIndicator.classList.remove("hidden");
        sendBtn.disabled = true;
        userInput.disabled = true;
        // Disable option chips
        document.querySelectorAll(".option-chip").forEach(btn => btn.disabled = true);
    } else {
        loadingIndicator.classList.add("hidden");
        sendBtn.disabled = false;
        userInput.disabled = false;
        userInput.value = "";
        userInput.focus();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scorecard / Final Report Card
// ─────────────────────────────────────────────────────────────────────────────

function showScorecard(report) {
    primaryRecTitle.innerText = report.primary_recommendation;
    secondaryRecTitle.innerText = report.secondary_alternative;
    
    justificationList.innerHTML = "";
    report.justification_analysis.forEach(bullet => {
        const li = document.createElement("li");
        li.innerHTML = bullet;
        justificationList.appendChild(li);
    });
    
    scorecardOverlay.classList.remove("hidden");
}

// ─────────────────────────────────────────────────────────────────────────────
// Event Bindings
// ─────────────────────────────────────────────────────────────────────────────

// Send button click
sendBtn.addEventListener("click", () => {
    const txt = userInput.value.trim();
    if (txt) {
        submitMessage(txt);
    }
});

// Textarea Enter key triggers submission
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const txt = userInput.value.trim();
        if (txt) {
            submitMessage(txt);
        }
    }
});

// Restart assessment
restartBtn.addEventListener("click", () => {
    initChat();
});
