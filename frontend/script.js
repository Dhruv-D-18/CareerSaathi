// ─────────────────────────────────────────────────────────────────────────────
// CareerSaathi: Cognitive Career & Admissions Advisory Suite (Navrachana University)
// Client State Machine, Claude-Style Artifact Split-View & Live NLP Telemetry
// ─────────────────────────────────────────────────────────────────────────────

let sessionState = {
    turn_count: 0,
    phase: "scenario",
    level: "after_12th",
    education_level: "12th Standard",
    stream: "Undergraduate Degrees",
    history: [],
    riasec_state: { "R": 16.7, "I": 16.7, "A": 16.7, "S": 16.7, "E": 16.7, "C": 16.7 },
    latest_message: "",
    action: "start",
    excluded_ids: [],
    rejected_degree: null
};

let latestDossierReport = null;
let latestTelemetry = null;
let currentStageNum = 1;
let isArtifactSplitOpen = false;
let activeArtifactTabName = "student";
let lastRenderedRecommendationTitle = "";

// 6 Holland RIASEC Metadata
const RIASEC_META = {
    "R": { name: "Realistic", label: "Realistic (Hands-on)", color: "#fb923c" },
    "I": { name: "Investigative", label: "Investigative (Analytical)", color: "#0284c7" },
    "A": { name: "Artistic", label: "Artistic (Creative)", color: "#9333ea" },
    "S": { name: "Social", label: "Social (Empathetic)", color: "#059669" },
    "E": { name: "Enterprising", label: "Enterprising (Leadership)", color: "#d97706" },
    "C": { name: "Conventional", label: "Conventional (Organized)", color: "#e11d48" }
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. Navigation & Screen Switching
// ─────────────────────────────────────────────────────────────────────────────

function showScreen(screenId) {
    document.querySelectorAll(".screen-view").forEach(s => s.classList.remove("active"));
    const target = document.getElementById(screenId);
    if (target) target.classList.add("active");
}

function returnToLanding() {
    showScreen("landing-screen");
}

function resetAssessment() {
    sessionState = {
        turn_count: 0,
        phase: "scenario",
        level: sessionState.level || "after_12th",
        education_level: sessionState.level === "after_10th" ? "10th Standard" : "12th Standard",
        stream: sessionState.level === "after_10th" ? "Post-10th Streams" : "Undergraduate Degrees",
        history: [],
        riasec_state: { "R": 16.7, "I": 16.7, "A": 16.7, "S": 16.7, "E": 16.7, "C": 16.7 },
        latest_message: "",
        action: "start",
        excluded_ids: [],
        rejected_degree: null
    };
    latestDossierReport = null;
    lastRenderedRecommendationTitle = "";
    currentStageNum = 1;
    toggleArtifactPane(false);

    document.getElementById("chat-messages-container").innerHTML = "";
    document.getElementById("quick-chips-wrapper").innerHTML = "";
    document.getElementById("open-dossier-btn").classList.add("hidden");
    const dockedBar = document.getElementById("docked-dossier-bar");
    if (dockedBar) dockedBar.classList.add("hidden");

    startAssessment(sessionState.level);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Initial Assessment Trigger from Landing Page
// ─────────────────────────────────────────────────────────────────────────────

async function startAssessment(level) {
    sessionState.level = level;
    sessionState.turn_count = 0;
    sessionState.education_level = level === "after_10th" ? "10th Standard" : "12th Standard";
    sessionState.stream = level === "after_10th" ? "Post-10th Streams" : "Undergraduate Degrees";
    sessionState.action = "start";
    sessionState.history = [];
    sessionState.excluded_ids = [];
    sessionState.rejected_degree = null;
    latestDossierReport = null;

    // Update Top Navigation Tag
    const tagEl = document.getElementById("current-pathway-tag");
    if (tagEl) {
        tagEl.innerText = level === "after_10th" ? "After 10th: Choosing a Stream" : "After 12th: Choosing a Degree";
    }

    // Switch to Workspace Screen
    showScreen("workspace-screen");
    toggleArtifactPane(false);

    // Clear chat
    document.getElementById("chat-messages-container").innerHTML = "";
    document.getElementById("quick-chips-wrapper").innerHTML = "";
    document.getElementById("open-dossier-btn").classList.add("hidden");
    const initDocked = document.getElementById("docked-dossier-bar");
    if (initDocked) initDocked.classList.add("hidden");

    showLoading(true);
    try {
        const payload = {
            turn_count: 0,
            phase: "scenario",
            level: level,
            education_level: sessionState.education_level,
            stream: sessionState.stream,
            history: [],
            riasec_state: sessionState.riasec_state,
            latest_message: "",
            action: "start",
            excluded_ids: []
        };

        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Server error initializing session.");
        const data = await res.json();
        handleServerResponse(data);
    } catch (err) {
        console.error(err);
        appendBotMessage("Welcome to CareerSaathi! To help find what stream or degree fits how you think: between building software/tech, conducting science lab experiments, creative writing/design, or business leadership, which one feels most natural to you?");
    } finally {
        showLoading(false);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. User Message Submission & Server Communication
// ─────────────────────────────────────────────────────────────────────────────

function handleTextareaKeydown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleUserSubmit(event);
    }
}

async function handleUserSubmit(event) {
    if (event) event.preventDefault();
    const textarea = document.getElementById("chat-textarea");
    const text = textarea.value.trim();
    if (!text) return;

    // Append User Message Bubble
    appendUserMessage(text);
    textarea.value = "";
    adjustTextareaHeight(textarea);

    // Clear suggestion chips
    document.getElementById("quick-chips-wrapper").innerHTML = "";

    showLoading(true);

    try {
        const actionToSend = (sessionState.action === "recalibrate") ? "recalibrate" : "chat";
        const payload = {
            turn_count: sessionState.turn_count,
            phase: sessionState.phase,
            level: sessionState.level,
            education_level: sessionState.education_level,
            stream: sessionState.stream,
            history: sessionState.history,
            riasec_state: sessionState.riasec_state,
            latest_message: text,
            action: actionToSend,
            excluded_ids: sessionState.excluded_ids || [],
            rejected_degree: sessionState.rejected_degree
        };

        // Reset action back to normal chat after sending
        sessionState.action = "chat";

        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("API error occurred.");
        const data = await res.json();
        handleServerResponse(data);
    } catch (err) {
        console.error(err);
        appendBotMessage("I am processing your insights. Could you tell me a little more about whether you prefer working deeply with computational logic or collaborating on creative designs?");
    } finally {
        showLoading(false);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Server Response Handler (Questions, Recalibration & Dossier)
// ─────────────────────────────────────────────────────────────────────────────

function handleServerResponse(data) {
    if (data.state) {
        sessionState = data.state;
        sessionState.action = "chat"; // Always ensure subsequent messages default to chat mode
    }

    if (data.telemetry) {
        latestTelemetry = data.telemetry;
        updateMiniTelemetry(data.telemetry);
        updateExaminerTelemetry(data.telemetry);
    }

    // Update Journey Rail Stage
    if (data.stage) {
        updateJourneyRail(data.stage, data.stage_name);
    }

    // Append Bot Message
    if (data.question) {
        appendBotMessage(data.question);
    }

    // Render Quick Option Chips
    if (data.options && data.options.length > 0) {
        renderQuickChips(data.options);
    }

    // Store latest report if provided & detect recommendation changes
    if (data.report) {
        const newTitle = data.report.primary_recommendation || data.report.primary_degree_name;
        const previousTitle = lastRenderedRecommendationTitle;
        const isUpdate = (previousTitle !== "" && previousTitle !== newTitle) || data.is_updated_recommendation;
        lastRenderedRecommendationTitle = newTitle;

        latestDossierReport = data.report;
        renderDossierArtifact(data.report);

        // Update Docked Bar UI dynamically so student knows a new suggestion was made
        const dockedBar = document.getElementById("docked-dossier-bar");
        const dockedTitle = document.getElementById("docked-title");
        const dockedSubtitle = document.getElementById("docked-subtitle");
        const dockedIcon = document.getElementById("docked-bar-icon");

        if (dockedBar) {
            dockedBar.classList.remove("hidden");
            if (isUpdate && previousTitle) {
                dockedBar.classList.add("pulse-highlight");
                if (dockedIcon) dockedIcon.innerText = "✨";
                if (dockedTitle) dockedTitle.innerHTML = `<span class="badge-updated">UPDATED MATCH</span> <strong>${escapeHtml(newTitle)}</strong>`;
                if (dockedSubtitle) dockedSubtitle.innerText = "CareerSaathi has updated your Admissions Dossier • Click to review";
                
                // Append in-chat notification card if chat pane doesn't have split open
                appendInChatUpdateCard(newTitle, data.report.primary_fit);
            } else {
                if (dockedIcon) dockedIcon.innerText = "📄";
                if (dockedTitle) dockedTitle.innerHTML = `<strong>Admissions Dossier Ready</strong>: ${escapeHtml(newTitle)}`;
                if (dockedSubtitle) dockedSubtitle.innerText = "Click to view side-by-side recommendation & RIASEC profile";
            }
        }
    }

    // Docked Dossier Banner directly above input box + Navbar Pill
    const dockedBar = document.getElementById("docked-dossier-bar");
    const openDossierBtn = document.getElementById("open-dossier-btn");
    if (data.can_complete) {
        if (dockedBar) dockedBar.classList.remove("hidden");
        if (openDossierBtn) openDossierBtn.classList.remove("hidden");
    }

    // Check if assessment completed or dossier should split-open
    if (data.type === "final_report") {
        updateJourneyRail(4, "Your recommendation");
        if (data.report) {
            renderDossierArtifact(data.report);
            toggleArtifactPane(true);
        }
    }

    // If split pane is currently open, re-render to reflect live updates
    if (isArtifactSplitOpen && latestDossierReport) {
        renderDossierArtifact(latestDossierReport);
    }
}

function appendInChatUpdateCard(title, fitScore) {
    const container = document.getElementById("chat-messages-container");
    const cardDiv = document.createElement("div");
    cardDiv.className = "chat-dossier-update-card";
    cardDiv.innerHTML = `
        <div class="update-card-left">
            <span class="update-card-badge">✨ NEW RECOMMENDATION READY</span>
            <h4 class="update-card-title">${escapeHtml(title)}</h4>
            <p class="update-card-desc">CareerSaathi has updated your admissions dossier (${fitScore}% match). Click to open the side pane.</p>
        </div>
        <button type="button" class="btn btn-crimson btn-sm update-card-btn" onclick="toggleArtifactPane(true)">
            View Dossier &nearr;
        </button>
    `;
    container.appendChild(cardDiv);
    scrollChatToBottom();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Chat Rendering Helpers (Bubbles & In-Chat Artifact Card)
// ─────────────────────────────────────────────────────────────────────────────

function appendUserMessage(text) {
    const container = document.getElementById("chat-messages-container");
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-msg user";
    msgDiv.id = `msg-turn-${sessionState.turn_count}`;
    msgDiv.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
    container.appendChild(msgDiv);
    scrollChatToBottom();
}

function appendBotMessage(text) {
    const container = document.getElementById("chat-messages-container");
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-msg bot";
    msgDiv.innerHTML = `<div class="msg-bubble">${formatMarkdown(text)}</div>`;
    container.appendChild(msgDiv);
    scrollChatToBottom();
}

function injectArtifactCardInChat() {
    const container = document.getElementById("chat-messages-container");
    const cardDiv = document.createElement("div");
    cardDiv.className = "chat-artifact-card";
    cardDiv.onclick = () => toggleArtifactPane(true);
    cardDiv.innerHTML = `
        <div class="artifact-card-left">
            <span class="artifact-card-icon">📄</span>
            <div>
                <div class="artifact-card-title">CareerSaathi Admissions Dossier Generated</div>
                <div class="artifact-card-subtitle">Click to open Side-by-Side Artifact View</div>
            </div>
        </div>
        <span class="artifact-card-action">View Dossier ↗</span>
    `;
    container.appendChild(cardDiv);
    scrollChatToBottom();
}

function renderQuickChips(options) {
    const wrapper = document.getElementById("quick-chips-wrapper");
    wrapper.innerHTML = "";
    options.forEach(opt => {
        const btn = document.createElement("button");
        btn.className = "quick-chip-btn";
        btn.innerText = opt;
        btn.onclick = () => {
            document.getElementById("chat-textarea").value = opt;
            handleUserSubmit();
        };
        wrapper.appendChild(btn);
    });
}

function showLoading(show) {
    const loader = document.getElementById("chat-loading");
    if (loader) {
        if (show) loader.classList.remove("hidden");
        else loader.classList.add("hidden");
    }
    scrollChatToBottom();
}

function scrollChatToBottom() {
    const container = document.getElementById("chat-messages-container");
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

function adjustTextareaHeight(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Journey Rail Management (Interests -> Thinking Style -> Strengths)
// ─────────────────────────────────────────────────────────────────────────────

function updateJourneyRail(stageNum, stageName) {
    currentStageNum = stageNum;
    
    // Update Step items with checkmarks and active states
    for (let i = 1; i <= 4; i++) {
        const stepEl = document.getElementById(`step-${i}`);
        if (!stepEl) continue;
        const numEl = stepEl.querySelector(".step-num");
        
        stepEl.classList.remove("active", "completed");
        if (i < stageNum) {
            stepEl.classList.add("completed");
            if (numEl) numEl.innerHTML = "✓";
        } else if (i === stageNum) {
            stepEl.classList.add("active");
            if (numEl) numEl.innerHTML = `${i}`;
        } else {
            if (numEl) numEl.innerHTML = `${i}`;
        }
    }

    // Update Stage Title in Chat Area
    const titleEl = document.getElementById("stage-title-text");
    const guideEl = document.getElementById("stage-guide-text");
    if (titleEl && stageName) {
        titleEl.innerText = stageName;
    }
    if (guideEl) {
        if (stageNum === 1) guideEl.innerText = "Tell us what subjects and real-world projects excite you.";
        else if (stageNum === 2) guideEl.innerText = "Tell us how you naturally solve problems and handle chaos.";
        else if (stageNum === 3) guideEl.innerText = "Exploring your ideal work environment and collaboration style.";
        else if (stageNum >= 4) guideEl.innerText = "Your personalized Navrachana University admissions recommendation.";
    }
}

function updateMiniTelemetry(telemetry) {
    if (!telemetry) return;
    
    // Entropy Tag
    const entTag = document.getElementById("rail-entropy-tag");
    if (entTag && telemetry.entropy !== undefined) {
        entTag.innerText = `H = ${telemetry.entropy.toFixed(3)}`;
    }

    // Stream Affinity Bars
    if (telemetry.indian_streams) {
        const container = document.getElementById("stream-mini-bars");
        if (container) {
            container.innerHTML = "";
            for (const [stream, pct] of Object.entries(telemetry.indian_streams)) {
                let colorClass = "pcm-bar";
                if (stream.includes("PCB")) colorClass = "pcb-bar";
                else if (stream.includes("Commerce")) colorClass = "comm-bar";
                else if (stream.includes("Arts") || stream.includes("Liberal")) colorClass = "arts-bar";

                const row = document.createElement("div");
                row.className = "mini-stream-row";
                row.innerHTML = `
                    <span class="mini-stream-label" title="${stream}">${stream}</span>
                    <div class="mini-bar-track"><div class="mini-bar-fill ${colorClass}" style="width: ${pct}%"></div></div>
                    <span class="mini-stream-val">${pct}%</span>
                `;
                container.appendChild(row);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Claude-Style Artifact Split-Pane Controls
// ─────────────────────────────────────────────────────────────────────────────

function toggleArtifactPane(open) {
    const pane = document.getElementById("artifact-split-pane");
    if (!pane) return;

    if (open === undefined) {
        isArtifactSplitOpen = !isArtifactSplitOpen;
    } else {
        isArtifactSplitOpen = open;
    }

    if (isArtifactSplitOpen) {
        pane.classList.add("open");
        if (latestDossierReport) {
            renderDossierArtifact(latestDossierReport);
        }
        if (latestTelemetry) {
            updateExaminerTelemetry(latestTelemetry);
        }
    } else {
        pane.classList.remove("open");
    }
}

function switchArtifactTab(tabName) {
    activeArtifactTabName = tabName;
    document.querySelectorAll(".artifact-tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".artifact-tab-content").forEach(c => c.classList.remove("active"));

    if (tabName === "student") {
        document.getElementById("tab-student-dossier").classList.add("active");
        document.getElementById("content-student-dossier").classList.add("active");
    } else {
        document.getElementById("tab-examiner-xai").classList.add("active");
        document.getElementById("content-examiner-xai").classList.add("active");
        if (latestTelemetry) {
            drawRadarChart("xai-radar-svg", latestTelemetry.riasec || sessionState.riasec_state);
        }
    }
}

function toggleExaminerMode() {
    toggleArtifactPane(true);
    switchArtifactTab("examiner");
}

function focusChatForReview() {
    // Scrolls to earliest messages so evaluator can re-read the chat
    const chatContainer = document.getElementById("chat-messages-container");
    if (chatContainer) {
        chatContainer.scrollTo({ top: 0, behavior: "smooth" });
    }
}

function scrollToUserTurn(quoteText) {
    // Find the message bubble containing the quote and highlight it
    const bubbles = document.querySelectorAll(".chat-msg.user .msg-bubble");
    for (const b of bubbles) {
        if (b.innerText.toLowerCase().includes(quoteText.substring(0, 20).toLowerCase())) {
            b.scrollIntoView({ behavior: "smooth", block: "center" });
            b.style.transition = "box-shadow 0.3s ease";
            b.style.boxShadow = "0 0 0 4px #fbbf24";
            setTimeout(() => { b.style.boxShadow = ""; }, 2000);
            break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Human-in-the-Loop Recalibration Trigger ("I Didn't Like This")
// ─────────────────────────────────────────────────────────────────────────────

function triggerRecalibration() {
    const rejectedDegree = latestDossierReport ? latestDossierReport.primary_recommendation : "the recommendation";
    sessionState.action = "recalibrate";
    sessionState.rejected_degree = rejectedDegree;

    // Append counselor prompt in chat
    appendBotMessage(`Understood! What specifically didn't feel right about **${rejectedDegree}**? (For example: *too much math, prefer hands-on building, want visual arts*, or *prefer working with people*?)`);
    
    // Focus chat textarea with a pre-typed prompt helper
    const textarea = document.getElementById("chat-textarea");
    if (textarea) {
        textarea.value = `I don't like ${rejectedDegree} because `;
        textarea.focus();
    }

    // Scroll chat into view
    scrollChatToBottom();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Dossier Artifact Rendering (Claude Wireframe 3)
// ─────────────────────────────────────────────────────────────────────────────

function renderDossierArtifact(report) {
    if (!report) return;

    // 1. Hero Best Fit Card
    const heroTitle = document.getElementById("hero-degree-title");
    const heroMeta = document.getElementById("hero-degree-meta");
    const heroAlts = document.getElementById("hero-alternatives-text");

    if (heroTitle) heroTitle.innerText = report.primary_recommendation;
    if (heroMeta) {
        heroMeta.innerText = `Specialisation: ${report.career_profile || "Comprehensive Focus"}. Stream: ${report.primary_stream} (${report.primary_fit}% Fit).`;
    }
    if (heroAlts && report.alternatives) {
        heroAlts.innerText = report.alternatives.join(", ");
    }

    // 2. RIASEC Horizontal Bars
    const barsContainer = document.getElementById("riasec-horizontal-bars");
    if (barsContainer && report.all_riasec_scores) {
        barsContainer.innerHTML = "";
        for (const [code, meta] of Object.entries(RIASEC_META)) {
            const score = report.all_riasec_scores[code] || 16.7;
            const row = document.createElement("div");
            row.className = "riasec-bar-row";
            row.innerHTML = `
                <span class="riasec-bar-name" title="${meta.label}">${meta.name}</span>
                <div class="riasec-bar-track">
                    <div class="riasec-bar-fill" style="width: ${score}%; background-color: ${meta.color}"></div>
                </div>
                <span class="riasec-bar-code">${code}</span>
            `;
            barsContainer.appendChild(row);
        }
    }

    // 3. Recommended Indian Entrance Exams
    const examsContainer = document.getElementById("entrance-exams-chips");
    if (examsContainer && report.target_entrance_exams) {
        examsContainer.innerHTML = "";
        report.target_entrance_exams.forEach(ex => {
            const chip = document.createElement("span");
            chip.className = "exam-chip";
            chip.innerText = ex;
            examsContainer.appendChild(chip);
        });
    }

    const profileDesc = document.getElementById("degree-profile-desc");
    if (profileDesc && report.career_profile) {
        profileDesc.innerText = report.career_profile;
    }

    // 4. "Why this fits you" Evidence Card (Quotes exact student replies)
    const quotesContainer = document.getElementById("evidence-quotes-container");
    if (quotesContainer && report.evidence_quotes) {
        quotesContainer.innerHTML = "";
        report.evidence_quotes.forEach(item => {
            const quoteDiv = document.createElement("div");
            quoteDiv.className = "evidence-quote-item";
            quoteDiv.innerHTML = `
                <strong class="quote-label">You said:</strong> 
                <span class="quote-jump-link" onclick="scrollToUserTurn('${escapeQuotes(item.quote)}')">"${escapeHtml(item.quote)}"</span>
                <p style="margin-top: 6px; color: var(--text-secondary);">${escapeHtml(item.explanation)}</p>
            `;
            quotesContainer.appendChild(quoteDiv);
        });
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. AI Reasoning & Telemetry Rendering (XAI)
// ─────────────────────────────────────────────────────────────────────────────

function updateExaminerTelemetry(telemetry) {
    if (!telemetry) return;

    // 1. Radar Chart in Telemetry Tab
    if (telemetry.riasec) {
        drawRadarChart("xai-radar-svg", telemetry.riasec);
    }

    // 2. Shannon Entropy Math
    const entValEl = document.getElementById("xai-entropy-val");
    const entFillEl = document.getElementById("xai-entropy-fill");
    const entStatusEl = document.getElementById("xai-entropy-status");

    if (entValEl && telemetry.entropy !== undefined) {
        entValEl.innerText = telemetry.entropy.toFixed(3);
        const fillPct = Math.min(100, Math.max(0, telemetry.entropy * 100));
        if (entFillEl) entFillEl.style.width = `${fillPct}%`;
        if (entStatusEl) {
            entStatusEl.innerText = telemetry.entropy < 0.65 
                ? "Distribution converged (H < 0.65). Strong confidence." 
                : "Distribution is broad (H > 0.65). Exploring multiple disciplines.";
        }
    }

    // 3. POS Tokens & Verbs (Linguistic Parsing)
    const posWrap = document.getElementById("xai-pos-tokens");
    if (posWrap && telemetry.highlights) {
        posWrap.innerHTML = "";
        telemetry.highlights.forEach(h => {
            const pill = document.createElement("span");
            pill.className = "pos-pill";
            pill.innerHTML = `<strong>${escapeHtml(h.word)}</strong> &rarr; ${h.category}`;
            posWrap.appendChild(pill);
        });
        if (telemetry.highlights.length === 0) {
            posWrap.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Answering questions will extract POS action tokens.</span>';
        }
    }

    // 4. TF-IDF Cloud (Feature Attribution)
    const tfidfWrap = document.getElementById("xai-tfidf-cloud");
    if (tfidfWrap && telemetry.highlights) {
        tfidfWrap.innerHTML = "";
        telemetry.highlights.forEach(h => {
            const chip = document.createElement("span");
            chip.className = "tfidf-chip";
            chip.style.backgroundColor = h.color || "#0284c7";
            chip.innerText = `${h.word} (${h.code})`;
            tfidfWrap.appendChild(chip);
        });
        if (telemetry.highlights.length === 0) {
            tfidfWrap.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">TF-IDF n-grams will populate here.</span>';
        }
    }

    // 5. Decision Tree Splits (Rule Traversal)
    const treeWrap = document.getElementById("xai-tree-splits");
    if (treeWrap && telemetry.decision_path) {
        treeWrap.innerHTML = "";
        telemetry.decision_path.forEach(node => {
            const item = document.createElement("div");
            item.className = "tree-split-item";
            item.innerHTML = `<strong>Node ${node.node_id}:</strong> ${escapeHtml(node.description)} &rarr; <em>${escapeHtml(node.decision)}</em>`;
            treeWrap.appendChild(item);
        });
        if (telemetry.decision_path.length === 0) {
            treeWrap.innerHTML = '<span style="font-size: 11px; color: var(--text-muted);">Tree trace appears after 2+ turns.</span>';
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. SVG Radar Chart Generator (Holland RIASEC 6D R^6)
// ─────────────────────────────────────────────────────────────────────────────

function drawRadarChart(svgId, riasecScores) {
    const svg = document.getElementById(svgId);
    if (!svg) return;

    const codes = ["R", "I", "A", "S", "E", "C"];
    const labels = ["Realistic (R)", "Investigative (I)", "Artistic (A)", "Social (S)", "Enterprising (E)", "Conventional (C)"];
    const cx = 140;
    const cy = 115;
    const maxRadius = 80;

    let svgHtml = "";

    // Draw background concentric hexagons (25%, 50%, 75%, 100%)
    [0.25, 0.5, 0.75, 1.0].forEach(level => {
        let pts = [];
        for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i - Math.PI / 2;
            const r = maxRadius * level;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
        }
        svgHtml += `<polygon points="${pts.join(" ")}" fill="none" stroke="#e2e8f0" stroke-width="1" />`;
    });

    // Draw axis lines and labels
    for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i - Math.PI / 2;
        const x2 = cx + maxRadius * Math.cos(angle);
        const y2 = cy + maxRadius * Math.sin(angle);
        svgHtml += `<line x1="${cx}" y1="${cy}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#cbd5e1" stroke-width="1" />`;

        // Text label position
        const lx = cx + (maxRadius + 16) * Math.cos(angle);
        const ly = cy + (maxRadius + 14) * Math.sin(angle) + 4;
        const code = codes[i];
        svgHtml += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="10" font-weight="700" fill="#475569" text-anchor="middle">${code}</text>`;
    }

    // Plot data polygon
    let dataPts = [];
    for (let i = 0; i < 6; i++) {
        const code = codes[i];
        const val = riasecScores ? (riasecScores[code] || 16.7) : 16.7;
        // Normalize val: 0 to 50% maps to 0 to maxRadius
        const normalizedR = Math.min(maxRadius, (val / 45.0) * maxRadius);
        const angle = (Math.PI / 3) * i - Math.PI / 2;
        const x = cx + normalizedR * Math.cos(angle);
        const y = cy + normalizedR * Math.sin(angle);
        dataPts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }

    svgHtml += `
        <polygon points="${dataPts.join(" ")}" fill="rgba(168, 29, 36, 0.22)" stroke="#a81d24" stroke-width="2" />
    `;

    // Data points circles
    dataPts.forEach(pt => {
        const [x, y] = pt.split(",");
        svgHtml += `<circle cx="${x}" cy="${y}" r="3.5" fill="#a81d24" stroke="#ffffff" stroke-width="1.5" />`;
    });

    svg.innerHTML = svgHtml;
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Utilities & Formatting
// ─────────────────────────────────────────────────────────────────────────────

function formatMarkdown(text) {
    if (!text) return "";
    let formatted = escapeHtml(text);
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    formatted = formatted.replace(/\*(.*?)\*/g, "<em>$1</em>");
    formatted = formatted.replace(/\n\n/g, "<br><br>");
    formatted = formatted.replace(/\n/g, "<br>");
    return formatted;
}

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function escapeQuotes(str) {
    if (!str) return "";
    return str.replace(/'/g, "\\'").replace(/"/g, "&quot;");
}

function printDossier() {
    // Ensure the student dossier tab is active for printing
    switchArtifactTab('student');
    
    // Set a clean document title so the browser defaults to a nice PDF filename
    const oldTitle = document.title;
    const degName = (latestDossierReport && latestDossierReport.primary_degree_name)
        ? latestDossierReport.primary_degree_name.replace(/[^a-zA-Z0-9]/g, "_")
        : "Admissions_Dossier";
    document.title = `Navrachana_University_CareerSaathi_${degName}`;
    
    window.print();
    
    // Restore title after print dialog closes
    setTimeout(() => {
        document.title = oldTitle;
    }, 1000);
}
