"""
CareerSathi: Cognitive Career & Admissions Advisory Suite (Navrachana University)
CSE708 NLP Lab Practical — End-to-End Dialogue State Orchestrator
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

# Import NLP and RIASEC engines
from riasec import (
    compute_riasec_scores,
    update_belief_state,
    calculate_shannon_entropy,
    map_riasec_to_indian_streams,
    get_career_recommendations,
    apply_negative_feedback,
    RIASEC_CODES
)
from nlp_engine import (
    preprocess_linguistic_text,
    extract_linguistic_attributions,
    trace_decision_tree_path,
    is_interrogative,
    extract_representative_quotes
)
from generator import (
    generate_dynamic_question,
    get_opening_scenario,
    generate_personalized_justifications,
    answer_student_question,
    generate_recalibration_inquiry,
    query_local_gemma
)

app = FastAPI(title="CareerSathi: Cognitive Admissions & Career Advisory Suite | Navrachana University")

# ─────────────────────────────────────────────────────────────────────────────
# Request and Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatState(BaseModel):
    turn_count: int = 0
    phase: str = "scenario"                      # "scenario" | "completed" | "recalibrating"
    level: str = "after_12th"                    # "after_10th" | "after_12th"
    education_level: Optional[str] = "12th Standard"
    stream: Optional[str] = "General"
    history: List[Dict[str, str]] = []           # [{"role": "user"|"assistant", "content": "..."}]
    riasec_state: Dict[str, float] = {}          # {"R": 16.7, "I": 16.7, ...}
    latest_message: str = ""
    action: Optional[str] = "chat"               # "start" | "chat" | "recalibrate" | "force_complete"
    excluded_ids: List[str] = []                 # Blacklisted career IDs from negative feedback
    rejected_degree: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Core Chat API
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(state: ChatState):
    new_history = list(state.history)
    current_riasec = dict(state.riasec_state) if state.riasec_state else {k: 16.67 for k in RIASEC_CODES.keys()}
    user_input = state.latest_message.strip()
    is_10th = (state.level == "after_10th")
    
    # ── ACTION 1: Initial Start from Landing Page ────────────────────────────
    if state.action == "start" or (state.turn_count == 0 and not user_input):
        state.turn_count = 1
        state.phase = "scenario"
        state.education_level = "10th Standard" if is_10th else "12th Standard"
        state.stream = "Post-10th Streams" if is_10th else "Undergraduate Degrees"
        
        opening_q = get_opening_scenario(state.stream, state.education_level)
        new_history.append({"role": "assistant", "content": opening_q})
        
        state.history = new_history
        state.riasec_state = current_riasec
        
        # Initial suggested chips to get student started smoothly
        init_chips = [
            "Building software or tech tools",
            "Science experiments in a lab",
            "Creative writing, arts or design",
            "Business, finance or marketing"
        ] if is_10th else [
            "Coding and software engineering",
            "Medical, biology and healthcare",
            "Business analytics and finance",
            "UI/UX product design and arts"
        ]
        
        return {
            "type": "question",
            "question": opening_q,
            "options": init_chips,
            "can_complete": False,
            "stage": 1,
            "stage_name": "Interests",
            "state": state.model_dump(),
            "telemetry": {
                "riasec": current_riasec,
                "entropy": 1.0,
                "dialogue_mode": "EXPLORING (Interests)",
                "indian_streams": map_riasec_to_indian_streams(current_riasec),
                "highlights": [],
                "decision_path": []
            }
        }

    # Record user message in history
    if user_input:
        new_history.append({"role": "user", "content": user_input})
        
    # ── ACTION 2: Recalibration / "I Didn't Like This Suggestion" ─────────────
    if state.action == "recalibrate" or "don't like" in user_input.lower() or "didnt like" in user_input.lower() or "not interested" in user_input.lower():
        state.phase = "recalibrating"
        rejected = state.rejected_degree or "the previous recommendation"
        
        # Apply negative evidence to update belief state
        updated_riasec, new_exclusions = apply_negative_feedback(current_riasec, user_input, rejected)
        current_riasec = updated_riasec
        state.riasec_state = current_riasec
        state.excluded_ids = list(set(state.excluded_ids + new_exclusions))
        
        recal_inquiry = generate_recalibration_inquiry(rejected, user_input)
        new_history.append({"role": "assistant", "content": recal_inquiry})
        
        state.history = new_history
        state.turn_count += 1
        
        # Compute fresh recommendations reflecting the penalty
        recal_recs = get_career_recommendations(
            scores=current_riasec,
            stream_filter=state.stream,
            excluded_ids=state.excluded_ids,
            is_10th=is_10th
        )
        
        return {
            "type": "recalibration",
            "question": recal_inquiry,
            "options": ["Something more creative & design-led", "More pure technology & coding", "Healthcare & human sciences"],
            "can_complete": True,
            "stage": 4,
            "stage_name": "Recalibrating",
            "report": build_dossier_payload(recal_recs, current_riasec, new_history, is_10th),
            "state": state.model_dump(),
            "telemetry": {
                "riasec": current_riasec,
                "entropy": 0.5,
                "dialogue_mode": "RECALIBRATING (Constraint Active)",
                "indian_streams": map_riasec_to_indian_streams(current_riasec),
                "highlights": extract_linguistic_attributions(user_input),
                "decision_path": trace_decision_tree_path(user_input)
            }
        }

    # Extract all cumulative user utterances
    user_utterances = [msg["content"] for msg in new_history if msg["role"] == "user"]
    cumulative_text = " ".join(user_utterances)
    
    # ── SESSIONS 7-10: Linguistic Preprocessing & Morpheme Extraction ────────
    prep_results = preprocess_linguistic_text(cumulative_text)
    current_turn_prep = preprocess_linguistic_text(user_input)
    is_uncertain = current_turn_prep.get("is_uncertain", False)
    user_asked_question = is_interrogative(user_input)
    
    # ── SESSIONS 26-27: Dense S-BERT Embedding & Bayesian Update ────────────
    turn_riasec = compute_riasec_scores(cumulative_text)
    updated_riasec = update_belief_state(current_riasec, turn_riasec, alpha=0.55)
    state.riasec_state = updated_riasec
    
    # ── INFORMATION THEORY: Shannon Entropy & Convergence ────────────────────
    entropy_val, is_ambiguous = calculate_shannon_entropy(updated_riasec)
    stream_affinities = map_riasec_to_indian_streams(updated_riasec)
    highlights = extract_linguistic_attributions(cumulative_text)
    decision_path = trace_decision_tree_path(cumulative_text)
    
    # Determine Journey Rail Stage
    # 1: Interests (turns 1-2), 2: Thinking Style (turns 3-4), 3: Strengths (turns 5-6), 4: Recommendation (turn 6+)
    if state.turn_count <= 2:
        stage_num = 1
        stage_name = "Interests"
    elif state.turn_count <= 4:
        stage_num = 2
        stage_name = "Thinking style"
    elif state.turn_count <= 6:
        stage_num = 3
        stage_name = "Strengths"
    else:
        stage_num = 4
        stage_name = "Deep Dive"
        
    scenario_turns_done = len(user_utterances)
    can_complete = scenario_turns_done >= 5 or state.turn_count >= 5
    
    # ── TERMINATION CHECK: Turn 15 Hard Cap or Explicit Completion ───────────
    hit_max_cap = state.turn_count >= 15
    wants_completion = state.action == "force_complete" or "show dossier" in user_input.lower() or "generate report" in user_input.lower()
    
    if (can_complete and wants_completion) or hit_max_cap:
        state.phase = "completed"
        
        # If user asked a question at Turn 15, answer it authoritatively first!
        answer_prefix = ""
        if user_asked_question:
            direct_ans = answer_student_question(user_input, state.stream, state.education_level, new_history)
            answer_prefix = f"{direct_ans}\n\nFactoring this in with our entire conversation, here is your finalized Navrachana University Career & Admissions Dossier!"
            new_history.append({"role": "assistant", "content": answer_prefix})
            
        recs = get_career_recommendations(
            scores=updated_riasec,
            stream_filter=state.stream,
            excluded_ids=state.excluded_ids,
            is_10th=is_10th
        )
        report = build_dossier_payload(recs, updated_riasec, new_history, is_10th)
        state.history = new_history
        
        return {
            "type": "final_report",
            "report": report,
            "can_complete": True,
            "stage": 4,
            "stage_name": "Your recommendation",
            "state": state.model_dump(),
            "telemetry": {
                "riasec": updated_riasec,
                "entropy": entropy_val,
                "dialogue_mode": "CONVERGED",
                "indian_streams": stream_affinities,
                "highlights": highlights,
                "decision_path": decision_path,
                "can_complete": True
            }
        }
        
    # ── GENERATION: Generate Next Dynamic Conversational Turn ────────────────
    state.turn_count += 1
    
    # If user asked a question in earlier turns, answer it and then ask next scenario
    if user_asked_question:
        direct_ans = answer_student_question(user_input, state.stream, state.education_level, new_history)
        next_q = generate_dynamic_question(
            turn_num=state.turn_count,
            user_reply=user_input,
            riasec_scores=updated_riasec,
            is_ambiguous=is_ambiguous,
            education_level=state.education_level,
            stream=state.stream,
            history=new_history,
            is_uncertain=is_uncertain
        )
        full_response = f"{direct_ans} {next_q}"
    else:
        full_response = generate_dynamic_question(
            turn_num=state.turn_count,
            user_reply=user_input,
            riasec_scores=updated_riasec,
            is_ambiguous=is_ambiguous,
            education_level=state.education_level,
            stream=state.stream,
            history=new_history,
            is_uncertain=is_uncertain
        )
        
    new_history.append({"role": "assistant", "content": full_response})
    state.history = new_history
    
    # Compile dynamic report preview if student is already in turn 5+ (for split-screen artifact)
    dossier_preview = None
    if can_complete:
        recs = get_career_recommendations(
            scores=updated_riasec,
            stream_filter=state.stream,
            excluded_ids=state.excluded_ids,
            is_10th=is_10th
        )
        dossier_preview = build_dossier_payload(recs, updated_riasec, new_history, is_10th)
        
    # Contextual quick-reply chips based on stage
    quick_chips = []
    if stage_num == 2:
        quick_chips = ["I make a list and divide the tasks", "I jump straight into fixing the hardest problem", "I focus on keeping the team calm"]
    elif stage_num == 3:
        quick_chips = ["Deep solo technical focus", "Collaborative team brainstorming", "Hands-on building & testing"]

    return {
        "type": "question",
        "question": full_response,
        "options": quick_chips,
        "can_complete": can_complete,
        "stage": stage_num,
        "stage_name": stage_name,
        "report": dossier_preview,
        "state": state.model_dump(),
        "telemetry": {
            "riasec": updated_riasec,
            "entropy": entropy_val,
            "dialogue_mode": f"STAGE {stage_num}: {stage_name.upper()}",
            "indian_streams": stream_affinities,
            "highlights": highlights,
            "decision_path": decision_path,
            "can_complete": can_complete
        }
    }


def build_dossier_payload(recs: Dict, riasec_scores: Dict[str, float], history: List[Dict[str, str]], is_10th: bool) -> Dict:
    """Compiles the rich document artifact payload."""
    primary_deg = recs["primary"]
    secondary_deg = recs["secondary"]
    top_traits = recs["top_riasec"]
    
    user_utterances = [msg["content"] for msg in history if msg["role"] == "user"]
    
    # Extract evidence quotes ("You said: ...")
    evidence_quotes = extract_representative_quotes(user_utterances, top_traits)
    
    # Personalized justifications
    justifications = generate_personalized_justifications(
        primary_deg, secondary_deg, top_traits, user_utterances
    )
    
    return {
        "is_10th": is_10th,
        "primary_recommendation": primary_deg["degree"],
        "primary_stream": primary_deg["stream"],
        "primary_fit": primary_deg["fit_score"],
        "primary_id": primary_deg.get("id", ""),
        "secondary_recommendation": secondary_deg["degree"],
        "secondary_stream": secondary_deg["stream"],
        "secondary_fit": secondary_deg["fit_score"],
        "alternatives": recs.get("alternatives", []),
        "target_entrance_exams": primary_deg.get("exams", ["CUET UG"]),
        "career_profile": primary_deg.get("profile", ""),
        "top_riasec_traits": top_traits,
        "all_riasec_scores": riasec_scores,
        "evidence_quotes": evidence_quotes,
        "justifications": justifications
    }


# ─────────────────────────────────────────────────────────────────────────────
# Static Assets Mounting
# ─────────────────────────────────────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"  [Warning] Frontend directory '{frontend_dir}' not found.")
