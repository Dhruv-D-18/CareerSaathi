"""
CareerSaathi: Cognitive Career & Admissions Advisory Suite
End-to-End Conversational Dialogue State Orchestrator & Multi-Stage Recommender
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import re
import math
import random

# Import NLP and RIASEC engines
from riasec import (
    compute_riasec_scores,
    update_belief_state,
    calculate_shannon_entropy,
    map_riasec_to_indian_streams,
    get_career_recommendations,
    apply_negative_feedback,
    generate_cognitive_journey_report,
    generate_academic_feasibility_matrix,
    RIASEC_CODES
)
from nlp_engine import (
    preprocess_linguistic_text,
    extract_linguistic_attributions,
    trace_decision_tree_path,
    is_interrogative,
    extract_representative_quotes,
    extract_academic_scores,
    assess_conversational_signal
)
from generator import (
    generate_dynamic_question,
    get_opening_scenario,
    generate_personalized_justifications,
    answer_student_question,
    generate_recalibration_inquiry,
    query_local_gemma,
    get_time_greeting,
    get_initial_onboarding_inquiry,
    get_academic_score_acknowledgment,
    get_broad_horizon_closure_message
)

app = FastAPI(title="CareerSaathi: Cognitive Admissions & Career Advisory Suite")

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
    score_tier: Optional[str] = "passion_first"  # "high" | "competitive" | "moderate" | "foundational" | "passion_first"
    score_raw: Optional[str] = ""

# ─────────────────────────────────────────────────────────────────────────────
# Core Chat API
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(state: ChatState):
    new_history = list(state.history)
    current_riasec = dict(state.riasec_state) if state.riasec_state else {k: 16.67 for k in RIASEC_CODES.keys()}
    user_input = state.latest_message.strip()
    is_10th = (state.level == "after_10th")
    
    # ── ACTION 1: Initial Start from Landing Page (Turn 0: Time-Aware Greeting & Marks) ──
    if state.action == "start" or (state.turn_count == 0 and not user_input):
        state.turn_count = 0
        state.phase = "scenario"
        state.action = "chat"  # MUST be reset to chat so subsequent turns never repeat start!
        state.education_level = "10th Standard" if is_10th else "12th Standard"
        state.stream = "Post-10th Streams" if is_10th else "Undergraduate Degrees"
        
        opening_q = get_initial_onboarding_inquiry(state.education_level)
        new_history.append({"role": "assistant", "content": opening_q})
        
        state.history = new_history
        state.riasec_state = current_riasec
        
        # Initial suggested chips to capture marks / exam plans easily
        init_chips = [
            "85%+ in 10th Boards",
            "70% – 85% in 10th",
            "55% – 70% in 10th",
            "Under 55% in 10th",
            "Explore based on interests first"
        ] if is_10th else [
            "85%+ Boards (JEE / NEET / CUET)",
            "70% – 85% Boards",
            "55% – 70% Boards / State CET",
            "Under 55% Boards",
            "Explore based on interests first"
        ]
        
        return {
            "type": "question",
            "question": opening_q,
            "options": init_chips,
            "can_complete": False,
            "stage": 1,
            "stage_name": "Academic Background",
            "state": state.model_dump(),
            "telemetry": {
                "riasec": current_riasec,
                "entropy": 1.0,
                "dialogue_mode": "ONBOARDING (Academic Profile)",
                "indian_streams": map_riasec_to_indian_streams(current_riasec),
                "highlights": [],
                "decision_path": []
            }
        }

    # Record user message in history
    if user_input:
        new_history.append({"role": "user", "content": user_input})
        
        # Check if student mentions academic scores or competitive marks
        score_info = extract_academic_scores(user_input)
        if score_info:
            state.score_tier = score_info["tier"]
            state.score_raw = score_info["raw_score"]

    # ── ACTION 1B: Onboarding Score Validation ➔ Interests (Turn 0 -> Turn 1) ──
    if state.turn_count == 0:
        state.turn_count = 1
        state.phase = "scenario"
        
        score_ack = get_academic_score_acknowledgment(
            user_input=user_input,
            score_tier=state.score_tier,
            score_raw=state.score_raw,
            edu_level=state.education_level
        )
        opening_dilemma = get_opening_scenario(state.stream, state.education_level)
        transition_reply = f"{score_ack}\n\n{opening_dilemma}"
        
        new_history.append({"role": "assistant", "content": transition_reply})
        state.history = new_history
        
        interest_chips = [
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
            "question": transition_reply,
            "options": interest_chips,
            "can_complete": False,
            "stage": 1,
            "stage_name": "Interests",
            "state": state.model_dump(),
            "telemetry": {
                "riasec": current_riasec,
                "entropy": 1.0,
                "dialogue_mode": "STAGE 1: INTERESTS",
                "indian_streams": map_riasec_to_indian_streams(current_riasec),
                "highlights": extract_linguistic_attributions(user_input),
                "decision_path": []
            }
        }
        
    # Extract all cumulative user utterances
    user_utterances = [msg["content"] for msg in new_history if msg["role"] == "user"]
    cumulative_text = " ".join(user_utterances)

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
            is_10th=is_10th,
            user_text=cumulative_text
        )
        
        return {
            "type": "recalibration",
            "question": recal_inquiry,
            "options": ["Something more creative & design-led", "More pure technology & coding", "Healthcare & human sciences"],
            "can_complete": True,
            "stage": 4,
            "stage_name": "Recalibrating",
            "report": build_dossier_payload(recal_recs, current_riasec, new_history, is_10th, state.score_tier, state.score_raw),
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
    
    # ── STAGE 1: Linguistic Preprocessing & Morpheme Extraction ──────────────
    prep_results = preprocess_linguistic_text(cumulative_text)
    current_turn_prep = preprocess_linguistic_text(user_input)
    is_uncertain = current_turn_prep.get("is_uncertain", False)
    user_asked_question = is_interrogative(user_input)
    
    # ── STAGE 2: Dense S-BERT Embedding & Bayesian RIASEC Update ─────────────
    turn_riasec = compute_riasec_scores(cumulative_text)
    updated_riasec = update_belief_state(current_riasec, turn_riasec, alpha=0.55)
    state.riasec_state = updated_riasec
    
    # ── INFORMATION THEORY: Shannon Entropy & Signal Evaluation ──────────────
    entropy_val, is_ambiguous = calculate_shannon_entropy(updated_riasec)
    signal_eval = assess_conversational_signal(new_history, updated_riasec)
    is_converged = signal_eval["is_converged"]
    is_persistently_vague = signal_eval["is_persistently_vague"]
    
    stream_affinities = map_riasec_to_indian_streams(updated_riasec)
    highlights = extract_linguistic_attributions(cumulative_text)
    decision_path = trace_decision_tree_path(cumulative_text)
    
    # Smooth Journey Rail Stage (1: Interests, 2: Thinking Style, 3: Strengths, 4: Deep Dive / Recommendation)
    if state.turn_count <= 1:
        stage_num = 1
        stage_name = "Interests"
    elif state.turn_count <= 3:
        stage_num = 2
        stage_name = "Thinking style"
    elif state.turn_count <= 5:
        stage_num = 3
        stage_name = "Strengths"
    else:
        stage_num = 4
        stage_name = "Broad Horizon Discovery" if not is_converged else "Your recommendation"
        
    scenario_turns_done = len(user_utterances)
    hit_max_cap = state.turn_count >= 6
    wants_completion = state.action == "force_complete" or "show dossier" in user_input.lower() or "generate report" in user_input.lower() or "show me recommended" in user_input.lower()
    can_complete = is_converged or hit_max_cap or state.turn_count >= 4
    
    # ── ACTION 4: Converged Advising Mode (Post-Convergence Chat) ─────────────
    # Once assessment has reached convergence, answer student's follow-ups directly!
    if state.phase == "converged":
        active_mode = "exploratory" if (is_persistently_vague or hit_max_cap) else "targeted"
        recs = get_career_recommendations(
            scores=updated_riasec,
            stream_filter=state.stream,
            excluded_ids=state.excluded_ids,
            is_10th=is_10th,
            user_text=cumulative_text,
            mode=active_mode
        )
        report = build_dossier_payload(recs, updated_riasec, new_history, is_10th, state.score_tier, state.score_raw, mode=active_mode)
        
        # Answer student's query directly as an authoritative counselor
        direct_ans = answer_student_question(user_input, state.stream, state.education_level, new_history)
        advising_reply = f"{direct_ans}\n\nFeel free to explore your full dossier on the right, or ask me any other question about admissions, curriculum, or entrance exams!"
        
        new_history.append({"role": "assistant", "content": advising_reply})
        state.history = new_history
        
        return {
            "type": "advising",
            "question": advising_reply,
            "options": ["Tell me about entrance exams", "What is the 4-year curriculum like?", "Tell me about campus placements", "Download Official PDF Dossier"],
            "can_complete": True,
            "stage": 4,
            "stage_name": "Your recommendation",
            "report": report,
            "state": state.model_dump(),
            "telemetry": {
                "riasec": updated_riasec,
                "entropy": entropy_val,
                "dialogue_mode": "CONVERGED ADVISING",
                "indian_streams": stream_affinities,
                "highlights": highlights,
                "decision_path": decision_path,
                "can_complete": True
            }
        }
        
    # ── ACTION 5: Natural Convergence Milestone (Entropy Resolved + Substantive Input) ────
    # Finalize Admissions Dossier as soon as signal has converged (turn 3+ with clear preference)!
    if (is_converged and len(user_utterances) >= 3) or (len(user_utterances) >= 4 and not is_persistently_vague):
        state.phase = "converged"
        recs = get_career_recommendations(
            scores=updated_riasec,
            stream_filter=state.stream,
            excluded_ids=state.excluded_ids,
            is_10th=is_10th,
            user_text=cumulative_text,
            mode="targeted"
        )
        rec1 = recs["primary"]
        rec2 = recs.get("secondary", rec1)
        rec3 = recs.get("tertiary", rec2)
        primary_title = rec1.get("title", rec1["degree"])
        t2_title = rec2.get("title", rec2["degree"])
        t3_title = rec3.get("title", rec3["degree"])

        report = build_dossier_payload(recs, updated_riasec, new_history, is_10th, state.score_tier, state.score_raw, mode="targeted")
        
        celebratory_closure = (
            f"🎯 Based on our conversation across your interests, problem-solving style, and team collaboration, "
            f"I have finalized your Admissions Dossier!\n\n"
            f"Rather than locking you into a single rigid path, here are your **Top 3 Recommended Pathways** based on your natural cognitive alignment:\n\n"
            f"1. **{primary_title}** (`{rec1['fit_score']}% Fit`) — *Your #1 natural match for your core interests*\n"
            f"2. **{t2_title}** (`{rec2['fit_score']}% Fit`) — *Adjacent specialisation with technical/applied depth*\n"
            f"3. **{t3_title}** (`{rec3['fit_score']}% Fit`) — *Versatile complementary track with broad career flexibility*\n\n"
            f"Your complete academic roadmap, target entrance exams, and feasibility analysis for all three are now ready in the side pane. "
            f"You can click on any of the 3 recommendations in your dossier to explore its specific roadmap!"
        )
        new_history.append({"role": "assistant", "content": celebratory_closure})
        state.history = new_history
        
        return {
            "type": "converged",
            "question": celebratory_closure,
            "options": ["Tell me about entrance exams", "What is the 4-year curriculum like?", "Tell me about campus placements", "Download Official PDF Dossier"],
            "can_complete": True,
            "is_updated_recommendation": True,
            "stage": 4,
            "stage_name": "Your recommendation",
            "report": report,
            "state": state.model_dump(),
            "telemetry": {
                "riasec": updated_riasec,
                "entropy": entropy_val,
                "dialogue_mode": "CONVERGED (High Confidence)",
                "indian_streams": stream_affinities,
                "highlights": highlights,
                "decision_path": decision_path,
                "can_complete": True
            }
        }

    # ── ACTION 6: Turn 15 Limit Cap or Vague Completion (Broad Horizon Protocol) ───────────
    # When student reaches Turn 15 or explicitly asks for completion without converging:
    if hit_max_cap or (wants_completion and not is_converged):
        state.phase = "converged"
        recs = get_career_recommendations(
            scores=updated_riasec,
            stream_filter=state.stream,
            excluded_ids=state.excluded_ids,
            is_10th=is_10th,
            user_text=cumulative_text,
            mode="exploratory"
        )
        report = build_dossier_payload(recs, updated_riasec, new_history, is_10th, state.score_tier, state.score_raw, mode="exploratory")
        
        closure_msg = get_broad_horizon_closure_message(state.education_level, is_10th, recs, cumulative_text)
        new_history.append({"role": "assistant", "content": closure_msg})
        state.history = new_history
        
        return {
            "type": "converged",
            "is_exploratory": True,
            "question": closure_msg,
            "options": ["What are the weekend micro-projects?", "Tell me about broad stream options", "What entrance exams keep options open?", "Download Official PDF Dossier"],
            "can_complete": True,
            "is_updated_recommendation": True,
            "stage": 4,
            "stage_name": "Broad Horizon Discovery",
            "report": report,
            "state": state.model_dump(),
            "telemetry": {
                "riasec": updated_riasec,
                "entropy": entropy_val,
                "dialogue_mode": "BROAD HORIZON (Exploratory / High Optionality)",
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
    
    # During active questioning turns, do not prematurely declare dossier finalized
    dossier_preview = None
        
    # Contextual quick-reply chips based on stage and active domain
    sorted_codes = sorted(updated_riasec.items(), key=lambda x: x[1], reverse=True)
    top_c = sorted_codes[0][0] if sorted_codes else "R"
    cum_lower = cumulative_text.lower()
    
    # Check if student explicitly disavowed gaming or other domains
    is_negated_gaming = any(re.search(pat, user_input.lower()) for pat in [
        r'\b(?:not\s+into|not\s+interested|dont\s+like|don\'t\s+like|hate|dislike)\s+(?:gaming|games|game)\b',
        r'\bnot\s+into\s+gaming\b'
    ])
    if is_negated_gaming and "bsc_animation_vfx" not in state.excluded_ids:
        state.excluded_ids.append("bsc_animation_vfx")

    u_lower = user_input.lower()
    is_confusion = any(re.search(pat, u_lower.strip()) for pat in [
        r'\b(?:did\s*not\s*get|didn\'?t\s*get|not\s*get\s*it|did\s*not\s*understand|didn\'?t\s*understand)\b',
        r'\b(?:what\s*do\s*you\s*mean|what\s*does\s*that\s*mean|confused|confusing)\b'
    ])

    # Check if student just expressed self-doubt or confusion
    doubt_keywords = [
        "what i am good at", "what im good at", "not good at", "no skills",
        "dont know what i am", "don't know what i am", "clueless",
        "not sure what i like", "dont know what to do", "don't know what to do",
        "dont know", "don't know", "no idea", "nothing", "idk",
        "i have no idea", "not sure"
    ]
    is_self_doubt = any(k in user_input.lower() for k in doubt_keywords) or is_uncertain

    past_doubts = sum(1 for m in new_history if m["role"] == "user" and any(k in m["content"].lower() for k in doubt_keywords))
    quick_chips = []
    if is_self_doubt:
        if past_doubts > 1:
            quick_chips = [
                "Show me recommended pathways now",
                "I lean towards creative arts & media",
                "I lean towards technology & gadgets",
                "I lean towards business & commerce"
            ]
        else:
            quick_chips = [
                "Creative visuals, doodling or stories",
                "Tech gadgets or logic puzzles",
                "Chatting with friends & helping people",
                "Exploring business or organizing things"
            ]
    elif is_negated_gaming or is_confusion:
        if top_c == "A" or any(w in cum_lower for w in ["creative", "writing", "art", "design"]):
            quick_chips = [
                "Writing stories, articles & scripts",
                "Visual art, sketching & painting",
                "Both writing and visual arts"
            ]
        elif top_c in ["E", "C"] or any(w in cum_lower for w in ["business", "finance"]):
            quick_chips = [
                "Marketing, branding & sales",
                "Finance, numbers & investment",
                "Startup management & operations"
            ]
        else:
            quick_chips = [
                "Writing software & coding logic",
                "Designing user interfaces",
                "Hands-on hardware & gadgets"
            ]
    elif state.turn_count == 2:
        if any(w in cum_lower for w in ["creative", "writing", "art", "design"]):
            quick_chips = [
                "Writing stories, articles & scripts",
                "Visual art, sketching & painting",
                "Both writing and visual arts"
            ]
        elif any(w in cum_lower for w in ["software", "tech", "tools", "code"]):
            quick_chips = [
                "Software coding & backend logic",
                "User interface & frontend design",
                "Hardware, robotics & gadgets"
            ]
        elif any(w in cum_lower for w in ["business", "finance", "marketing", "trade"]):
            quick_chips = [
                "Marketing, branding & sales",
                "Finance, numbers & investment",
                "Startup management & operations"
            ]
        elif any(w in cum_lower for w in ["science", "lab", "experiment"]):
            quick_chips = [
                "Biological sciences & medicine",
                "Physics & experimental lab research",
                "Chemistry & materials science"
            ]
    elif stage_num == 2:
        if top_c in ["E", "C"] or any(w in cum_lower for w in ["business", "finance", "market", "startup", "commerce", "trade"]):
            quick_chips = ["I analyze market trends & numbers", "I talk directly to customers to understand needs", "I divide tasks and coordinate the team"]
        elif top_c == "A" or any(w in cum_lower for w in ["design", "art", "creative", "writing", "ui", "ux"]):
            quick_chips = ["I sketch visual ideas & prototypes", "I focus on user emotions & storytelling", "I explore unconventional creative angles"]
        elif any(w in cum_lower for w in ["medic", "bio", "doctor", "health", "science", "lab"]):
            quick_chips = ["I research scientific data & root causes", "I run structured experiments in a lab", "I focus on practical patient impact"]
        else:
            quick_chips = ["I break it down into logical steps", "I experiment hands-on to see what works", "I research existing solutions first"]
    elif stage_num == 3:
        # If student hasn't shared their score range yet, offer score tier / style options
        if state.score_tier == "passion_first":
            quick_chips = ["85%+ / High Target", "70% – 85% Competitive", "55% – 70% (Smart Pathways)", "Prefer exploring purely on passion"]
        else:
            if top_c in ["E", "C"] or any(w in cum_lower for w in ["business", "finance", "market", "startup", "commerce", "trade"]):
                quick_chips = ["Leading & negotiating with teams", "Independent financial/market analysis", "Pitching to clients & driving growth"]
            elif top_c == "A" or any(w in cum_lower for w in ["design", "art", "creative", "writing", "ui", "ux"]):
                quick_chips = ["Collaborative creative studio", "Deep solo artistic flow", "Designing interactive experiences"]
            elif any(w in cum_lower for w in ["medic", "bio", "doctor", "health", "science", "lab"]):
                quick_chips = ["Collaborative clinical/lab teamwork", "Deep independent scientific research", "Hands-on patient care"]
            else:
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


def build_dossier_payload(
    recs: Dict, 
    riasec_scores: Dict[str, float], 
    history: List[Dict[str, str]], 
    is_10th: bool,
    score_tier: str = "passion_first",
    score_raw: str = "",
    mode: str = "targeted"
) -> Dict:
    """Compiles the rich document artifact payload with cognitive journey, Top 3 recommendations, and feasibility matrix."""
    primary_deg = recs["primary"]
    secondary_deg = recs.get("secondary", primary_deg)
    tertiary_deg = recs.get("tertiary", secondary_deg)
    top_three = recs.get("top_three", [primary_deg, secondary_deg, tertiary_deg])
    top_traits = recs["top_riasec"]
    effective_mode = recs.get("mode", mode)
    
    user_utterances = [msg["content"] for msg in history if msg["role"] == "user"]
    user_full_text = " ".join(user_utterances)
    
    # Extract evidence quotes ("You said: ...")
    evidence_quotes = extract_representative_quotes(user_utterances, top_traits, primary_deg)
    
    # Personalized justifications
    justifications = generate_personalized_justifications(
        primary_deg, secondary_deg, top_traits, user_utterances
    )
    
    # Cognitive Journey & Psychological Alignment Synthesis (Interests, Thinking Style, Strengths)
    cognitive_journey = generate_cognitive_journey_report(
        riasec_scores, user_full_text, primary_deg, is_10th, mode=effective_mode
    )
    
    # Pre-generate Feasibility Matrix for each of the Top 3 options
    feasibility_map = {
        0: generate_academic_feasibility_matrix(primary_deg, score_tier, is_10th, score_raw),
        1: generate_academic_feasibility_matrix(secondary_deg, score_tier, is_10th, score_raw),
        2: generate_academic_feasibility_matrix(tertiary_deg, score_tier, is_10th, score_raw)
    }
    
    primary_title = recs.get("recommended_title", primary_deg["degree"])
    
    return {
        "is_10th": is_10th,
        "mode": effective_mode,
        "confidence_level": recs.get("confidence_level", "high" if effective_mode == "targeted" else "exploratory"),
        "primary_recommendation": primary_title,
        "primary_degree_name": primary_deg["degree"],
        "primary_stream": primary_deg["stream"],
        "primary_fit": primary_deg["fit_score"],
        "primary_id": primary_deg.get("id", ""),
        "secondary_recommendation": secondary_deg.get("title", secondary_deg["degree"]),
        "secondary_degree_name": secondary_deg["degree"],
        "secondary_stream": secondary_deg["stream"],
        "secondary_fit": secondary_deg["fit_score"],
        "secondary_id": secondary_deg.get("id", ""),
        "tertiary_recommendation": tertiary_deg.get("title", tertiary_deg["degree"]),
        "tertiary_degree_name": tertiary_deg["degree"],
        "tertiary_stream": tertiary_deg["stream"],
        "tertiary_fit": tertiary_deg["fit_score"],
        "tertiary_id": tertiary_deg.get("id", ""),
        "top_three": top_three,
        "feasibility_map": feasibility_map,
        "alternatives": recs.get("alternatives", []),
        "target_entrance_exams": primary_deg.get("exams", ["CUET UG"]),
        "career_profile": primary_deg.get("profile", ""),
        "top_riasec_traits": top_traits,
        "all_riasec_scores": riasec_scores,
        "evidence_quotes": evidence_quotes,
        "justifications": justifications,
        "cognitive_journey": cognitive_journey,
        "academic_feasibility": feasibility_map[0]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Static Assets Mounting
# ─────────────────────────────────────────────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"  [Warning] Frontend directory '{frontend_dir}' not found.")
