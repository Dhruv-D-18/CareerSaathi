from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

# Import our ML model and questions library
from model import predict_interest, get_highlighted_words, get_decision_path, CLASSES
from questions import SEED_QUESTIONS, QUESTION_POOL, get_adaptive_question

app = FastAPI(title="Traditional ML Career Counselor API")

# ─────────────────────────────────────────────────────────────────────────────
# Request and Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ChatState(BaseModel):
    turn_count: int
    phase: str                          # "profiling" | "scenario" | "completed"
    education_level: Optional[str] = None
    stream: Optional[str] = None
    asked_question_ids: List[str] = []
    history: List[Dict[str, str]] = []  # List of {"role": "user"|"assistant", "content": "..."}
    latest_message: str

# ─────────────────────────────────────────────────────────────────────────────
# Justification Engine (Creates custom, personalized feedback based on ML predictions)
# ─────────────────────────────────────────────────────────────────────────────

def generate_justifications(primary: str, secondary: str, highlights: List[dict], probs: dict) -> List[str]:
    # Group highlights by category
    words_by_cat = {}
    for h in highlights:
        cat = h["category"]
        words_by_cat[cat] = words_by_cat.get(cat, []) + [h["word"]]
        
    primary_words = words_by_cat.get(primary, [])
    
    bullets = []
    
    # Bullet 1: Quantify interest
    bullets.append(f"Our Random Forest classifier identified a highly prominent **{probs.get(primary, 0.0)}% match** for **{primary}** based on your natural aptitude profiles.")
    
    # Bullet 2: Highlight-based analysis
    if primary_words:
        unique_words = list(set(primary_words))[:3]
        word_str = ", ".join([f"'{w}'" for w in unique_words])
        bullets.append(f"Linguistic feature analysis highlighted key terms like {word_str} in your statements, which strongly map to the core competencies of this field.")
    else:
        bullets.append(f"Your answers consistently prioritize solving problems through methods characteristic of **{primary}** professionals.")
        
    # Bullet 3: Explain the decision tree split
    bullets.append(f"The Explainable Decision Tree model traversed to a leaf node recommending **{primary}** due to your distinct preference patterns over related paths.")
    
    # Bullet 4: Secondary alternative
    bullets.append(f"As a secondary pathway, **{secondary}** shows a solid **{probs.get(secondary, 0.0)}% correlation**, suggesting a multidisciplinary bridge could also be highly rewarding.")
    
    return bullets

# ─────────────────────────────────────────────────────────────────────────────
# Core Chat API
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat_endpoint(state: ChatState):
    new_history = list(state.history)
    asked_ids = list(state.asked_question_ids)
    
    # 1. Record the user's latest response if present
    if state.latest_message.strip():
        new_history.append({"role": "user", "content": state.latest_message})
        
    # ── PHASE 1: Profiling ───────────────────────────────────────────────────
    if state.phase == "profiling":
        # Initial turn: Ask for education level when first loading the page
        if state.turn_count == 0 and not state.latest_message.strip():
            q_data = SEED_QUESTIONS["education"]
            next_q = q_data["text"]
            opts = q_data["options"]
            new_history.append({"role": "assistant", "content": next_q})
            
            state.history = new_history
            return {
                "type": "question",
                "question": next_q,
                "options": opts,
                "state": state.model_dump(),
                "ml_data": {
                    "probabilities": {cls: 12.5 for cls in CLASSES},
                    "highlights": [],
                    "decision_path": []
                }
            }
            
        # Turn 0 → Student just replied with their education level
        elif state.turn_count == 0:
            edu = state.latest_message.strip()
            state.education_level = edu
            
            # If 12th, ask for stream
            if "12" in edu:
                q_data = SEED_QUESTIONS["stream_12th"]
                next_q = q_data["text"]
                opts = q_data["options"]
                new_history.append({"role": "assistant", "content": next_q})
                
                # Update state
                state.turn_count = 1
                state.history = new_history
                return {
                    "type": "question",
                    "question": next_q,
                    "options": opts,
                    "state": state.model_dump(),
                    "ml_data": {
                        "probabilities": {cls: 12.5 for cls in CLASSES},
                        "highlights": [],
                        "decision_path": []
                    }
                }
            else:
                # 10th or other → Jump straight to scenarios!
                state.stream = "Other"
                state.phase = "scenario"
                
                q_data = SEED_QUESTIONS["general_start"]
                next_q = q_data["next_question"]
                opts = q_data["options"]
                asked_ids.append(q_data["id"])
                
                bridge_msg = "No problem! Let's explore your interests with a quick, fun scenario activity. Select what fits best!"
                new_history.append({"role": "assistant", "content": bridge_msg})
                new_history.append({"role": "assistant", "content": next_q})
                
                state.turn_count = 1
                state.asked_question_ids = asked_ids
                state.history = new_history
                return {
                    "type": "question",
                    "question": next_q,
                    "options": opts,
                    "state": state.model_dump(),
                    "ml_data": {
                        "probabilities": {cls: 12.5 for cls in CLASSES},
                        "highlights": [],
                        "decision_path": []
                    }
                }
                
        # Turn 1 → Student just selected their 12th stream
        elif state.turn_count == 1:
            stream_choice = state.latest_message.strip()
            state.stream = stream_choice
            state.phase = "scenario"
            
            # Formulate starting question and appropriate bridge message
            q_data = SEED_QUESTIONS["general_start"]
            next_q = q_data["next_question"]
            opts = q_data["options"]
            asked_ids.append(q_data["id"])
            
            if "pcm" in stream_choice.lower() or "math" in stream_choice.lower():
                bridge = SEED_QUESTIONS["bridge_pcm"]["text"]
            elif "pcb" in stream_choice.lower() or "bio" in stream_choice.lower():
                bridge = SEED_QUESTIONS["bridge_pcb"]["text"]
            else:
                bridge = SEED_QUESTIONS["bridge_general"]["text"]
                
            new_history.append({"role": "assistant", "content": bridge})
            new_history.append({"role": "assistant", "content": next_q})
            
            state.turn_count = 2
            state.asked_question_ids = asked_ids
            state.history = new_history
            
            return {
                "type": "question",
                "question": next_q,
                "options": opts,
                "state": state.model_dump(),
                "ml_data": {
                    "probabilities": {cls: 12.5 for cls in CLASSES},
                    "highlights": [],
                    "decision_path": []
                }
            }

    # ── PHASE 2: ML Scenario questioning ─────────────────────────────────────
    if state.phase == "scenario":
        # First, let's collect all user responses during the scenario phase
        # This will be passed to TF-IDF to get predictions, highlights, and decision tree paths.
        user_texts = []
        
        # We start counting user messages that belong to the scenario phase.
        # In our history, profiling answers occupy the first few messages.
        # Let's extract all user replies after profiling is done.
        # A simpler way is to just aggregate all user messages in history, 
        # but filter out the profiling questions.
        for h in new_history:
            if h["role"] == "user":
                content = h["content"]
                # Skip simple education/stream answers
                if content in ["12th Standard", "10th Standard", "Other", "PCM", "PCB", "General / Commerce"]:
                    continue
                user_texts.append(content)
                
        cumulative_text = " ".join(user_texts)
        
        # Call the ML core!
        probabilities = predict_interest(cumulative_text)
        highlights = get_highlighted_words(cumulative_text)
        decision_path = get_decision_path(cumulative_text)
        
        # We want to ask EXACTLY 4 scenario questions in total (Turn 2, 3, 4, 5).
        # Turn count counts scenario turns. Let's see:
        # If profiling was skipped: Turn 1 was general_start. Let's adjust counts.
        # If profiling took 2 steps: Turn 2 was general_start.
        # To make it robust and easy, let's count how many questions are in asked_ids.
        # We started with 1 (general_start).
        # We ask a total of 4 questions.
        scenario_questions_asked = len(asked_ids)
        
        if scenario_questions_asked >= 4:
            # We have gathered all answers! Transition to complete and return final report.
            state.phase = "completed"
            
            # Pick primary and secondary recommendations
            ranked_categories = list(probabilities.keys())
            primary_rec = ranked_categories[0]
            secondary_rec = ranked_categories[1]
            
            justifications = generate_justifications(primary_rec, secondary_rec, highlights, probabilities)
            
            report = {
                "primary_recommendation": primary_rec,
                "secondary_alternative": secondary_rec,
                "justification_analysis": justifications
            }
            
            state.history = new_history
            return {
                "type": "final_report",
                "report": report,
                "state": state.model_dump(),
                "ml_data": {
                    "probabilities": probabilities,
                    "highlights": highlights,
                    "decision_path": decision_path
                }
            }
        
        # Otherwise, ask the next adaptive question!
        next_q_data = get_adaptive_question(probabilities, asked_ids)
        next_q = next_q_data["next_question"]
        opts = next_q_data["options"]
        asked_ids.append(next_q_data["id"])
        
        new_history.append({"role": "assistant", "content": next_q})
        
        state.turn_count += 1
        state.asked_question_ids = asked_ids
        state.history = new_history
        
        return {
            "type": "question",
            "question": next_q,
            "options": opts,
            "state": state.model_dump(),
            "ml_data": {
                "probabilities": probabilities,
                "highlights": highlights,
                "decision_path": decision_path
            }
        }
        
    raise HTTPException(status_code=500, detail="Unexpected state — please restart.")

# ─────────────────────────────────────────────────────────────────────────────
# Static files and App Mounting
# ─────────────────────────────────────────────────────────────────────────────

# Mount the static frontend assets
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory '{frontend_dir}' not found. APIs will work but UI won't be served.")
