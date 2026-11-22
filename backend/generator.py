"""
Local Language Generation Engine (Gemma-2B via Local Ollama)
CSE708 NLP Lab Practical — Syllabus Sessions 28–29 (Language Modeling & On-Device NLG)
"""

import requests
import json
from typing import Dict, List, Optional
from riasec import RIASEC_CODES

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ollama Connectivity & Generation Client
# ─────────────────────────────────────────────────────────────────────────────

def query_local_gemma(prompt: str, timeout: int = 25, max_tokens: int = 95) -> Optional[str]:
    """
    Sends an inference request to the local Ollama instance running on localhost:11434.
    Runs 100% locally on the student's machine (accelerated by RTX 4060 GPU).
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.60,
                "top_p": 0.88,
                "num_predict": max_tokens
            }
        }
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        if res.status_code == 200:
            text = res.json().get("response", "").strip()
            # Clean unwanted artifacts
            text = text.replace('"', '').replace('**', '').strip()
            # Strip generic AI filler prefixes
            for filler in [
                "That's a great choice!", "That's a great idea!", "That's a great starting point!",
                "That's an interesting thought!", "That's wonderful!", "Great choice!"
            ]:
                if text.lower().startswith(filler.lower()):
                    text = text[len(filler):].strip()
            return text
    except Exception as e:
        print(f"  [Generator Notice] Local Ollama call bypassed/timed out ({e}). Utilizing deterministic dilemma generator.")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dynamic Conversational Question Generator (Natural & Stage-Aware)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dynamic_question(
    turn_num: int,
    user_reply: str,
    riasec_scores: Dict[str, float],
    is_ambiguous: bool,
    education_level: str,
    stream: str,
    history: List[Dict[str, str]],
    is_uncertain: bool = False
) -> str:
    """
    Dynamically generates the next question in real time based on the student's
    latest answer and current Journey Stage (1. Interests -> 2. Thinking Style -> 3. Strengths).
    """
    # Identify top 2 competing RIASEC traits
    sorted_traits = sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)
    top_code, top_val = sorted_traits[0]
    sec_code, sec_val = sorted_traits[1]
    
    top_name = RIASEC_CODES[top_code]["name"]
    sec_name = RIASEC_CODES[sec_code]["name"]
    top_desc = RIASEC_CODES[top_code]["title"]
    sec_desc = RIASEC_CODES[sec_code]["title"]
    
    # Determine Journey Stage
    if turn_num <= 2:
        stage_focus = "STAGE 1: INTERESTS & PASSIONS. Deepen what excites them about what they just mentioned."
    elif turn_num <= 4:
        stage_focus = "STAGE 2: THINKING STYLE & PROBLEM SOLVING. Ask how they tackle a crisis or break down a tough challenge."
    else:
        stage_focus = "STAGE 3: STRENGTHS & WORK ENVIRONMENT. Ask about their ideal work environment (solo deep focus vs team leadership, abstract theory vs building physical/visual things)."
    
    # Compile recent dialogue context (last 3 exchanges)
    recent_context = ""
    for msg in history[-3:]:
        role = "Student" if msg["role"] == "user" else "Counselor"
        recent_context += f"{role}: {msg['content']}\n"

    # Contextual prompt engineering with strict anti-loop and anti-filler rules
    if is_uncertain:
        prompt = f"""You are a warm, empathetic Indian career counselor talking with a student ({education_level}).
Recent chat:
{recent_context}
The student is feeling unsure and said: "{user_reply}".

CRITICAL RULES:
1. Do NOT start with "That's a great choice" or generic filler.
2. Acknowledge their hesitation with kindness in one brief phrase.
3. Ask a very simple, grounded question comparing two easy, natural activities (e.g., solving a logic puzzle on a computer vs creative writing/sketching or helping people).
4. Do NOT use multiple-choice (A/B/C) or bullets. Maximum 30 words:"""

    else:
        prompt = f"""You are an insightful, empathetic Indian career counselor at Navrachana University talking with a student ({education_level}, {stream}).
Recent chat:
{recent_context}
Student's latest response: "{user_reply}"
Current Goal: {stage_focus}

CRITICAL RULES:
1. DIRECT RELEVANCE: Look at what the student literally said ("{user_reply}"). If they mentioned coding, software, mobile apps, or automation, talk about SOFTWARE, CODING, or APPS. Never force business, sales, or unrelated topics!
2. NO PREACHING: Never lecture the student about "focusing on academic goals" or "studying hard for entrance exams". Always stay encouraging, curious, and collaborative.
3. NO GENERIC FILLER: Do NOT start with "That's a great choice!", "That's a great idea!", or "Interesting!". Start directly with your insight or question.
4. Keep it conversational, like a thoughtful mentor talking 1-on-1. Maximum 35 words. No bullets or A/B/C:"""

    response = query_local_gemma(prompt, timeout=25, max_tokens=85)
    
    # Fail-Safe Fallback
    if not response or len(response.strip()) < 15:
        if is_uncertain:
            return "It's completely normal to feel unsure right now! Let's make it simpler: between these two, do you enjoy solving logical puzzles and tech on a computer, or do you prefer creative design, writing, and hands-on projects?"
        response = get_deterministic_fallback_question(top_code, sec_code, is_ambiguous, stream, user_reply, turn_num)
        
    return response

# ─────────────────────────────────────────────────────────────────────────────
# 3. Direct Student Question Answerer (Mixed-Initiative Dialogue)
# ─────────────────────────────────────────────────────────────────────────────

def answer_student_question(question: str, stream: str, edu_level: str, history: List[Dict[str, str]]) -> str:
    """
    Answers a student's direct question concisely and authoritatively via Gemma-2B,
    addressing entrance exams, syllabus doubts, or career viability without preaching.
    """
    prompt = f"""You are a helpful, knowledgeable career counselor talking with a student ({edu_level}, {stream}).
The student is asking: "{question}"

Answer their question directly and helpfully in 1 to 2 conversational sentences (under 40 words).
Do NOT give generic lectures about studying hard or academic goals. Answer their specific curiosity:"""

    ans = query_local_gemma(prompt, timeout=25, max_tokens=85)
    if not ans or len(ans.strip()) < 15:
        q_lower = question.lower()
        if "bio" in q_lower or "pcb" in q_lower:
            return "Yes, absolutely! Students with PCB can enter Bioinformatics, Biotechnology, and Cognitive Sciences with foundational bridge courses in math and computational tools."
        elif "ca" in q_lower or "account" in q_lower:
            return "Chartered Accountancy demands rigorous statutory auditing and taxation discipline through the ICAI exams (Foundation, Intermediate, Final), often paired with a B.Com degree."
        elif "replace" in q_lower or "ai" in q_lower:
            return "AI is automating repetitive tasks, but it is dramatically increasing demand for engineers who understand machine learning architecture, problem-solving, and human-centric systems."
        return "That is an important question. In Indian universities, interdisciplinary pathways and entrance criteria (like CUET or JEE) offer flexible options tailored to your strengths."
    return ans

# ─────────────────────────────────────────────────────────────────────────────
# 4. Human-in-the-Loop Recalibration Inquiry
# ─────────────────────────────────────────────────────────────────────────────

def generate_recalibration_inquiry(rejected_degree: str, user_critique: str = "") -> str:
    """
    Generates a natural diagnostic response when a student dislikes the recommendation.
    """
    if user_critique:
        return f"Understood! Thanks for being honest. You mentioned '{user_critique}' regarding {rejected_degree}. What would feel more exciting — something leaning more towards creative design, analytical technology, hands-on engineering, or human sciences?"
    return f"That's completely fair! Finding the right path often starts with ruling out what doesn't fit. What specifically felt off about {rejected_degree}? Was it the daily subject work, the career environment, or something else you'd rather focus on?"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Fail-Safe Deterministic Dilemma Generator (Offline Backup)
# ─────────────────────────────────────────────────────────────────────────────

def get_deterministic_fallback_question(code_a: str, code_b: str, is_ambiguous: bool, stream: str, user_text: str = "", turn_num: int = 1) -> str:
    user_lower = user_text.lower()
    
    # Stage 2 (Turns 3-4): Thinking style under crisis
    if turn_num in [3, 4]:
        return "When a major group project runs into chaos the day before submission, what is your natural reaction? Do you immediately make an organized checklist, take ownership of fixing the hardest technical piece yourself, or focus on coordinating the team?"
        
    # Context-specific tech fallback
    if any(k in user_lower for k in ["code", "coding", "software", "program", "app", "tech", "platform"]):
        return "Since you enjoy the coding and technology side, would you prefer writing the backend algorithms and database architecture, or designing the visual user interface and user experience?"
    
    # Context-specific business fallback
    if any(k in user_lower for k in ["business", "market", "startup", "invest", "money", "sales"]):
        return "In a new venture, would you feel more energized pitching corporate strategy and marketing to clients, or managing detailed financial budgets and auditing balance sheets?"
        
    # Standard RIASEC dilemmas
    pair = (code_a, code_b)
    if "I" in pair and "R" in pair:
        return "When working on a high-tech engineering project, do you prefer developing the software code and computational algorithms, or physically assembling and testing the mechanical hardware?"
    elif "I" in pair and "E" in pair:
        return "Would you rather spend your time deeply analyzing complex datasets and designing algorithms, or pitching strategic business models and leading teams?"
    elif "A" in pair:
        return "Do you find yourself more fulfilled by creative artistic design and visual prototyping, or structured analytical problem solving?"
    else:
        name_a = RIASEC_CODES[code_a]["name"]
        name_b = RIASEC_CODES[code_b]["name"]
        return f"Between tackling the {name_a} aspects of a project or leaning into the {name_b} side, which one feels more natural to your style of thinking?"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Opening Scenario Generator (Turn 0 Icebreaker)
# ─────────────────────────────────────────────────────────────────────────────

def get_opening_scenario(stream: str, edu_level: str) -> str:
    """
    Provides a balanced, relatable scenario grounded in realistic student life,
    specifically adapted for After 10th (Stream selection) vs After 12th (Degree selection).
    """
    stream_clean = stream.lower() if stream else ""
    edu_clean = edu_level.lower() if edu_level else ""
    
    # After 10th Standard: Choosing a Stream
    if "10" in edu_clean or "stream" in stream_clean:
        return "To help work out which stream fits how you think: imagine you have complete freedom to choose a project for a school exhibition. Would you rather build a working tech/science project, run a commercial trade stall, or design a creative magazine/exhibit?"
        
    # After 12th: Specific Stream or Degree
    if "pcm" in stream_clean or "math" in stream_clean:
        return "Welcome! Imagine you are invited to join an inter-college technical expo. Would you rather write the software algorithms and build the app, design and assemble the physical robotics hardware, or analyze mathematical data models?"
    elif "pcb" in stream_clean or "bio" in stream_clean:
        return "Welcome! Imagine you are spending a day at an innovation science center. Would you rather investigate cellular biology and genetic tests in a lab, explore medical healthcare and patient therapy, or study ecological and plant sciences?"
    elif "commerce" in stream_clean:
        return "Welcome! Imagine you are working on a new business project. Would you rather create marketing campaigns and negotiate partnership deals, analyze financial data and investment budgets, or study economic market trends?"
    elif "arts" in stream_clean or "liberal" in stream_clean:
        return "Welcome! Imagine you are contributing to a creative cultural initiative. Would you rather write original stories and articles, design visual graphics and digital media, or explore human psychology and behavioral counseling?"
    else:
        return "To discover your ideal university degree: when you think about spending the next four years diving deep into a subject, does your mind naturally lean towards solving algorithmic logic and technology, investigating natural/medical sciences, managing business/finance, or creative design and humanities?"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Personalized Career Justification Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_personalized_justifications(
    primary_degree: Dict,
    secondary_degree: Dict,
    top_traits: List[Dict],
    student_utterances: List[str]
) -> List[str]:
    t1 = top_traits[0]
    t2 = top_traits[1]
    
    bullets = []
    
    # Bullet 1: Quantitative Psychometric Match
    bullets.append(
        f"**Psychometric Trait Alignment**: Sentence-BERT dense vector analysis identified a prominent **{t1['score']}% match** for **{t1['name']} ({t1['code']})** and **{t2['score']}% match** for **{t2['name']} ({t2['code']})**, establishing a strong cognitive profile for **{primary_degree['degree']}**."
    )
    
    # Bullet 2: Contextual User Answer Grounding
    if student_utterances:
        meaningful_utterances = [u for u in student_utterances if len(u.strip()) > 10]
        sample_quote = meaningful_utterances[-1].strip() if meaningful_utterances else student_utterances[-1].strip()
        if len(sample_quote) > 65:
            sample_quote = sample_quote[:65] + "..."
        bullets.append(
            f"**Cognitive Grounding**: Throughout our conversation (e.g., when discussing *\"{sample_quote}\"*), your problem-solving instinct consistently matched the core demands of this discipline."
        )
    else:
        bullets.append(
            f"**Cognitive Grounding**: Your responses consistently demonstrated an instinctive preference for {t1['name']} problem-solving over competing disciplines."
        )
        
    # Bullet 3: Multidisciplinary Alternate
    bullets.append(
        f"**Interdisciplinary Alternate**: As a high-potential secondary track, **{secondary_degree['degree']}** in {secondary_degree['stream']} shows a strong correlation, providing a versatile career safety net."
    )
    
    # Bullet 4: Indian Testing & Roadmap
    exams_str = ", ".join(primary_degree["exams"])
    bullets.append(
        f"**Target Indian Entrance Pathway**: To pursue this specialization, your primary target examinations include **{exams_str}**, followed by admissions into leading central institutes."
    )
    
    return bullets
