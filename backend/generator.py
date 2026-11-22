"""
Local Language Generation Engine (Gemma-2B via Local Ollama)
Empathetic Dialogue Generation, Constraint Enforcement & On-Device NLG
"""

import requests
import json
import re
from typing import Dict, List, Optional
from riasec import RIASEC_CODES

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ollama Connectivity & Generation Client
# ─────────────────────────────────────────────────────────────────────────────

def query_local_gemma(prompt: str, timeout: int = 25, max_tokens: int = 80) -> Optional[str]:
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
                "temperature": 0.55,
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
                "That's an interesting thought!", "That's wonderful!", "Great choice!",
                "That's fascinating!", "That's interesting!"
            ]:
                if text.lower().startswith(filler.lower()):
                    text = text[len(filler):].strip()
                    
            # Programmatically strip robotic parroting openings
            text = re.sub(r'^(you mentioned [^.?!]+[.?!]\s*)+', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'^(so,?\s+you really thrive in [^.?!]+[.?!]\s*)+', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'^(it\'?s great you\'?re interested in [^.?!]+[.?!]\s*)+', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'^(building software,?\s*huh\??\s*i\'?m sensing [^.?!]+[.?!]\s*)+', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'^(\w+\s+really interests you\.\s*)+', '', text, flags=re.IGNORECASE).strip()
            
            # Remove any unwanted lecture preaching if model hallucinated entrance exam talk
            if "entrance exams like" in text.lower() or "focus on your academic goals" in text.lower():
                sentences = re.split(r'(?<=[.?!])\s+', text)
                clean_sentences = [
                    s for s in sentences 
                    if not any(k in s.lower() for k in ["academic goals", "cuet/jee", "entrance exam", "crucial for admission"])
                ]
                text = " ".join(clean_sentences).strip()
                
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
    
    # Compile recent dialogue context (last 6 exchanges)
    recent_context = ""
    history_lower = ""
    for msg in history[-6:]:
        role = "Student" if msg["role"] == "user" else "Counselor"
        content = msg["content"]
        recent_context += f"{role}: {content}\n"
        history_lower += f" {content.lower()}"

    # Track what has already been discussed in dialogue to strictly avoid repeating questions
    team_already_discussed = any(w in history_lower for w in ["team", "solo", "collaborat", "alone", "group", "leading"])
    coding_discussed = any(w in history_lower for w in ["code", "coding", "software", "app", "mobile", "algorithm", "developer"])
    user_ux_discussed = any(w in history_lower for w in ["user", "ux", "ui", "experience", "design", "screens"])
    backend_logic_discussed = any(w in history_lower for w in ["backend", "intricate logic", "logic behind", "algorithms or", "crafting user", "visual appeal"])
    
    # Resolve student shorthand like "the first one", "first", "coding"
    resolved_reply = user_reply
    u_clean = user_reply.strip().lower()
    if u_clean in ["the first one", "first one", "1", "option 1", "first"]:
        resolved_reply = f"{user_reply} (student selects the first option from your previous question)"
    elif u_clean in ["the second one", "second one", "2", "option 2", "second"]:
        resolved_reply = f"{user_reply} (student selects the second option from your previous question)"

    # Detect student's active domain from RIASEC codes and conversation keywords
    is_business = top_code in ["E", "C"] or any(w in history_lower for w in ["business", "finance", "market", "trade", "startup", "commerce", "money", "invest", "entrepreneur"])
    is_creative = top_code == "A" or any(w in history_lower for w in ["design", "art", "creative", "writing", "magazine", "media", "ui", "ux", "film", "draw"])
    is_healthcare = (top_code in ["I", "S"]) and any(w in history_lower for w in ["medic", "doctor", "medicine", "biology", "health", "biotech", "patient", "clinic", "hospital"])

    # Determine Journey Stage with dynamic non-repeating sub-focus tailored to their domain
    if turn_num <= 2:
        if is_business:
            stage_focus = "Deepen what excites them about business — e.g. launching a startup venture, financial markets/investments, or brand marketing."
        elif is_creative:
            stage_focus = "Deepen what excites them about creative arts — e.g. visual UI/product design, creative writing, illustration, or multimedia."
        elif is_healthcare:
            stage_focus = "Deepen what excites them in science/health — e.g. clinical patient care, laboratory research, or biotechnology."
        else:
            stage_focus = "Deepen what excites them about technology — e.g. software coding, AI algorithms, physical robotics, or tech systems."
    elif turn_num <= 4:
        if is_business:
            stage_focus = "Ask how they like to tackle business challenges — understanding customer needs, analyzing financial data, or creative marketing."
        elif is_creative:
            stage_focus = "Ask how they like to create — spontaneous intuitive experimentation or structured user research and prototyping."
        elif is_healthcare:
            stage_focus = "Ask how they approach scientific problem solving — rigorous empirical testing or diagnosing practical real-life cases."
        else:
            stage_focus = "Ask how they like to tackle problem-solving — breaking down logical algorithms or experimenting hands-on with tools."
    elif not team_already_discussed:
        if is_business:
            stage_focus = "Ask whether they prefer leading and negotiating with teams or diving deep into independent financial/strategic analysis."
        elif is_creative:
            stage_focus = "Ask whether they thrive in collaborative creative studios or deep solo artistic flow."
        elif is_healthcare:
            stage_focus = "Ask whether they prefer collaborative hospital/clinic teamwork or independent scientific research."
        else:
            stage_focus = "Ask whether they prefer working independently with deep focus or collaborating actively in a group."
    elif is_business:
        stage_focus = "Ask what kind of industry or real-world problem they would love to build a venture or business around."
    elif is_creative:
        stage_focus = "Ask what kind of audience or real-world cause they want their creative work to impact."
    elif is_healthcare:
        stage_focus = "Ask what real-world health or scientific breakthrough they would love to contribute to."
    elif (coding_discussed or user_ux_discussed) and not backend_logic_discussed:
        stage_focus = "Ask whether they prefer working on the core logic and technical algorithms, or crafting the user experience and interface."
    else:
        stage_focus = "Ask what real-world domain (like education, climate, productivity, or gaming) they would love to build technology for."

    # Prompt design for Gemma 2B: Direct, positive, without negative constraint leakage
    if is_uncertain:
        prompt = f"""You are CareerSaathi, an encouraging career mentor for Indian students ({education_level}).
Recent conversation:
{recent_context}
The student feels unsure and said: "{resolved_reply}".

Respond warmly in 1 or 2 conversational sentences (under 30 words).
Ask a simple comparison: do they prefer solving technical/logic puzzles on a computer, or creative design and working with people?"""

    else:
        prompt = f"""You are CareerSaathi, an empathetic, smart career mentor for Indian students ({education_level}, {stream}).
Recent conversation:
{recent_context}
Student: "{resolved_reply}"
Goal: {stage_focus}

Respond naturally like a friendly guide.
1. Be direct and conversational. Do not repeat what they just said.
2. Ask ONE engaging, open question advancing the conversation towards their goal.
3. Keep it under 30 words. No multiple choice, bullets, or lectures."""

    response = query_local_gemma(prompt, timeout=25, max_tokens=80)
    
    # Intercept repetition of solo vs team if already discussed
    if response and team_already_discussed and any(w in response.lower() for w in ["solo", "collaborat", "work alone"]):
        if is_business:
            response = "What industry or customer challenge would you be most excited to build a business or venture around?"
        elif is_creative:
            response = "What kind of visual project or creative medium are you most excited to master?"
        else:
            response = "What real-world problem would you be most excited to solve — like education, healthcare, or daily productivity?"

    # Intercept repetition of backend vs UI if already discussed
    if response and backend_logic_discussed and any(w in response.lower() for w in ["intricate logic", "backend or", "user interface or", "algorithms or crafting", "visual appeal"]):
        response = "Between building systems that solve everyday problems for millions of users or designing intelligent algorithms, which feels more inspiring?"
        
    # Fail-Safe Fallback
    if not response or len(response.strip()) < 15:
        if is_uncertain:
            return "It's completely normal to feel unsure right now! Between these two, do you enjoy solving logical puzzles and tech on a computer, or do you prefer creative design, writing, and hands-on projects?"
        response = get_deterministic_fallback_question(top_code, sec_code, is_ambiguous, stream, user_reply, turn_num)
        
    return response

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
    
    # Context-specific business fallback
    if any(k in user_lower for k in ["business", "market", "startup", "invest", "money", "sales", "trade", "finance"]) or code_a in ["E", "C"]:
        if turn_num in [3, 4]:
            return "When a major business initiative hits an unexpected roadblock, what is your natural reaction? Do you step up to lead and align the team, crunch the numbers to pivot the strategy, or focus on talking to customers to find a solution?"
        return "In a new venture, would you feel more energized pitching strategy and marketing to clients, or managing detailed financial budgets and operations?"

    # Context-specific creative arts fallback
    if any(k in user_lower for k in ["design", "art", "creative", "writing", "ui", "ux"]) or code_a == "A":
        if turn_num in [3, 4]:
            return "When a creative showcase or design deadline is approaching fast, what is your natural move? Do you experiment with bold visual revisions, seek input from peers, or double down on perfecting the details?"
        return "Do you find yourself more fulfilled by creative artistic design and visual prototyping, or structured analytical problem solving?"

    # Stage 2 (Turns 3-4): Thinking style under pressure (General / Tech)
    if turn_num in [3, 4]:
        return "When a major project hits an unexpected roadblock before submission, what is your natural reaction? Do you make a structured checklist, take ownership of solving the core bottleneck yourself, or focus on coordinating with everyone?"
        
    # Context-specific tech fallback
    if any(k in user_lower for k in ["code", "coding", "software", "program", "app", "tech", "platform"]) or code_a in ["R", "I"]:
        return "Since you enjoy the technical side, would you prefer writing the backend algorithms and database architecture, or designing the visual user interface and user experience?"
        
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
