"""
Local Language Generation Engine (Gemma-2B via Local Ollama)
Empathetic Dialogue Generation, Constraint Enforcement & On-Device NLG
"""

import requests
import json
import re
import random
import datetime
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
                "That's fascinating!", "That's interesting!", "That's awesome!", "Awesome!", "Cool!"
            ]:
                if text.lower().startswith(filler.lower()):
                    text = text[len(filler):].strip()
                    
            # Programmatically strip robotic parroting and canned cheer openings
            text = re.sub(r'^(it\'?s great you (?:enjoy|like|are into|love|thrive in) [^.?!]+[.?!]\s*)+', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'^(that\'?s (?:awesome|great|cool|wonderful|amazing)!?[^.?!]*[.?!]\s*)+', '', text, flags=re.IGNORECASE).strip()
            text = re.sub(r'^(awesome!?[^.?!]*[.?!]\s*)+', '', text, flags=re.IGNORECASE).strip()
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
    user_history_lower = ""
    for msg in history[-6:]:
        role = "Student" if msg["role"] == "user" else "Counselor"
        content = msg["content"]
        recent_context += f"{role}: {content}\n"
        history_lower += f" {content.lower()}"
        if msg["role"] == "user":
            user_history_lower += f" {content.lower()}"

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

    # Detect student's active domain from RIASEC codes and USER-ONLY conversation keywords
    is_business = top_code in ["E", "C"] or any(w in user_history_lower for w in ["business", "finance", "market", "trade", "startup", "commerce", "money", "invest", "entrepreneur"])
    is_creative = top_code == "A" or any(w in user_history_lower for w in ["design", "art", "creative", "writing", "magazine", "media", "ui", "ux", "film", "draw", "animation"])
    is_healthcare = (top_code in ["I", "S"]) and any(w in user_history_lower for w in ["medic", "doctor", "medicine", "biology", "health", "biotech", "patient", "clinic", "hospital", "pharma"])

    # Determine Journey Stage with dynamic non-repeating sub-focus tailored to their domain
    if turn_num <= 2:
        if is_business:
            stage_focus = "Deepen what excites them about business — e.g. launching a startup venture, financial markets/investments, or brand marketing."
        elif is_creative:
            stage_focus = "Deepen what excites them about creative arts — e.g. creative literature, storytelling, fine arts, or graphic illustration. Do NOT mention games, apps, or coding."
        elif is_healthcare:
            stage_focus = "Deepen what excites them in science/health — e.g. clinical patient care, laboratory research, or biotechnology."
        else:
            stage_focus = "Deepen what excites them about technology — e.g. software coding, AI algorithms, physical robotics, or tech systems."
    elif turn_num <= 4:
        if is_business:
            stage_focus = "Ask how they like to tackle business challenges — understanding customer needs, analyzing financial data, or creative marketing."
        elif is_creative:
            stage_focus = "Ask how they approach creative projects — spontaneous intuitive expression or structured planning and message development. Do NOT mention video games."
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
        stage_focus = "Ask what real-world domain (like education, climate, productivity, or interactive media) they would love to build technology for."

    # ── GUARD 1: Negation & Disavowal Detection ("not into gaming", "hate coding", etc.) ──
    is_negated_gaming = any(re.search(pat, u_clean) for pat in [
        r'\b(?:not\s+into|not\s+interested|dont\s+like|don\'t\s+like|hate|dislike|never\s+play|not\s+a\s+fan|no)\s+(?:gaming|games|game|video\s*games)\b',
        r'\bnot\s+into\s+gaming\b',
        r'\bno\s+gaming\b'
    ])
    is_negated_coding = any(re.search(pat, u_clean) for pat in [
        r'\b(?:not\s+into|not\s+interested|dont\s+like|don\'t\s+like|hate|dislike|no)\s+(?:coding|code|programming|software|apps)\b'
    ])
    is_negated_math = any(re.search(pat, u_clean) for pat in [
        r'\b(?:not\s+into|not\s+interested|dont\s+like|don\'t\s+like|hate|dislike|no)\s+(?:math|maths|numbers)\b'
    ])

    if is_negated_gaming:
        if is_creative or any(w in user_history_lower for w in ["creative", "art", "writing", "design", "arts"]):
            return "Understood, ruling out gaming and apps completely! Focusing strictly on your creative interests: do you lean more towards writing (like stories, articles, and literature) or visual arts (like sketching, painting, and graphic illustration)?"
        elif is_business:
            return "Got it, leaving gaming and tech aside! In business: what kind of challenges excite you more — marketing, branding, and customer strategy, or financial planning and management?"
        else:
            return "Understood, ruling out gaming completely! What kind of real-world activities do you enjoy more: creative visual arts/writing, scientific experiments, or organizing business ideas?"

    if is_negated_coding:
        return "Got it, leaving coding and software aside! What feels more exciting to you: hands-on mechanical hardware, creative design and arts, or real-world business and management?"

    if is_negated_math:
        return "Understood, no heavy mathematical theory! Would you rather explore hands-on creative arts and media, or practical commerce, law, and communication?"

    # ── GUARD 2: Confusion & Non-Comprehension ("i did not get it", "didn't understand", "meaning?") ──
    confusion_patterns = [
        r'\b(?:did\s*not\s*get|didn\'?t\s*get|not\s*get\s*it|did\s*not\s*understand|didn\'?t\s*understand)\b',
        r'\b(?:what\s*do\s*you\s*mean|what\s*does\s*that\s*mean|what\s*meaning|confused|confusing|not\s*clear)\b',
        r'^(?:what|huh|pardon|meaning\??|mean\??|what\s*meaning\??|what\s*does\s*it\s*mean\??|\?+)$'
    ]
    is_confusion = any(re.search(pat, u_clean) for pat in confusion_patterns)
    if is_confusion:
        if is_creative or any(w in user_history_lower for w in ["creative", "art", "writing", "design", "arts"]):
            return "No worries at all, let's keep it super simple! When you think of creative work: do you prefer expressing yourself through words and stories, or through visual art like drawing and design?"
        elif is_business:
            return "No worries at all, let's make it super simple! In business: do you enjoy leading projects and talking to people, or working with numbers, money, and planning?"
        elif is_healthcare:
            return "No worries at all, let's simplify! In science: are you more curious about human biology and medicine, or laboratory research?"
        else:
            return "No problem at all, let's take a step back! In your free time, what kind of activities feel most natural to you: creative visual arts, exploring tech gadgets, or organizing and chatting with friends?"

    # ── GUARD 3: Turn 2 Guaranteed First-Step Domain Alignment ──────────────
    if turn_num == 2:
        if any(w in u_clean for w in ["creative", "writing", "art", "arts", "design", "magazine", "literature", "draw"]):
            return "Wonderful! Within the creative arts and writing: do you feel more drawn to storytelling and writing (like fiction, scripts, and articles), or visual expression (like sketching, painting, and graphic design)?"
        elif any(w in u_clean for w in ["software", "tech", "tools", "code", "coding", "program"]):
            return "Awesome! When thinking about technology: does your curiosity lean more towards writing code and building software logic, or designing how users interact with the visual interface?"
        elif any(w in u_clean for w in ["science", "lab", "experiment", "biology", "chemistry"]):
            return "Fascinating! In scientific discovery: do you lean more towards biological sciences and healthcare, or physics, chemistry, and experimental laboratory research?"
        elif any(w in u_clean for w in ["business", "finance", "marketing", "trade", "stall", "commerce"]):
            return "Great! In the world of business and economics: are you more excited by marketing, branding, and launching new ideas, or analyzing financial data, investments, and market trends?"

    # ── GUARD 4: Explicit Student Vulnerability & Progressive Self-Doubt ──────
    doubt_keywords = [
        "what i am good at", "what im good at", "not good at", "no skills",
        "dont know what i am", "don't know what i am", "clueless",
        "not sure what i like", "dont know what to do", "don't know what to do",
        "dont know", "don't know", "no idea", "nothing", "idk",
        "i have no idea", "not sure"
    ]
    is_self_doubt = any(k in user_reply.lower() for k in doubt_keywords) or is_uncertain
    past_doubts = sum(1 for m in history if m["role"] == "user" and any(k in m["content"].lower() for k in doubt_keywords))
    
    if is_self_doubt:
        # If student says "i don't know" a second time, do NOT repeat the same paragraph!
        # Give them agency to see recommendations directly or try an everyday example.
        if past_doubts >= 1:
            return (
                "No pressure at all! We don't have to keep guessing with questions. "
                "Would you like to explore your initial recommended career pathways right now so you can see what fits, "
                "or would you rather test out a quick real-world example?"
            )

        if is_creative or any(w in user_history_lower for w in ["creative", "writing", "art", "arts", "design"]):
            return (
                "That is completely okay! You don't need to know what you're 'good at' right now — "
                "discovering what clicks for you is why we're exploring together without any pressure. "
                "In your free time, do you find yourself drawn to visual things like sketches, videos, or stories, "
                "or exploring other hobbies?"
            )
        elif is_business or any(w in user_history_lower for w in ["business", "finance"]):
            return (
                "That is completely normal! No one expects you to have it all figured out yet — "
                "we'll take it one step at a time with zero pressure. "
                "When you have free time, do you enjoy brainstorming ideas and chatting with people, "
                "or do you prefer quiet, independent problem solving?"
            )
        elif is_healthcare or any(w in user_history_lower for w in ["medic", "bio", "doctor"]):
            return (
                "That is completely okay! Figuring out your natural strengths takes time and experimentation — there is zero rush. "
                "In daily life, do you find yourself curious about how living things work and helping people, or practical laboratory experiments?"
            )
        else:
            return (
                "That is completely okay and normal! You don't have to have your strengths figured out right now — "
                "figuring out what you naturally enjoy is why we're having this chat without pressure. "
                "Let's make it super simple: do you enjoy creative visuals, doodling, and storytelling, "
                "or exploring gadgets, tools, and logic puzzles?"
            )

    is_positive_gaming = (not is_negated_gaming) and any(k in user_reply.lower() for k in ["gaming", "gamer", "play games", "playing games", "video games"])

    # Prompt design for Gemma 2B: Direct, positive, without negative constraint leakage
    if is_positive_gaming:
        prompt = f"""You are CareerSaathi, a friendly, smart career counselor talking with an Indian student ({education_level}).
Recent conversation:
{recent_context}
The student mentioned enjoying gaming: "{resolved_reply}".

Ask ONE engaging question about what part of games engages them most — the visual design & storylines, the competitive team strategy, or wondering how the game engine and mechanics were programmed.
Keep it conversational, relatable, and under 25 words."""

    else:
        prompt = f"""You are CareerSaathi, an empathetic, smart career mentor for Indian students ({education_level}, {stream}).
Recent conversation:
{recent_context}
Student: "{resolved_reply}"
Goal: {stage_focus}

Respond naturally like a friendly guide.
1. Be direct and conversational. Do not repeat what they just said.
2. Ask ONE engaging, open question advancing the conversation towards their goal.
3. Ground your question in relatable Indian student experiences (e.g., school exhibition projects, literature, creative arts, gadgets, clubs, local ventures). Never use foreign clichés like 'lemonade stands'.
4. If discussing creative arts, DO NOT mention video games, apps, or coding.
5. Keep it under 30 words. No multiple choice, bullets, or lectures."""

    response = query_local_gemma(prompt, timeout=25, max_tokens=80)
    
    # Intercept foreign clichés like lemonade stands
    if response and "lemonade" in response.lower():
        response = "Have you ever thought about what kind of creative or tech project you'd enjoy building — like a magazine exhibit, a school club showcase, or a visual design piece?"

    # Intercept repetition of solo vs team if already discussed
    if response and team_already_discussed and any(w in response.lower() for w in ["solo", "collaborat", "work alone"]):
        if is_business:
            response = "What industry or customer challenge would you be most excited to build a business or venture around?"
        elif is_creative:
            response = "When you create stories or artwork, do you lean more towards digital media (like digital illustration and animation) or written literature?"
        else:
            response = "What real-world problem would you be most excited to solve — like education, healthcare, or daily productivity?"

    # Intercept repetition of backend vs UI if already discussed
    if response and backend_logic_discussed and any(w in response.lower() for w in ["intricate logic", "backend or", "user interface or", "algorithms or crafting", "visual appeal"]):
        response = "Between building systems that solve everyday problems for millions of users or designing intelligent algorithms, which feels more inspiring?"

    # ── GUARD 5: Output Guardrail against Rogue Game/App Injections in Creative Track ──
    if (is_creative or any(w in user_history_lower for w in ["creative", "writing", "art"])) and not any(k in user_history_lower for k in ["gaming", "gamer"]):
        if response and any(w in response.lower() for w in ["game level", "design a game", "game or app", "app idea", "video game", "coolest game"]):
            if turn_num in [3, 4]:
                response = "When you work on a creative piece, what feels most natural to you — experimenting intuitively with visual ideas and colors, or planning out a structured storyline and message first?"
            else:
                response = "In your creative work, do you lean more towards digital media (like animation and digital illustration) or written literature?"

    if is_negated_gaming and response and any(w in response.lower() for w in ["game", "gaming", "game level"]):
        response = "Leaving games and tech aside: what kind of creative medium would you love to spend more time exploring — creative literature and writing, or visual design and fine arts?"

    # ── GUARD 6: Anti-Redundancy & Repetition Breaker ────────────────────────
    # Strictly ensure that no question or thematic prompt previously asked is ever repeated!
    def already_asked(q_text: str) -> bool:
        if not q_text:
            return False
        q_low = q_text.lower()
        for m in history:
            if m["role"] == "assistant":
                p_low = m["content"].lower()
                for key_phrase in [
                    "visual project or creative medium",
                    "creative medium are you most excited",
                    "solo, or do you thrive in a group",
                    "school festival poster",
                    "accomplishment when you saw your art",
                    "what kind of change do you want your art",
                    "design a game or app"
                ]:
                    if key_phrase in q_low and key_phrase in p_low:
                        return True
                if len(q_low) > 35 and q_low[:35] in p_low:
                    return True
        return False

    if already_asked(response):
        if is_creative:
            response = "Between designing visual graphics and media, or writing deep narrative stories, which one feels closest to what you'd love doing long-term?"
        elif is_business:
            response = "In a business venture, would you feel more energized pitching strategy to clients, or managing detailed financial budgets and operations?"
        elif is_healthcare:
            response = "In healthcare, would you rather focus on direct clinical patient care, or laboratory research and diagnostics?"
        else:
            response = "Between building software systems that serve millions of users or designing hands-on smart gadgets, which inspires you more?"

    # Gaming Fail-Safe
    if is_positive_gaming and (not response or len(response.strip()) < 15):
        return "When you play games, what hooks you most: the visual art and storylines, the competitive team strategy, or wondering how the game engine and mechanics were coded?"

    # General Fail-Safe Fallback
    if not response or len(response.strip()) < 15:
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
# 4. Warm Conversational Onboarding & Academic Score Acknowledgment
# ─────────────────────────────────────────────────────────────────────────────

def get_time_greeting() -> str:
    """Returns a time-aware greeting based on local system time."""
    current_hour = datetime.datetime.now().hour
    if current_hour < 12:
        return "Good morning"
    elif current_hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


def get_initial_onboarding_inquiry(edu_level: str) -> str:
    """
    Warm time-aware onboarding message collecting marks and competitive exam status upfront.
    """
    greeting = get_time_greeting()
    is_10th = "10" in (edu_level or "").lower()
    if is_10th:
        return (
            f"{greeting}! Welcome to **CareerSaathi** — your personalized AI career & admissions counselor.\n\n"
            f"Before we dive into your counselling and find the perfect stream for you, "
            f"what are your approximate **10th marks or percentage**? "
            f"Also, are you preparing for or considering any competitive streams (like Science with JEE/NEET, Commerce with CA, or Humanities)?"
        )
    else:
        return (
            f"{greeting}! Welcome to **CareerSaathi** — your personalized AI career & admissions counselor.\n\n"
            f"Before we dive into your counselling and map out your best-fit university degrees, "
            f"what are your approximate **10th / 12th board marks or percentage**, "
            f"and have you appeared for (or are preparing for) any competitive exams like JEE, NEET, CUET, CET, or CLAT?"
        )


def get_academic_score_acknowledgment(user_input: str, score_tier: str, score_raw: str, edu_level: str) -> str:
    """
    Empathetically acknowledges the student's score range as standard formality,
    reassuring them that counseling recommendations are 100% aptitude- and interest-first.
    """
    tier = (score_tier or "passion_first").lower()
    raw = score_raw.strip() if score_raw else ""
    score_display = f" ({raw})" if raw else ""
    
    if tier == "high":
        return f"That's a fantastic academic foundation{score_display}! We'll factor your score into your final admissions reality check, but for our session today, **we are exploring 100% based on your natural curiosities, thinking instincts, and strengths with zero pressure**."
    elif tier == "competitive":
        return f"That's a very solid score{score_display}! We'll keep your marks in mind for cutoff requirements and admissions gateways later, but for our counseling today, **we are exploring purely based on your genuine interests and cognitive style**."
    elif tier == "moderate":
        return f"Thank you for sharing honestly{score_display}! Board percentages are just one dimension for admissions logistics, but for our advisory today, **we are exploring 100% based on what genuinely excites you and your natural problem-solving instincts**."
    elif tier == "foundational":
        return f"Thank you for being open and honest{score_display}! Academic board marks are only one metric for admissions logistics — what truly defines your long-term career success are your real-world curiosities and practical skills, which is what we will focus on today!"
    else:
        return "Wonderful! Exploring from your genuine curiosities and natural thinking instincts with zero pressure is the most rewarding way to choose a direction."


# ─────────────────────────────────────────────────────────────────────────────
# 5. Opening Scenario Generator (Turn 1 Icebreaker)
# ─────────────────────────────────────────────────────────────────────────────

def get_opening_scenario(stream: str, edu_level: str) -> str:
    """
    Provides a balanced, dynamic scenario grounded in realistic student life,
    specifically adapted for After 10th (Stream selection) vs After 12th (Degree selection).
    Draws from an icebreaker pool so opening turns feel fresh and conversational every time.
    """
    stream_clean = stream.lower() if stream else ""
    edu_clean = edu_level.lower() if edu_level else ""
    
    # After 10th Standard: Choosing a Stream (Dynamic Pool)
    if "10" in edu_clean or "stream" in stream_clean:
        scenarios_10th = [
            "To help work out which stream fits how you think: imagine you have complete freedom to choose a project for a school exhibition. Would you rather build a working tech/science project, run a commercial trade stall, or design a creative magazine/exhibit?",
            "Welcome! To explore what stream fits your natural style of thinking: when you have free time on the weekend, what kind of activities do you find yourself drawn to — exploring tech tools and gadgets, experimenting with science, managing practical business ideas, or creative arts and writing?",
            "Welcome! Imagine your school invites you to lead an initiative: would you rather develop a technical/scientific solution, organize a student-run commercial marketplace, or direct a creative media and storytelling campaign?"
        ]
        return random.choice(scenarios_10th)
        
    # After 12th: Specific Stream or Degree (Dynamic Pool)
    if "pcm" in stream_clean or "math" in stream_clean:
        pcm_scenarios = [
            "Welcome! Imagine you are invited to join an inter-college technical expo. Would you rather write the software algorithms and build the app, design and assemble the physical robotics hardware, or analyze mathematical data models?",
            "Welcome! As you plan your engineering journey: what challenges excite you most — writing scalable software code and AI, engineering physical machines and aerospace vehicles, or designing smart architecture and infrastructure?"
        ]
        return random.choice(pcm_scenarios)
    elif "pcb" in stream_clean or "bio" in stream_clean:
        pcb_scenarios = [
            "Welcome! Imagine you are spending a day at an innovation science center. Would you rather investigate cellular biology and genetic tests in a lab, explore medical healthcare and patient therapy, or study ecological and plant sciences?",
            "Welcome! In the healthcare and biological world: do you feel most inspired by clinical patient care in a hospital, discovering new pharmaceutical drugs in a lab, or studying genetics and biotechnology?"
        ]
        return random.choice(pcb_scenarios)
    elif "commerce" in stream_clean:
        comm_scenarios = [
            "Welcome! Imagine you are working on a new business project. Would you rather create marketing campaigns and negotiate partnership deals, analyze financial data and investment budgets, or study economic market trends?",
            "Welcome! In the world of business and finance: would you rather launch an innovative startup venture, master the stock markets and investment banking, or ensure rigorous corporate auditing and financial accounting?"
        ]
        return random.choice(comm_scenarios)
    elif "arts" in stream_clean or "liberal" in stream_clean:
        arts_scenarios = [
            "Welcome! Imagine you are contributing to a creative cultural initiative. Would you rather write original stories and articles, design visual graphics and digital media, or explore human psychology and behavioral counseling?",
            "Welcome! In the liberal arts and design landscape: does your mind lean towards designing digital product UI/UX and visual branding, advocating for legal rights and corporate law, or understanding human psychology?"
        ]
        return random.choice(arts_scenarios)
    else:
        general_scenarios = [
            "To discover your ideal university degree: when you think about spending the next four years diving deep into a subject, does your mind naturally lean towards solving algorithmic logic and technology, investigating natural/medical sciences, managing business/finance, or creative design and humanities?",
            "Welcome! As you prepare for your undergraduate journey: what kind of real-world impact excites you most — building software and technology systems, diagnosing biological/health problems, leading corporate business ventures, or designing creative visual experiences?",
            "Welcome! Imagine you have complete academic freedom in college: would you rather immerse yourself in high-tech engineering and computational logic, experimental laboratory research and healthcare, strategic business and investments, or creative arts and design?"
        ]
        return random.choice(general_scenarios)

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


# ─────────────────────────────────────────────────────────────────────────────
# 6. Broad Horizon Discovery Protocol (Turn 15 Empathetic Closure)
# ─────────────────────────────────────────────────────────────────────────────

def get_broad_horizon_closure_message(education_level: str, is_10th: bool, recs: Dict, user_text: str = "") -> str:
    """
    Compassionate, authoritative, and validating closure message for students who
    have reached the exploration cap without locking into a narrow traditional specialty.
    Normalizes being undecided, explains why premature specialization is an ethical mistake,
    and presents 3 high-optionality gateway pathways.
    """
    rec1 = recs["primary"]
    rec2 = recs["secondary"]
    rec3 = recs["tertiary"]
    r1_title = rec1.get("title", rec1["degree"])
    r2_title = rec2.get("title", rec2["degree"])
    r3_title = rec3.get("title", rec3["degree"])
    
    closure = (
        f"🌱 **A Personalized Note from your CareerSaathi Counselor**:\n\n"
        f"I've really valued our conversation! Throughout our chat, I noticed that none of the rigid, conventional academic boxes seemed to immediately jump out at you — and I want to reassure you: **that is 100% normal and completely okay at this stage.**\n\n"
        f"You don't have an aptitude issue; you simply haven't encountered the specific project or subject that triggers your passion yet. "
        f"Because of that, **forcing you into a narrow, hyper-specialized career right now would be an ethical mistake.**\n\n"
        f"Instead, the most strategic decision is **Maximum Optionality**: choosing a broad gateway pathway that keeps 85%+ of career doors open while giving you the next 2 to 3 years to discover what you love.\n\n"
        f"Here is your **Broad Horizon Discovery Portfolio** (curated for versatility & low pressure):\n\n"
        f"1. **{r1_title}** (`{rec1['fit_score']}% Exploratory Fit`) — *Keeps maximum engineering, tech, and analytical options open*\n"
        f"2. **{r2_title}** (`{rec2['fit_score']}% Exploratory Fit`) — *Versatile foundation in applied business, economics, and real-world strategy*\n"
        f"3. **{r3_title}** (`{rec3['fit_score']}% Exploratory Fit`) — *Creative and multidisciplinary freedom without rigid silos*\n\n"
        f"I have prepared your full **Broad Horizon Discovery Report** in the side pane. "
        f"Rather than stressful entrance exam cutoffs, it includes **low-stakes weekend micro-experiments** to help you test what excites you hands-on!"
    )
    return closure
