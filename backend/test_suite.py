import sys
import os

sys.path.append(os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def run_test_scenario(name, level, turns):
    print("=" * 80)
    print(f"TEST SCENARIO: {name} ({level})")
    print("=" * 80)
    r = client.post("/api/chat", json={"action": "start", "level": level}).json()
    state = r["state"]
    print(f"Opening Bot Question: \"{r['question']}\"\n")
    
    for idx, user_msg in enumerate(turns, 1):
        r = client.post("/api/chat", json={**state, "latest_message": user_msg, "action": "chat"}).json()
        state = r["state"]
        bot_q = r["question"]
        stage = r.get("stage_name", "")
        print(f"[Turn {idx} - Stage: {stage}]")
        print(f"  User  : \"{user_msg}\"")
        print(f"  Saathi: \"{bot_q}\"\n")
    
    rep = r.get("report")
    if rep:
        print("-" * 80)
        print("FINAL DOSSIER RECOMMENDATION:")
        print(f"  Primary Title : {rep.get('primary_recommendation')}")
        print(f"  Stream        : {rep.get('primary_stream')}")
        print(f"  Fit Score     : {rep.get('primary_fit')}%")
        print(f"  Target Exams  : {', '.join(rep.get('target_entrance_exams', []))}")
        print(f"  Alternatives  : {', '.join(rep.get('alternatives', []))}")
        print("-" * 80)
    else:
        print("[WARNING] No dossier generated at this turn.")
    print("\n")

if __name__ == "__main__":
    # Test Case 1: 10th Pass -> Software & Mobile Apps (Science PCM)
    run_test_scenario(
        "10th Pass Student -> Software & Mobile Apps",
        "after_10th",
        [
            "Building software or tech tools",
            "I like how mobile apps work and solving computer logic",
            "Like automating daily tasks for people with clean code",
            "I prefer breaking down complex algorithms step by step",
            "Deep solo technical focus writing backend software"
        ]
    )

    # Test Case 2: 10th Pass -> Business, Marketing & Startups (Commerce)
    run_test_scenario(
        "10th Pass Student -> Business & Startups",
        "after_10th",
        [
            "Business, finance or marketing",
            "I enjoy business marketing campaigns, client negotiations and launching ventures",
            "When a project runs into chaos I reorganize the financial budget and negotiate",
            "Commercial market strategy, investment pitching and business growth",
            "Collaborative leadership, corporate operations and client management"
        ]
    )

    # Test Case 3: 12th Pass -> UI/UX Product Design (Liberal Arts / Design)
    run_test_scenario(
        "12th Pass Student -> Creative Digital UI/UX Design",
        "after_12th",
        [
            "UI/UX product design and arts",
            "I love visual design aesthetics, sketching user interfaces and digital prototyping",
            "I make visual wireframes, user journeys and creative layouts",
            "Typography, digital art styling, design thinking and creative storytelling",
            "Creative studio environment designing mobile app interfaces and graphics"
        ]
    )

    # Test Case 4: 12th Pass -> Biological Healthcare & Medical (PCB)
    run_test_scenario(
        "12th Pass Student -> Clinical Healthcare & Biology",
        "after_12th",
        [
            "Medical, biology and healthcare",
            "Investigating cellular genetics, microbiology and patient healing",
            "I carefully observe laboratory data and test biological hypotheses",
            "Human physiological anatomy, cellular immunology and healthcare diagnostics",
            "Clinical laboratory research, patient consultations and therapy"
        ]
    )
