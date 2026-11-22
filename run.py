import os
import sys
import time
import threading
import webbrowser
import requests
import uvicorn

def open_browser():
    """
    Waits 1.5 seconds for the server to spin up, then opens default browser.
    """
    time.sleep(1.5)
    url = "http://127.0.0.1:8080"
    print(f"\n[Launcher] Opening browser automatically to {url} ...")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 80)
    print("  CAREERSAATHI: COGNITIVE ADMISSIONS & CAREER ADVISORY SUITE")
    print("  Empirical NLP & Explainable AI (XAI) • 36 Comprehensive Academic Pathways")
    print("=" * 80)
    
    # Add backend directory to sys.path
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    if backend_path not in sys.path:
        sys.path.append(backend_path)
        
    start_time = time.time()
    try:
        print("  [NLP Pipeline] Initializing Linguistic Engine & Semantic Embedders...")
        import riasec
        import nlp_engine
        import generator
        elapsed = time.time() - start_time
        
        print(f"  [NLP Pipeline] [OK] Models compiled & ready in {elapsed:.2f} seconds!")
        print(f"  [NLP Pipeline] [OK] Holland RIASEC Codes: 6 Centroids in R^384 dense space")
        print(f"  [NLP Pipeline] [OK] TF-IDF Sparse Vocabulary: V = {nlp_engine.vocab_size} dimensions")
        print(f"  [NLP Pipeline] [OK] Random Forest Ensemble: {nlp_engine.rf_classifier.n_estimators} trees fitted")
        print(f"  [NLP Pipeline] [OK] Decision Tree XAI Traversal: max_depth={nlp_engine.dt_classifier.max_depth}")
        
        # Test local Ollama connection
        ollama_status = "OFFLINE (Deterministic Fallback Ready)"
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                ollama_status = f"ONLINE (Local Models: {', '.join(models)})"
        except Exception:
            pass
        print(f"  [Local Generation] [OK] Ollama Engine: {ollama_status}")
        print("-" * 80)
    except Exception as e:
        print(f"  [NLP Pipeline] [ERROR] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("  [Server] Mounting FastAPI Web App & Glassmorphism Static Assets...")
    print("  [Server] Running locally at: http://127.0.0.1:8080")
    print("=" * 80)
    
    # Start auto-browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start uvicorn server
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8080, log_level="info")
