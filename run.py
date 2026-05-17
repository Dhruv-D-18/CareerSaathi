import os
import sys
import time
import threading
import webbrowser
import uvicorn

def open_browser():
    """
    Waits 1.5 seconds for the server to start, then opens the user's default browser.
    """
    time.sleep(1.5)
    url = "http://127.0.0.1:8080"
    print(f"\n[Launcher] Opening browser automatically to {url} ...")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 75)
    print("  TRADITIONAL ML CAREER COUNSELLOR - UNIVERSITY LAB PRACTICAL")
    print("=" * 75)
    print("  [ML Pipeline] Bootstrapping Synthetic Dataset & Training...")
    
    # Add backend to path so we can import model and main
    sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
    
    start_time = time.time()
    try:
        import model
        elapsed = time.time() - start_time
        
        # Display model stats on terminal
        print(f"  [ML Pipeline] [OK] Successfully trained classifiers in {elapsed:.3f} seconds!")
        print(f"  [ML Pipeline] [OK] Training samples generated: {len(model.X_train)}")
        print(f"  [ML Pipeline] [OK] TF-IDF Vocabulary dimension: {len(model.vectorizer.vocabulary_)}")
        print(f"  [ML Pipeline] [OK] Random Forest Ensembles: {model.rf_classifier.n_estimators} estimators")
        print(f"  [ML Pipeline] [OK] Decision Tree depth restriction: max_depth={model.dt_classifier.max_depth}")
        print("-" * 75)
    except Exception as e:
        print(f"  [ML Pipeline] [ERROR] Error training models: {e}")
        sys.exit(1)
        
    print("  [Server] Initializing FastAPI Web App & Static Files mount...")
    print("  [Server] Running locally at: http://127.0.0.1:8080")
    print("=" * 75)
    
    # Start auto-browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Start uvicorn server
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8080, log_level="info")
