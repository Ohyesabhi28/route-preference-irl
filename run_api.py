"""
run_api.py – Start the Flask API server (also serves the dashboard).
Run AFTER run_pipeline.py has completed.
"""
import sys, os
# default to the best model (v3 features); override with IRL_FEATURES=v1|v2
os.environ.setdefault("IRL_FEATURES", "v3")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == "__main__":
    from api import app, _load_assets
    print("[Startup] Loading assets ...")
    _load_assets()
    print("[Startup] Dashboard: http://localhost:5000  (open this, not the .html file)")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
