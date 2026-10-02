"""
run_api.py – Start the Flask API server.
Run AFTER run_pipeline.py has completed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from api import app, _load_assets

print("[Startup] Loading assets …")
_load_assets()
print("[Startup] Starting server on http://localhost:5000")
app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
