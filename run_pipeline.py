"""
run_pipeline.py – One-shot runner: train IRL + evaluate.
Run this script first before starting the API server.
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from train import main
main()
