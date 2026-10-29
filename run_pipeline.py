"""
run_pipeline.py – One-shot runner: train IRL + evaluate.
Region: set IRL_REGION=porto (default) or sf.
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == "__main__":      # guard needed: training uses multiprocessing (spawn on Windows)
    from train import main
    main()
