"""
run_visualise.py – Generate all report figures after pipeline has run.
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == "__main__":
    from visualise import generate_all_figures
    generate_all_figures()
