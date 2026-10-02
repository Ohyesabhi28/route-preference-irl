"""
run_visualise.py – Generate all report figures after pipeline has run.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from visualise import generate_all_figures
generate_all_figures()
