"""Pytest bootstrap: put src/ on the path so `pytest tests/` works from the
delivery-package root without any PYTHONPATH juggling.

The notebooks each keep local copies of the .py modules (so every notebook
folder runs standalone); the test suite imports the canonical copies in src/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# learner/ on path so the harness can import the student-completed sahayak_starter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "learner"))
