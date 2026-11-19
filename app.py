"""
BinSense — Cloud Production Entrypoint
Ensures uniform imports and path resolution when deployed on Streamlit Cloud,
Render, Hugging Face Spaces, Railway, or Docker.
"""

import os
import sys

# Resolve absolute paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BINSENSE_DIR = os.path.join(ROOT_DIR, "binsense")

# Ensure binsense directory is in sys.path
if BINSENSE_DIR not in sys.path:
    sys.path.insert(0, BINSENSE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Switch working directory to binsense
os.chdir(BINSENSE_DIR)

app_path = os.path.join(BINSENSE_DIR, "app.py")

# Set __file__ so internal modules resolve relative paths accurately
g = globals()
g["__file__"] = app_path

with open(app_path, "r", encoding="utf-8") as f:
    app_code = compile(f.read(), app_path, "exec")
    exec(app_code, g)
