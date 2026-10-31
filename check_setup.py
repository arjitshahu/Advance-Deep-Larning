"""Quick checks before running ingestion or the UI."""
import importlib.util
import shutil
import sys

modules = ["streamlit", "fitz", "sentence_transformers", "faiss", "numpy", "requests", "pytest"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
print("Python:", sys.version.split()[0])
print("Ollama executable:", shutil.which("ollama") or "not found in PATH")
if missing:
    print("Missing Python modules:", ", ".join(missing))
    print("Run: pip install -r requirements.txt")
    raise SystemExit(1)
print("Python dependencies look installed.")
