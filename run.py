#!/usr/bin/env python3
"""Launcher. Loads API keys from keys.local.py (created for you on first run),
then opens the Streamlit chat in your browser.

Usage:  python run.py

On the very first run this creates keys.local.py from the template and asks you
to paste your ANTHROPIC_API_KEY into it. keys.local.py is git-ignored and never
shipped, so your keys stay local.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import keys

keys.apply()

_backend = os.environ.get("QA_LLM_BACKEND", "anthropic").strip().lower() or "anthropic"
if _backend == "anthropic":
    keys.require("ANTHROPIC_API_KEY")
else:
    _base = os.environ.get("QA_LLM_BASE_URL", "http://127.0.0.1:8645/v1")
    _model = os.environ.get("QA_MODEL", "")
    print(f"LLM backend: {_backend} via {_base}")
    if not _model:
        print(
            "Note: QA_MODEL is not set, so the default 'claude-sonnet-4-6' will be\n"
            "sent to the endpoint. Set QA_MODEL in keys.local.py to a model your\n"
            "endpoint actually serves (e.g. 'Hermes-4-405B' for `hermes proxy`;\n"
            "for the Hermes API server the model field is cosmetic)."
        )

print("Keys loaded from keys.local.py. Launching Streamlit chat (Ctrl+C to stop)...")
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")],
    env=os.environ.copy(),
)
