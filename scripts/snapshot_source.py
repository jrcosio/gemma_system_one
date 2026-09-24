"""Copia exacta de las fuentes bajo su hash (``env.snapshot_sources``), sin necesidad de commit.

Uso: uv run python scripts/snapshot_source.py [directorio]   (por defecto artifacts/source)
Desde la fase 5 también se hace automáticamente al empezar cada run y al guardar cada checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from gemma_system_one.env import snapshot_sources

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(snapshot_sources(out_dir=out), indent=2))
