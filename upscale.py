"""Entry point leve para executar o backend."""
import sys
from pathlib import Path

# Garante que o diretório raiz do projeto esteja no sys.path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.app import start_backend

if __name__ == "__main__":
    start_backend()
