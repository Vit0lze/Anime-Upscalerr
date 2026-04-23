"""Bootstrap curto para Google Colab.

Uso no Colab:
!python colab/bootstrap_backend.py --repo https://github.com/<org>/<repo>.git --ref main
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="URL Git do projeto")
    parser.add_argument("--ref", default="main", help="Branch/tag/commit")
    parser.add_argument("--dest", default="/content/backend", help="Pasta destino (deve terminar em 'backend')")
    args = parser.parse_args()

    dest = Path(args.dest)
    if dest.exists():
        run(["rm", "-rf", str(dest)])

    run(["git", "clone", "--depth", "1", "--branch", args.ref, args.repo, str(dest)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=dest)
    # Rodar do diretório PAI para que 'backend' seja importável como pacote
    run([sys.executable, str(dest / "upscale.py")], cwd=str(dest.parent))


if __name__ == "__main__":
    main()
