#!/usr/bin/env python
"""Télécharge le modèle MoveNet SinglePose Thunder (ONNX) dans models/.

Source : Xenova/movenet-singlepose-thunder (Apache-2.0, opset 11, ~25 Mo).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/Xenova/movenet-singlepose-thunder/resolve/main/onnx/model.onnx"
EXPECTED_SIZE = 25_067_197
DEST = Path(__file__).parent / "models" / "movenet_thunder.onnx"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size == EXPECTED_SIZE:
        print(f"Déjà présent : {DEST} ({DEST.stat().st_size} octets)")
        return 0
    print(f"Téléchargement -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    size = DEST.stat().st_size
    if size != EXPECTED_SIZE:
        print(f"⚠ Taille inattendue : {size} (attendu {EXPECTED_SIZE}).", file=sys.stderr)
        return 1
    print(f"OK ({size} octets).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
