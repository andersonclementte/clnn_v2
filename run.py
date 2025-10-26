#!/usr/bin/env python3
"""
Script principal do HuMob Challenge - ponto de entrada único.
"""

import sys
from pathlib import Path

# Adiciona src ao Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importa e executa o script principal
if __name__ == "__main__":
    from scripts.train import main
    main()
