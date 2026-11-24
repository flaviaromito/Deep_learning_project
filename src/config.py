"""Lettura del file di configurazione config.yaml (nella cartella principale del progetto).

Uso:
    from src.config import load_config
    cfg = load_config()
    lr = cfg["training"]["learning_rate"]
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path=PROJECT_ROOT / "config.yaml"):
    """Legge il file YAML e lo restituisce come dizionario."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
