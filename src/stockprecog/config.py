"""Configuração central do pipeline. Single source of truth de hiperparâmetros."""
from __future__ import annotations

from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

for _d in (DATA_DIR, RAW_DIR, PROC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- universo (composição ATUAL — survivorship bias conhecido, ver README) ---
# Subconjunto líquido do IBrX. Será trocado por composição point-in-time depois.
UNIVERSE: list[str] = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "B3SA3", "ABEV3", "WEGE3",
    "RENT3", "SUZB3", "ITSA4", "JBSS3", "RAIL3", "PRIO3", "GGBR4", "BPAC11",
    "RADL3", "EQTL3", "VBBR3", "LREN3", "CSAN3", "HAPV3", "TOTS3", "CMIG4",
    "UGPA3", "ELET3", "EMBR3", "BRFS3", "CPLE6", "NTCO3",
]

# Ativos com ajuste NÃO-CONFIÁVEL: ausentes da base histórica da brapi (buraco do
# lado deles, confirmado por 3 endpoints; não é quota). Só temos fallback
# COTAHIST+dividendos, que carrega gap de bonificação. NÃO usar no modelo até ter
# fonte ajustada confiável. Ver reference-b3-data-idiosyncrasies.
UNRELIABLE: list[str] = ["BRFS3", "CPLE6"]

# Universo efetivamente usável pelo pipeline.
USABLE_UNIVERSE: list[str] = [t for t in UNIVERSE if t not in UNRELIABLE]

START = "2010-01-01"
END = "2026-06-15"

# --- triple-barrier labeling ---
PT_MULT = 2.0       # profit-take em múltiplos de sigma
SL_MULT = 2.0       # stop-loss em múltiplos de sigma
HORIZON = 10        # barreira vertical (dias úteis)
VOL_SPAN = 20       # span EWMA pra vol diária

# --- validação temporal ---
EMBARGO_DAYS = HORIZON  # embargo >= horizon pra não vazar label do triple-barrier
TEST_FRACTION = 0.30    # último trecho temporal como teste no loop mínimo

SEED = 42
