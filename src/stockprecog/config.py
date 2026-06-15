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

# --- universo (líquido B3, scrape REST brapi: ADV>=R$3M, >=300 barras) ---
# 111 ativos (4x o subset inicial de 28). Inclui IPOs pós-2018 -> reduz survivorship
# vs pegar só sobreviventes de hoje. Composição point-in-time completa ainda pendente.
UNIVERSE: list[str] = [
    "ABCB4", "ABEV3", "ALOS3", "ALPA4", "AMER3", "ANIM3", "ASAI3", "AURE3",
    "AXIA3", "AZZA3", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BEEF3",
    "BHIA3", "BMOB3", "BRAP4", "BRAV3", "BRKM5", "BRSR6", "CASH3", "CBAV3",
    "CEAB3", "CMIG4", "CMIN3", "COGN3", "CPFE3", "CSAN3", "CSMG3", "CSNA3",
    "CURY3", "CVCB3", "CXSE3", "CYRE3", "DXCO3", "ECOR3", "EGIE3", "EMBJ3",
    "ENEV3", "EQTL3", "EZTC3", "FLRY3", "GGBR4", "GGPS3", "GMAT3", "GOAU4",
    "HAPV3", "HYPE3", "INTB3", "IRBR3", "ISAE4", "ITSA4", "ITUB3", "ITUB4",
    "LIGT3", "LJQQ3", "LOGG3", "LREN3", "LWSA3", "MDIA3", "MDNE3", "MGLU3",
    "MLAS3", "MOTV3", "MOVI3", "MRVE3", "MULT3", "MYPK3", "NATU3", "ONCO3",
    "ORVR3", "PCAR3", "PETR3", "PETR4", "PGMN3", "PLPL3", "POMO4", "PRIO3",
    "PSSA3", "QUAL3", "RADL3", "RAIL3", "RAPT4", "RDOR3", "RECV3", "RENT3",
    "SAUD3", "SBFG3", "SBSP3", "SEER3", "SIMH3", "SMFT3", "SMTO3", "SUZB3",
    "TEND3", "TIMS3", "TOTS3", "TTEN3", "TUPY3", "UGPA3", "UNIP6", "USIM5",
    "VALE3", "VAMO3", "VBBR3", "VIVA3", "VULC3", "WEGE3", "YDUQ3",
]

# Ativos com ajuste não-confiável (buraco da base brapi). Vazio agora — o universo
# acima já vem só de tickers com histórico ajustado válido no scrape REST.
UNRELIABLE: list[str] = []

# Universo efetivamente usável pelo pipeline.
USABLE_UNIVERSE: list[str] = [t for t in UNIVERSE if t not in UNRELIABLE]

START = "2010-01-01"
END = "2026-06-15"

# --- triple-barrier labeling ---
PT_MULT = 2.0       # profit-take em múltiplos de sigma
SL_MULT = 2.0       # stop-loss em múltiplos de sigma
HORIZON = 10        # barreira vertical (dias úteis)
VOL_SPAN = 20       # span EWMA pra vol diária

# --- fractional differentiation (cap.5) ---
FD_D = 0.4          # ordem de diferenciação (estacionário preservando memória)
FD_THRESH = 1e-3    # corte de peso FFD -> janela ~55 barras (era 282 c/ 1e-4)

# --- validação temporal ---
# Embargo cobre DOIS horizontes: (1) resolução do label (HORIZON=10) e (2) a memória
# da feature fracdiff (~55 barras). Sem o 2o, preços do teste vazam nas features de
# treino via convolução FFD (leak indireto pego na auditoria adversarial).
EMBARGO_DAYS = 55
TEST_FRACTION = 0.30    # último trecho temporal como teste no loop mínimo

SEED = 42
