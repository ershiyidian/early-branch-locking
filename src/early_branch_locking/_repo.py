from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RLVR_ROOT = REPO_ROOT
DATA_ROOT = REPO_ROOT / "data"
COUNTDOWN_DATA_ROOT = DATA_ROOT / "analysis_results" / "rlvr_passk"
RAW_DIR = COUNTDOWN_DATA_ROOT / "raw"
METRICS_DIR = COUNTDOWN_DATA_ROOT / "metrics"
FIGURES_DIR = COUNTDOWN_DATA_ROOT / "figures"
DATASET_DIR = REPO_ROOT / "dataset"
TEST_PARQUET = REPO_ROOT / "dataset" / "test.parquet"
MODEL_DIR = REPO_ROOT / "model"
EXTERNAL_DIR = REPO_ROOT / "external"
MATH_DATASET_DIR = DATASET_DIR / "math_eval"
COUNTDOWN_ACTOR_DIR = REPO_ROOT / "checkpoints" / "TinyZero" / "countdown-qwen2.5-3b" / "actor"
RLVR_DATA_ROOT = DATA_ROOT / "rlvr"
MATH_OUTPUTS_DIR = RLVR_DATA_ROOT / "outputs"
MATH_LOGS_DIR = RLVR_DATA_ROOT / "logs"
MATH_PLOTS_DIR = RLVR_DATA_ROOT / "plots"
OPD_LOG_DIR = MATH_LOGS_DIR / "opd"
C2F_OUTPUT_DIR = MATH_OUTPUTS_DIR / "experiments" / "c2f"
C2F_RAW_DIR = C2F_OUTPUT_DIR / "raw"
C2F_METRICS_DIR = C2F_OUTPUT_DIR / "metrics"
PAPER_ROOT = REPO_ROOT / "paper" / "iclr2027_early_branch_locking"
PAPER_FIGURES_DIR = PAPER_ROOT / "figures"


def ensure_repo_root_on_syspath() -> Path:
    """Allow every canonical entry point to run from any working directory."""
    root = REPO_ROOT
    module_dir = REPO_ROOT
    for path in (module_dir, root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root
