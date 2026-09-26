"""Paths and knobs. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", HOME / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", HOME / "er_work"))
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", HOME / "student_resource" / "output"))

WORKERS = int(os.environ.get("ER_WORKERS", os.cpu_count() or 4))
TOP_K = int(os.environ.get("ER_TOP_K", 30))
CHUNK = int(os.environ.get("ER_CHUNK", 150_000))
TRAIN_SAMPLE = int(os.environ.get("ER_TRAIN_SAMPLE", 200_000))
VAL_SAMPLE = int(os.environ.get("ER_VAL_SAMPLE", 100_000))
ROUNDS = int(os.environ.get("ER_ROUNDS", 1000))
SEED = 42

# Blocking key types: name -> (weight, cap). A key shared by more than `cap`
# Source 2/3 records is dropped. See blocking.py for what each key contains.
KEY_TYPES = {
    "A": (3, 200),  # two rarest skeleton name tokens
    "B": (3, 200),  # rarest name token + postcode
    "C": (2, 200),  # rarest name token + rarest address token
    "D": (2, 100),  # first 8 chars of concatenated core name
    "E": (1, 50),   # rarest name token alone
    "F": (2, 50),   # postcode + house number
    "G": (2, 100),  # rarest address token + house number
}

THRESH_GRID = [round(0.20 + 0.025 * i, 3) for i in range(31)]  # 0.20 .. 0.95

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "binary_logloss",
    "seed": SEED,
    "verbose": -1,
}
