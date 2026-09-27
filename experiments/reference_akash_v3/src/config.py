"""Paths and tunable parameters for the entity resolution pipeline.

Paths can be overridden with environment variables so the same code runs
on SageMaker, Kaggle or a laptop without edits.
"""
import os
from pathlib import Path

HOME = Path.home()
DATA_DIR = Path(os.environ.get("ER_DATA_DIR", HOME / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("ER_WORK_DIR", HOME / "er_work"))       # cached parquet, model
OUT_DIR = Path(os.environ.get("ER_OUT_DIR", HOME / "student_resource" / "output"))
N_WORKERS = int(os.environ.get("ER_WORKERS", os.cpu_count() or 4))

SEED = 42

# Blocking: each key type -> (weight in the first-stage score, max Source 2/3 block size).
# Blocks larger than the cap are dropped because they are too generic to be useful.
KEY_TYPES = {
    "A": (3.0, 1000),  # country + two rarest name tokens (skeleton form)
    "B": (3.0, 1000),  # country + rarest name token + postcode
    "C": (2.0, 1000),  # country + rarest name token + rarest address token
    "D": (2.0, 500),  # country + first 8 chars of the concatenated core name
    "E": (1.0, 300),  # country + rarest name token alone
    "F": (2.0, 300),   # country + postcode + house number
    "G": (2.0, 500),  # country + rarest address token + house number
    # v2: name-independent address channels. The team's labelled-noise profile found ~9% of
    # true links (15% India/S3) share no name token, while ~96% share an address token.
    "H": (2.0, 300),  # country + two rarest address tokens
    "I": (2.0, 300),   # country + postcode + rarest address token
    "J": (1.5, 300),  # country + second rarest address token + house number
    # v3: caps kept at v2 values; wider caps flooded the top-K with weak candidates in testing.
    # v3: position-based keys. In missed Indian pairs one address is a shortened copy of the
    # other, so the "rarest" tokens differ between the two sides; the first name token and
    # the last address token (usually the state) survive.
    "K": (2.0, 500),   # country + first skeleton name token + house number
    "L": (2.0, 500),   # country + house number + first letter of core name + last address token
}
TOP_K = int(os.environ.get("ER_TOP_K", 60))          # candidates kept per Source 1 entity
CHUNK = int(os.environ.get("ER_CHUNK", 75_000))      # Source 1 entities processed per chunk

# Training sample sizes (Source 1 entities); both are blocked against the full train S2/S3.
TRAIN_S1_SAMPLE = int(os.environ.get("ER_TRAIN_SAMPLE", 200_000))
VAL_S1_SAMPLE = int(os.environ.get("ER_VAL_SAMPLE", 100_000))

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}
LGB_ROUNDS = int(os.environ.get("ER_ROUNDS", 2000))
