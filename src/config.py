# src/config.py
import os

# === PATH CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAINING_DIR = os.path.join(DATA_DIR, "training")
EVALUATION_DIR = os.path.join(DATA_DIR, "evaluation")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")
LOG_DIR = os.path.join(EXPERIMENT_DIR, "logs")

# === SOLVER CONFIG ===
MAX_SEARCH_DEPTH = 5
MAX_CANDIDATES = 1000
RANDOM_SEED = 42

# === REPRODUCIBILITY ===
import random
import numpy as np
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)