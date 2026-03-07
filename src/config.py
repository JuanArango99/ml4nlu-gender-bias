# ── Language & model ──────────────────────────────────────────────────────────
# Set this to YOUR language: "ar" for Arabic, "es" for Spanish
LANGUAGE = "es"

# Models to run — all four are listed; comment out any you don't need.
# Note: xlm-roberta-large is slow on CPU — run it last or separately.
MODEL_NAMES = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "facebook/xlm-v-base",
    "microsoft/mdeberta-v3-base",
]
# To run a single model only, replace the list with a string:
# MODEL_NAMES = "xlm-roberta-base"

# Short slugs used in output filenames — do not change these.
MODEL_SLUG_MAP = {
    "xlm-roberta-base": "xlmr_base",
    "xlm-roberta-large": "xlmr_large",
    "facebook/xlm-v-base": "xlmv_base",
    "microsoft/mdeberta-v3-base": "mdeberta",
}

# ── Run mode ──────────────────────────────────────────────────────────────────
# "bulk"  → scores all professions, one progress line per word (paper results)
# "debug" → verbose trace for a single word (use first to verify setup works)
MODE = "bulk"

# ── Debug word (only used when MODE = "debug") ────────────────────────────────
# Set to any profession term from your professions CSV.
DEBUG_WORD = "médico"

# ── Output root ───────────────────────────────────────────────────────────────
OUTPUT_ROOT = "output"

# ── Context counts ────────────────────────────────────────────────────────────
# Do NOT change these — they must match the reference experiment values
# for cross-lingual comparability.
ANCHOR_CONTEXTS = 12  # sentences per anchor word
PROFESSION_CONTEXTS = 20  # sentences per profession term

# ── HuggingFace Dataset ───────────────────────────────────────────────────────
# Set to True to use a dataset from HuggingFace instead of local dummy files.
USE_HF_DATASET = True
HF_DATASET_NAME = "wikimedia/wikipedia"
# The config name depends on the LANGUAGE setting.
# Example: "20231101.es" for Spanish, "20231101.ar" for Arabic.
HF_DATASET_CONFIG = f"20231101.{LANGUAGE}"
HF_DATASET_SPLIT = "train"
# Maximum number of sentences to extract from the dataset.
HF_MAX_SENTENCES = 1000000   
