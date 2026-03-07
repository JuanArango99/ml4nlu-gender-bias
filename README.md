# Gender Bias in Multilingual Transformers — Collaborator Pipeline

This folder contains everything you need to run the gender bias analysis
for **your language** (Arabic or Spanish).

The pipeline probes four multilingual transformer models layer-by-layer and
measures whether profession embeddings are geometrically closer to a male
or female gender direction.  No fine-tuning, no labels — fully unsupervised.

---

## Directory Layout

```
group-share/
├── src/
│   ├── config.py          ← EDIT THIS first (language + model selection)
│   ├── main.py            ← entry point — run this
│   ├── data_loader.py     ← reads corpus, anchors, professions from data/
│   ├── embeddings.py      ← builds layer-wise gender geometry
│   ├── metrics.py         ← projection score + cosine diff per layer
│   ├── output.py          ← plots projection curve, writes CSVs, Tee logger
│   └── debug_mode.py      ← verbose single-profession trace (debug mode)
├── data/
│   ├── arabic/
│   │   ├── corpus_ar.txt         ← one sentence per line
│   │   ├── anchors_ar.csv        ← gender anchor words (pronouns, kinship)
│   │   └── professions_ar.csv    ← 30 profession terms to score
│   └── spanish/
│       ├── corpus_es.txt
│       ├── anchors_es.csv
│       └── professions_es.csv
├── requirements.txt
└── README.md              ← you are here
```

---
## Quick-Start

### 1. Create a Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Arabic collaborator only:** also install `arabic-reshaper` and `python-bidi`
> if you want Arabic text to render correctly in matplotlib output (optional).

### 2. Edit `src/config.py`

Open `src/config.py` and set **three things**:

```python
LANGUAGE = "ar"        # change to "es" for Spanish
MODE     = "debug"     # start with "debug" to verify setup, then switch to "bulk"
DEBUG_WORD = "طبيب"    # any term from your professions CSV (for debug mode)
```

You can comment out models you don't need to run fewer models.

### 3. Run

```bash
cd group-share       # must run from this folder, not from src/
python src/main.py
```

Output lands in `output/{lang}/{model_name}/`.

---

## Two Modes Explained

| Mode | What it does | When to use |
|------|-------------|-------------|
| `debug` | Scores one profession in full detail: shows every context sentence used, per-layer scores, balance label (M/F/N), verdict | Run first to confirm your data loads and the model can find the word |
| `bulk` | Scores all 30 professions, one line per profession; writes bias CSV, Spearman CSV, and a projection-curve PNG | Use for final results |

---

## Data Files — Schema

### `corpus_{lang}.txt`
Plain text, one sentence per line, UTF-8.  The pipeline searches this file
for sentences containing each anchor/profession word via substring match.

### `anchors_{lang}.csv`
```
gender,term
male,هو
male,رجل
female,هي
female,امرأة
```
The pipeline uses these words to build the gender direction in embedding space.
Each term needs at least **12 sentences** in the corpus (set by `ANCHOR_CONTEXTS`).
If a term appears fewer than 12 times it is silently skipped — this is fine as
long as most anchors have enough coverage.

### `professions_{lang}.csv`
```
profession_id,profession_lang,profession
ar_001,ar,طبيب
ar_002,ar,ممرضة
```
Only the `profession` column is used.  Each term needs at least **20 sentences**
in the corpus (set by `PROFESSION_CONTEXTS`).  Terms with fewer occurrences
are skipped and reported in the console output.

---

## Output Files

After a bulk run, `output/{lang}/{model_folder}/` contains:

| File | Columns | Description |
|------|---------|-------------|
| `{lang}_bias_by_layer.csv` | language, model, term, layer, proj, cosdiff | Raw per-term per-layer scores |
| `{lang}_spearman_by_layer.csv` | language, model, layer, spearman_rho, spearman_p, n | Agreement between the two bias estimators at each layer |
| `{lang}_projection_layer_mean.csv` | layer, mean_proj | Mean projection across all professions per layer |
| `figs/{lang}_{slug}_projection_curve.png` | — | Layer-mean projection curve plot |

### Columns explained

- **proj**: dot product of the profession embedding with the gender direction
  vector.  Positive = male-leaning, negative = female-leaning.
- **cosdiff**: `cosine(word, male_centroid) − cosine(word, female_centroid)`.
  A second independent estimator; should agree strongly with `proj` (Spearman ρ
  close to 1.0 indicates the two measures are consistent).
- **spearman_rho**: agreement between `proj` and `cosdiff` rankings across all
  professions at a given layer.  Expect > 0.90 for well-formed models.

---

## Pipeline — How it works (step by step)

```
corpus + anchors
      │
      ▼
[1] Build gender geometry
      For each anchor word, collect 12 context sentences from the corpus.
      Extract the token embedding at each transformer layer (all layers).
      Average embeddings across sentences → one vector per anchor per layer.
      male_centroid[layer]   = mean of all male anchor vectors at that layer
      female_centroid[layer] = mean of all female anchor vectors at that layer
      gender_direction[layer] = male_centroid[layer] − female_centroid[layer]
      │
      ▼
[2] Score each profession
      Collect 20 context sentences for the profession term.
      Extract layer-wise token embeddings (same procedure).
      proj    = dot(profession_vec, gender_direction)       [signed magnitude]
      cosdiff = cos(word, male_centroid) − cos(word, female_centroid)  [angular]
      │
      ▼
[3] Aggregate
      Compute mean projection per layer across all 30 professions.
      Compute Spearman ρ between proj and cosdiff rankings per layer.
      │
      ▼
[4] Output
      Write CSVs + PNG projection curve.
```

---

## Config Reference

All settings live in `src/config.py`:

| Setting | Default | Notes |
|---------|---------|-------|
| `LANGUAGE` | `"ar"` | `"ar"` or `"es"` |
| `MODEL_NAMES` | list of 4 models | Comment out models to skip them |
| `MODE` | `"bulk"` | `"debug"` or `"bulk"` |
| `DEBUG_WORD` | `"طبيب"` | Must exist in your professions CSV and corpus |
| `OUTPUT_ROOT` | `"output"` | Change to `"output_test"` to avoid overwriting results |
| `ANCHOR_CONTEXTS` | `12` | Sentences per anchor — do not change |
| `PROFESSION_CONTEXTS` | `20` | Sentences per profession — do not change |

> **Do not change `ANCHOR_CONTEXTS` or `PROFESSION_CONTEXTS`.**
> These values must match the reference experiment for cross-lingual
> comparability of results.

---

## Models

The pipeline runs four models in sequence (each is downloaded automatically
from HuggingFace on first use):

| Model | Size | Notes |
|-------|------|-------|
| `xlm-roberta-base` | ~270 MB | Fast, good baseline |
| `xlm-roberta-large` | ~1.1 GB | Slower, higher capacity |
| `facebook/xlm-v-base` | ~270 MB | 1M vocab, better coverage for morphologically rich languages |
| `microsoft/mdeberta-v3-base` | ~280 MB | Disentangled attention — may behave differently |

First download for all four models takes ~2 GB of disk space total.
Subsequent runs use the local HuggingFace cache.

---

## Troubleshooting

**"Not enough sentences" / profession skipped**
The term appears fewer than 20 times in the corpus.  This is expected for some
rare or multi-word terms.  Up to ~3–4 skipped is acceptable; if many are
skipped, check that `config.LANGUAGE` matches the corpus and that your
profession terms are spelled the same way as they appear in the corpus.

**"INSUFFICIENT" for anchor in debug mode**
An anchor appears fewer than 12 times.  The anchor is skipped automatically.
As long as at least 5–6 anchors per gender survive, results are valid.

**mDeBERTa fails to load**
Install the extra packages:
```bash
pip install sentencepiece protobuf
```
and ensure `use_fast=False` is set for deberta — the code handles this
automatically.

**Slow on CPU**
xlm-roberta-large and mDeBERTa are the heaviest.  On a modern laptop, a full
bulk run across all 4 models takes roughly 15–30 minutes.  Run
`xlm-roberta-base` first to verify results quickly.
