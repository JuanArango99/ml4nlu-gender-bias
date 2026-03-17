"""
data_loader.py
--------------
Corpus / anchor / profession loading, tokenizer-level sentence search,
and gender-marker sentence classification.
"""

import csv
import re
from pathlib import Path

# Maps the two-letter language code to the data sub-directory name.
_LANG_DIR = {"ti": "tigrigna", "ar": "arabic", "es": "spanish"}

#  gender markers for sentence classification (debug reporting).
_MALE_MARKERS = {
    "abuelo",
    "bisabuelo",
    "bisnieto",
    "caballero",
    "chico",
    "cuñado",
    "él",
    "esposo",
    "hermano",
    "hijo",
    "hombre",
    "muchacho",
    "nieto",
    "niño",
    "padre",
    "padrino",
    "papá",
    "rey",
    "sobrino",
    "suegro",
    "tío",
    "varón",
    "yerno",
}

_FEMALE_MARKERS = {
    "abuela",
    "bisabuela",
    "bisnieta",
    "chica",
    "cuñada",
    "dama",
    "ella",
    "esposa",
    "hermana",
    "hija",
    "madre",
    "madrina",
    "mamá",
    "muchacha",
    "mujer",
    "nieta",
    "niña",
    "nuera",
    "reina",
    "señora",
    "señorita",
    "sobrina",
    "suegra",
    "tía",
}


def load_corpus(lang: str) -> list:
    import config

    if getattr(config, "USE_HF_DATASET", False):
        import nltk
        from datasets import load_dataset

        print(
            f"Loading {config.HF_MAX_SENTENCES} sentences from HuggingFace dataset {config.HF_DATASET_NAME} ({config.HF_DATASET_CONFIG})..."
        )

        # Load dataset in streaming mode so we don't download the whole thing
        dataset = load_dataset(
            config.HF_DATASET_NAME,
            config.HF_DATASET_CONFIG,
            split=config.HF_DATASET_SPLIT,
            streaming=True,
        )

        sentences = []
        for item in dataset:
            text = item.get("text", "")
            if not text:
                continue

            # Tokenize into sentences
            sents = nltk.sent_tokenize(text)

            # Add to our list, stripping whitespace
            for s in sents:
                clean_s = s.strip()
                if clean_s:
                    sentences.append(clean_s)

            if len(sentences) >= config.HF_MAX_SENTENCES:
                break

        print(f"Loaded {len(sentences[:config.HF_MAX_SENTENCES])} sentences.")
        return sentences[: config.HF_MAX_SENTENCES]
    else:
        path = Path("data") / _LANG_DIR[lang] / f"corpus_{lang}.txt"
        return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_anchors(lang: str) -> tuple:
    """Return (male_terms, female_terms) from the anchors CSV."""
    path = Path("data") / _LANG_DIR[lang] / f"anchors_{lang}.csv"
    male, female = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            gender = row["gender"].strip().lower()
            term = row["term"].strip()
            if gender == "male":
                male.append(term)
            elif gender == "female":
                female.append(term)
    return male, female


def load_professions(lang: str) -> list:
    path = Path("data") / _LANG_DIR[lang] / f"professions_{lang}.csv"
    with open(path, encoding="utf-8", newline="") as f:
        return [row["profession"].strip() for row in csv.DictReader(f) if row["profession"].strip()]


def classify_sentence(sentence: str, target_word: str = None) -> str:
    """Return 'F', 'M', or 'N' based on gender marker presence or adjacent articles."""
    sentence_lower = sentence.lower()
    
    # 1. Try to classify based on adjacent articles if a target word is provided
    if target_word:
        target_lower = target_word.lower()
        
        # Male articles (el, un, del, al, los, unos)
        male_pattern = rf'\b(?:el|un|del|al|los|unos)\s+(?:\w+\s+){{0,2}}{re.escape(target_lower)}\b'
        # Female articles (la, una, de la, a la, las, unas)
        female_pattern = rf'\b(?:la|una|de la|a la|las|unas)\s+(?:\w+\s+){{0,2}}{re.escape(target_lower)}\b'
        
        m_article_match = re.search(male_pattern, sentence_lower)
        f_article_match = re.search(female_pattern, sentence_lower)
        
        if m_article_match and not f_article_match:
            return "M"
        elif f_article_match and not m_article_match:
            return "F"
            
        # 2. Try to classify based on morphological gender of the target word itself
        epicene_suffixes = (
            "ista", "ata", "eta", "ota", "ita", "arca", "iatra", "nauta", "cida"
        )
        epicene_words = {
            "policía", "fisioterapeuta", "psiquiatra", "terapeuta", "pediatra", 
            "piloto", "modelo", "juez", "portavoz", "estudiante", "docente", 
            "cantante", "gerente", "representante", "agente", "presidente", 
            "jefe", "guía", "joven", "chef", "militar", "guardia"
        }
        
        if target_lower not in epicene_words and not any(target_lower.endswith(sfx) for sfx in epicene_suffixes):
            if target_lower.endswith("o"):
                return "M"
            elif target_lower.endswith("a"):
                return "F"
            
    # 3. Fallback to general gender markers counting
    words = re.findall(r"\w+", sentence_lower)

    f_count = sum(1 for w in words if w in _FEMALE_MARKERS)
    m_count = sum(1 for w in words if w in _MALE_MARKERS)

    if m_count > f_count:
        return "M"
    elif f_count > m_count:
        return "F"
    return "N"


def find_sentences(corpus: list, word: str, n: int, tokenizer, verbose: bool = False) -> list:
    """Return up to n sentences that contain word (tokenizer-level matching)."""
    word_tokens = tokenizer.tokenize(word)
    if verbose:
        print(f"\n  find_sentences('{word}', n={n})")
        print(f"    word tokenizes to: {word_tokens}")

    found = []
    for sentence in corpus:
        sent_tokens = tokenizer.tokenize(sentence)
        for i in range(len(sent_tokens) - len(word_tokens) + 1):
            if sent_tokens[i : i + len(word_tokens)] == word_tokens:
                found.append(sentence)
                break
        if len(found) >= n:
            break

    if verbose:
        print(f"    found {len(found)}/{n} sentences")
        for i, s in enumerate(found):
            print(f"    [{i+1}] {s}")
        if len(found) < n:
            print(f"    WARNING: only {len(found)} sentences — word may be SKIPPED in bulk mode")
    return found


def find_all_matching(corpus: list, word: str) -> list:
    """Return (row_idx, sentence) for every corpus sentence containing word."""
    return [(idx, s) for idx, s in enumerate(corpus) if word in s]
