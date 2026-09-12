"""
CRF-based NER for e-commerce cellphone product titles.

Pipeline:
1. Load train/test jsonl (char-offset entity spans).
2. Tokenize each title with a regex tokenizer that keeps alphanumeric runs with
   internal punctuation intact (e.g. "6,5", "128GB", "Mp+2Mp" splits into
   sensible units) while splitting off surrounding punctuation as its own token.
3. Project char-offset spans onto tokens to build BIO tag sequences. A token is
   labeled only when its span is FULLY contained inside an annotated entity
   span; a token that only partially overlaps an entity is logged and the
   example is skipped (never silently mislabeled).
4. Extract hand-crafted per-token features (case/shape, affixes, digit/unit
   patterns, neighbor window, position flags).
5. Train sklearn_crfsuite.CRF (lbfgs).
6. Evaluate on test.jsonl with seqeval, print + save metrics, print qualitative
   examples.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import sklearn_crfsuite
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.scheme import IOB2

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "annotations" / "train.jsonl"
TEST_PATH = ROOT / "data" / "annotations" / "test.jsonl"
RESULTS_DIR = ROOT / "notebooks" / "results"
RESULTS_PATH = RESULTS_DIR / "crf_metrics.json"

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
# Design choice: split on whitespace, but within a whitespace-delimited chunk
# keep alphanumeric runs (with an internal comma/dot used as a decimal
# separator, e.g. "6,5") together as one token, and split off surrounding
# punctuation (quotes, dashes, slashes, parentheses, "+") as separate tokens.
# This keeps units like "128GB", "6,5"", "13MP+2MP" -> "13MP", "+", "2MP"
# in a form the feature extractor can reason about, without merging unrelated
# words.
TOKEN_PATTERN = re.compile(
    r"\d+[.,]\d+|"      # decimal numbers like 6,5 or 6.5
    r"[A-Za-zÀ-ÿ]+\d+|" # letters directly followed by digits, e.g. A10s -> kept whole below via alnum run
    r"[A-Za-zÀ-ÿ0-9]+|" # generic alphanumeric run
    r"[^\sA-Za-zÀ-ÿ0-9]"  # single punctuation character
)


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Return list of (token_text, start_char, end_char) using finditer so
    offsets map back to the original string exactly."""
    tokens = []
    for m in TOKEN_PATTERN.finditer(text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


# ---------------------------------------------------------------------------
# BIO conversion
# ---------------------------------------------------------------------------
def spans_to_bio(
    text: str, entities: list[list], tokens: list[tuple[str, int, int]]
) -> list[str] | None:
    """Convert char-offset entity spans into BIO tags aligned to `tokens`.

    Returns None (and logs) if any token partially overlaps an entity span
    without being fully contained in it - such examples are skipped rather
    than mislabeled.
    """
    tags = ["O"] * len(tokens)
    for start, end, label in entities:
        first_token_in_span = True
        for i, (_, tok_start, tok_end) in enumerate(tokens):
            if tok_end <= start or tok_start >= end:
                continue  # no overlap with this entity
            fully_inside = tok_start >= start and tok_end <= end
            if not fully_inside:
                print(
                    f"[WARN] partial token/span overlap - skipping example. "
                    f"text={text!r} span=({start},{end},{label}) "
                    f"token=({tokens[i][0]!r},{tok_start},{tok_end})",
                    file=sys.stderr,
                )
                return None
            tags[i] = f"B-{label}" if first_token_in_span else f"I-{label}"
            first_token_in_span = False
    return tags


def load_dataset(path: Path) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Load a jsonl file, returning (token_lists, tag_lists, raw_texts) for
    examples that align cleanly."""
    sentences: list[list[str]] = []
    tag_lists: list[list[str]] = []
    texts: list[str] = []
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row["text"]
            entities = row.get("entities", [])
            tokens = tokenize(text)
            if not tokens:
                skipped += 1
                continue
            tags = spans_to_bio(text, entities, tokens)
            if tags is None:
                skipped += 1
                continue
            sentences.append([t[0] for t in tokens])
            tag_lists.append(tags)
            texts.append(text)
    print(f"[{path.name}] loaded {len(sentences)} examples, skipped {skipped}")
    return sentences, tag_lists, texts


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
UNIT_RE = re.compile(r"^(gb|mb|tb|mp|pol|hz|mah|ram)$", re.IGNORECASE)
DECIMAL_RE = re.compile(r"^\d+[.,]\d+$")  # e.g. 6,5 or 6.5 - typical of TELA
NUMERIC_WITH_UNIT_RE = re.compile(r"^\d+[a-zA-Z]+$")  # e.g. 128gb, 13mp


def word_shape(word: str) -> str:
    """Coarse capitalization/shape pattern, e.g. 'Galaxy' -> 'Xx', 'A10s' ->
    'Xdx', '128GB' -> 'dX', '6,5' -> 'd,d'."""
    shape = []
    for ch in word:
        if ch.isdigit():
            shape.append("d")
        elif ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        else:
            shape.append(ch)
    # collapse consecutive repeats to keep feature space small
    collapsed = [shape[0]] if shape else []
    for c in shape[1:]:
        if c != collapsed[-1]:
            collapsed.append(c)
    return "".join(collapsed)


def token_features(word: str) -> dict:
    lower = word.lower()
    return {
        "word.lower": lower,
        "word.shape": word_shape(word),
        "word.suffix2": lower[-2:],
        "word.suffix3": lower[-3:],
        "word.prefix2": lower[:2],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.has_digit": any(c.isdigit() for c in word),
        "word.is_unit": bool(UNIT_RE.match(lower)),
        "word.is_decimal": bool(DECIMAL_RE.match(word)),
        "word.is_numeric_with_unit": bool(NUMERIC_WITH_UNIT_RE.match(lower)),
        "word.len": len(word),
        "word.is_punct": not any(c.isalnum() for c in word),
    }


def word2features(sentence: list[str], i: int) -> dict:
    features = {"bias": 1.0}
    features.update({f"cur.{k}": v for k, v in token_features(sentence[i]).items()})

    if i == 0:
        features["BOS"] = True
    else:
        features.update(
            {f"-1.{k}": v for k, v in token_features(sentence[i - 1]).items()}
        )

    if i == len(sentence) - 1:
        features["EOS"] = True
    else:
        features.update(
            {f"+1.{k}": v for k, v in token_features(sentence[i + 1]).items()}
        )

    return features


def sent2features(sentence: list[str]) -> list[dict]:
    return [word2features(sentence, i) for i in range(len(sentence))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_sents, train_tags, _ = load_dataset(TRAIN_PATH)
    test_sents, test_tags, test_texts = load_dataset(TEST_PATH)

    X_train = [sent2features(s) for s in train_sents]
    y_train = train_tags
    X_test = [sent2features(s) for s in test_sents]
    y_test = test_tags

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    report_str = seq_classification_report(y_test, y_pred, scheme=IOB2, digits=4)
    report_dict = seq_classification_report(
        y_test, y_pred, scheme=IOB2, digits=4, output_dict=True
    )
    overall_f1 = seq_f1_score(y_test, y_pred, scheme=IOB2)

    print("\n=== seqeval classification report (test set) ===")
    print(report_str)
    print(f"Overall micro F1: {overall_f1:.4f}")

    def _json_default(o):
        if hasattr(o, "item"):  # numpy scalar (e.g. int64/float64) -> native
            return o.item()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_f1_micro": overall_f1,
                "report": report_dict,
                "n_train": len(train_sents),
                "n_test": len(test_sents),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
    print(f"\nSaved metrics to {RESULTS_PATH}")

    print("\n=== Qualitative examples (test set) ===")
    n_show = min(8, len(test_sents))
    for i in range(n_show):
        print(f"\n--- Example {i + 1} ---")
        print(f"Text: {test_texts[i]}")
        print(f"{'Token':<15}{'Gold':<12}{'Pred':<12}")
        for tok, gold, pred in zip(test_sents[i], y_test[i], y_pred[i]):
            marker = "" if gold == pred else "  <-- MISMATCH"
            print(f"{tok:<15}{gold:<12}{pred:<12}{marker}")


if __name__ == "__main__":
    main()
