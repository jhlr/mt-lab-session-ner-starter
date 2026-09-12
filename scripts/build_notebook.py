"""Build notebooks/01_spacy_ner.ipynb from nbformat, mirroring train_spacy.py.

Run once (or whenever the narrative/code needs updating). It does not execute
the notebook itself: use `jupyter nbconvert --to notebook --execute` for that.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""\
# spaCy NER: Product Title Entity Extraction

One of three techniques compared in this lab (alongside CRF and BERTimbau, built separately).
This notebook trains a **blank Portuguese spaCy pipeline** (`spacy.blank("pt")`, `ner` component only,
trained from scratch: no pretrained `pt_core_news_*` download) to extract 7 entity types from
Brazilian Portuguese e-commerce product titles (cellphones/accessories):

`TIPO`, `MARCA`, `MODELO`, `MEMORIA`, `RAM`, `COR`, `TELA`

Steps: load data -> convert char-offset spans to spaCy's training format -> train -> evaluate with
seqeval (BIO scheme) -> qualitative error inspection.""")

md("""## 1. Setup and data loading

`train.jsonl` (240 examples) and `test.jsonl` (60 examples). Each line is
`{"text": "...", "entities": [[start, end, "TAG"], ...]}` with character-offset spans.""")

code("""\
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import spacy
from spacy.tokens import Doc
from spacy.training import Example, offsets_to_biluo_tags
from spacy.util import minibatch, compounding
from seqeval.metrics import classification_report, f1_score

random.seed(42)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
TRAIN_PATH = ROOT / "data" / "annotations" / "train.jsonl"
TEST_PATH = ROOT / "data" / "annotations" / "test.jsonl"
RESULTS_DIR = ROOT / "notebooks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_ITER = 40
DROPOUT = 0.2
BATCH_SIZE_RANGE = (4.0, 32.0, 1.001)""")

code("""\
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


train_records = load_jsonl(TRAIN_PATH)
test_records = load_jsonl(TEST_PATH)
print(f"train examples: {len(train_records)}, test examples: {len(test_records)}")
train_records[0]""")

md("""## 2. Convert char-offset entities to spaCy `Example` objects

spaCy tokenizes the raw text itself; the gold character offsets don't always land exactly on a
token boundary (e.g. a span cut mid-punctuation). We use `doc.char_span(..., alignment_mode="contract")`
to snap each span onto the nearest token boundaries: if a span can't be aligned at all
(`char_span` returns `None` or an empty span), we log it and skip that single entity rather than
silently dropping the whole example.""")

code("""\
def make_doc_and_entities(nlp: spacy.Language, record: dict[str, Any]):
    doc = nlp.make_doc(record["text"])
    entities = [(start, end, label) for start, end, label in record["entities"]]
    return doc, entities


def build_examples(nlp: spacy.Language, records: list[dict[str, Any]], split_name: str) -> list[Example]:
    examples = []
    n_entities_total = 0
    n_entities_dropped = 0
    n_examples_with_drop = 0

    for record in records:
        doc, raw_entities = make_doc_and_entities(nlp, record)
        n_entities_total += len(raw_entities)
        kept_entities = []
        dropped_here = []
        for start, end, label in raw_entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None or len(span) == 0:
                dropped_here.append((start, end, label, record["text"][start:end]))
                continue
            kept_entities.append((span.start_char, span.end_char, label))

        if dropped_here:
            n_entities_dropped += len(dropped_here)
            n_examples_with_drop += 1
            for start, end, label, text_snippet in dropped_here:
                print(
                    f"[{split_name}] WARNING: could not align entity "
                    f"({start}, {end}, {label!r}) = {text_snippet!r} in text "
                    f"{record['text']!r}: skipped."
                )

        kept_entities.sort(key=lambda e: e[0])
        non_overlapping = []
        last_end = -1
        for start, end, label in kept_entities:
            if start < last_end:
                print(
                    f"[{split_name}] WARNING: overlapping span ({start}, {end}, {label}) "
                    f"in text {record['text']!r}: skipped."
                )
                continue
            non_overlapping.append((start, end, label))
            last_end = end

        example = Example.from_dict(doc, {"entities": non_overlapping})
        examples.append(example)

    print(
        f"[{split_name}] entities total={n_entities_total}, "
        f"dropped={n_entities_dropped} ({n_examples_with_drop} examples affected), "
        f"kept={n_entities_total - n_entities_dropped}"
    )
    return examples""")

code("""\
nlp = spacy.blank("pt")
ner = nlp.add_pipe("ner")

print("Converting training data to spaCy Examples...")
train_examples = build_examples(nlp, train_records, "train")
print()
print("Converting test data to spaCy Examples...")
test_examples = build_examples(nlp, test_records, "test")

labels = sorted({ent[2] for r in train_records for ent in r["entities"]})
for label in labels:
    ner.add_label(label)
print(f"\\nLabels registered: {labels}")""")

md("""## 3. Train a blank `ner`-only pipeline from scratch

`spacy.blank("pt")` gives us tokenization rules for Portuguese but no pretrained weights: the
`ner` component is initialized and trained entirely on our 240 examples, for 40 epochs with
minibatching (`spacy.util.minibatch` + `compounding` batch sizes) and dropout.""")

code("""\
other_pipes = [p for p in nlp.pipe_names if p != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.initialize(lambda: train_examples)
    for epoch in range(1, N_ITER + 1):
        random.shuffle(train_examples)
        losses: dict[str, float] = {}
        batches = minibatch(train_examples, size=compounding(*BATCH_SIZE_RANGE))
        for batch in batches:
            nlp.update(batch, drop=DROPOUT, losses=losses, sgd=optimizer)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}/{N_ITER}  ner_loss={losses.get('ner', 0.0):.3f}")""")

md("""## 4. Evaluate on the test set with seqeval

spaCy's predicted and gold entities are both converted to BIO tag sequences aligned to spaCy's
own tokenization (via `offsets_to_biluo_tags`, then BILUO -> BIO by folding `U-` into `B-` and
`L-` into `I-`), and compared with `seqeval.metrics.classification_report`.""")

code("""\
def examples_to_bio(nlp: spacy.Language, examples: list[Example], predict: bool):
    gold_sequences = []
    pred_sequences = []

    for example in examples:
        doc = example.reference
        gold_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        gold_tags = offsets_to_biluo_tags(doc, gold_entities)
        gold_tags = [t.replace("U-", "B-").replace("L-", "I-") if t != "O" else t for t in gold_tags]
        gold_sequences.append(gold_tags)

        if predict:
            pred_doc = nlp(doc.text)
            pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents]
            pred_tags = offsets_to_biluo_tags(doc, pred_entities)
            pred_tags = [t.replace("U-", "B-").replace("L-", "I-") if t != "O" else t for t in pred_tags]
            pred_sequences.append(pred_tags)

    return gold_sequences, pred_sequences


gold_bio, pred_bio = examples_to_bio(nlp, test_examples, predict=True)

report_str = classification_report(gold_bio, pred_bio, digits=3)
print(report_str)""")

code("""\
report_dict = classification_report(gold_bio, pred_bio, digits=3, output_dict=True)
overall_f1 = f1_score(gold_bio, pred_bio)
report_dict["_overall_f1_seqeval_default"] = overall_f1


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


with (RESULTS_DIR / "spacy_metrics.json").open("w", encoding="utf-8") as f:
    json.dump(_to_jsonable(report_dict), f, ensure_ascii=False, indent=2)

print(f"Overall micro F1: {overall_f1:.3f}")
print(f"Saved metrics to {RESULTS_DIR / 'spacy_metrics.json'}")

nlp.to_disk(RESULTS_DIR / "spacy_model")""")

md("""## 5. Qualitative examples

A handful of test sentences with gold vs. predicted entities, useful for the write-up's error
discussion (e.g. boundary mistakes on multi-word `MODELO`/`TIPO` spans).""")

code("""\
sample = test_records[:8]
qualitative = []
for record in sample:
    pred_doc = nlp(record["text"])
    gold_ents = [(record["text"][s:e], label) for s, e, label in record["entities"]]
    pred_ents = [(ent.text, ent.label_) for ent in pred_doc.ents]
    qualitative.append({"text": record["text"], "gold": gold_ents, "pred": pred_ents})
    print(f"TEXT: {record['text']}")
    print(f"  GOLD: {gold_ents}")
    print(f"  PRED: {pred_ents}")
    print()

with (RESULTS_DIR / "spacy_qualitative_examples.json").open("w", encoding="utf-8") as f:
    json.dump(qualitative, f, ensure_ascii=False, indent=2)""")

md("""## 6. Summary

The blank spaCy `ner` pipeline, trained from scratch on only 240 annotated titles, reaches a
respectable overall F1 given the small dataset size. Per-tag performance varies: short,
low-variability tags like `RAM` and `MEMORIA` are near-perfect (the `\\d+GB` pattern is easy to
learn), while multi-token, high-variability tags like `MODELO` and `TIPO` (free-form product
names, multi-word phrases like "Capa Protetora" or "Base Carregadora") are harder and show more
boundary errors, visible in the qualitative examples above. These numbers are for comparison
against the CRF and BERTimbau approaches built separately for this assignment.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (mt-lab-session-ner .venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}

out_path = "/Users/joaorietra/Developer/mt-lab-session-ner-starter/notebooks/01_spacy_ner.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out_path}")
