"""Train a blank spaCy NER pipeline on the e-commerce product title dataset.

Loads char-offset annotated JSONL data, converts it to spaCy's Example format,
trains a `ner`-only blank Portuguese pipeline from scratch, evaluates on the
held-out test set with seqeval (BIO scheme), and saves metrics + qualitative
examples for use in the accompanying notebook.
"""

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

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "annotations" / "train.jsonl"
TEST_PATH = ROOT / "data" / "annotations" / "test.jsonl"
RESULTS_DIR = ROOT / "notebooks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_ITER = 40
DROPOUT = 0.2
BATCH_SIZE_RANGE = (4.0, 32.0, 1.001)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_doc_and_entities(nlp: spacy.Language, record: dict[str, Any]) -> tuple[Doc, list[tuple[int, int, str]]]:
    doc = nlp.make_doc(record["text"])
    entities = [(start, end, label) for start, end, label in record["entities"]]
    return doc, entities


def build_examples(
    nlp: spacy.Language, records: list[dict[str, Any]], split_name: str
) -> list[Example]:
    """Convert char-offset records into spacy.training.Example objects.

    Uses char_span with alignment_mode="contract" to snap offsets onto token
    boundaries. Any entity that cannot be aligned (contract collapses it to
    None, e.g. a span that doesn't line up with any token boundary at all)
    is logged and skipped, not silently dropped.
    """
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

        # Deduplicate / resolve overlaps: spaCy's biluo/ner training requires
        # non-overlapping spans. Keep entities in order, drop any that overlap
        # a previously kept one (shouldn't occur with clean annotations, but
        # guard against it and report if it does).
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
    return examples


def examples_to_bio(nlp: spacy.Language, examples: list[Example], predict: bool) -> tuple[list[list[str]], list[list[str]]]:
    """Return (gold_bio_sequences, pred_bio_sequences) aligned per example.

    If predict=False, only gold tags are meaningful (pred list mirrors gold);
    used only when computing gold tags on their own. Normally called with
    predict=True to also run the trained model.
    """
    gold_sequences = []
    pred_sequences = []

    for example in examples:
        doc = example.reference
        gold_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        gold_tags = offsets_to_biluo_tags(doc, gold_entities)
        gold_tags = [t.replace("U-", "B-").replace("L-", "I-") if t not in ("O",) else t for t in gold_tags]
        # biluo -> bio conversion: B-/I-/O stay, U- becomes B-, L- becomes I-
        gold_sequences.append(gold_tags)

        if predict:
            pred_doc = nlp(doc.text)
            pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents]
            pred_tags = offsets_to_biluo_tags(doc, pred_entities)
            pred_tags = [t.replace("U-", "B-").replace("L-", "I-") if t not in ("O",) else t for t in pred_tags]
            pred_sequences.append(pred_tags)

    return gold_sequences, pred_sequences


def main() -> None:
    print("Loading data...")
    train_records = load_jsonl(TRAIN_PATH)
    test_records = load_jsonl(TEST_PATH)
    print(f"train examples: {len(train_records)}, test examples: {len(test_records)}")

    nlp = spacy.blank("pt")
    ner = nlp.add_pipe("ner")

    print("\nConverting training data to spaCy Examples...")
    train_examples = build_examples(nlp, train_records, "train")
    print("\nConverting test data to spaCy Examples...")
    test_examples = build_examples(nlp, test_records, "test")

    labels = sorted({ent[2] for r in train_records for ent in r["entities"]})
    for label in labels:
        ner.add_label(label)
    print(f"\nLabels registered: {labels}")

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    print("\nTraining...")
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.initialize(lambda: train_examples)
        for epoch in range(1, N_ITER + 1):
            random.shuffle(train_examples)
            losses: dict[str, float] = {}
            batches = minibatch(train_examples, size=compounding(*BATCH_SIZE_RANGE))
            for batch in batches:
                nlp.update(batch, drop=DROPOUT, losses=losses, sgd=optimizer)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{N_ITER}  ner_loss={losses.get('ner', 0.0):.3f}")

    print("\nEvaluating on test set...")
    gold_bio, pred_bio = examples_to_bio(nlp, test_examples, predict=True)

    report_str = classification_report(gold_bio, pred_bio, digits=3)
    print("\n" + report_str)

    report_dict = classification_report(gold_bio, pred_bio, digits=3, output_dict=True)
    overall_f1 = f1_score(gold_bio, pred_bio)
    report_dict["_overall_f1_seqeval_default"] = overall_f1

    def _to_jsonable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if hasattr(obj, "item"):  # numpy scalar (int64/float64/...)
            return obj.item()
        return obj

    with (RESULTS_DIR / "spacy_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(report_dict), f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to {RESULTS_DIR / 'spacy_metrics.json'}")

    # Save the trained pipeline so the notebook can reload it without retraining.
    model_dir = ROOT / "notebooks" / "results" / "spacy_model"
    nlp.to_disk(model_dir)
    print(f"Saved trained pipeline to {model_dir}")

    print("\n--- Qualitative examples (gold vs predicted) ---")
    sample = test_records[:8]
    qualitative = []
    for record in sample:
        pred_doc = nlp(record["text"])
        gold_ents = [(record["text"][s:e], label) for s, e, label in record["entities"]]
        pred_ents = [(ent.text, ent.label_) for ent in pred_doc.ents]
        qualitative.append({"text": record["text"], "gold": gold_ents, "pred": pred_ents})
        print(f"\nTEXT: {record['text']}")
        print(f"  GOLD: {gold_ents}")
        print(f"  PRED: {pred_ents}")

    with (RESULTS_DIR / "spacy_qualitative_examples.json").open("w", encoding="utf-8") as f:
        json.dump(qualitative, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
