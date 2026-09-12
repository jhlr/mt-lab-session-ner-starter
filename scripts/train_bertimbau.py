"""Fine-tune BERTimbau for token classification (NER) on the product-title dataset.

Loads char-offset span annotations, converts them to BIO labels aligned to
BERTimbau's WordPiece tokenization, fine-tunes with `transformers.Trainer`,
and evaluates on the held-out test set with seqeval.
"""

import json
import time
from pathlib import Path

import numpy as np
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "data" / "annotations" / "train.jsonl"
TEST_PATH = REPO_ROOT / "data" / "annotations" / "test.jsonl"
RESULTS_DIR = REPO_ROOT / "notebooks" / "results"
OUTPUT_DIR = REPO_ROOT / "models" / "bertimbau_run"  # gitignored, outside repo tracking

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
TAGS = ["TIPO", "MARCA", "MODELO", "MEMORIA", "RAM", "COR", "TELA"]


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_label_list() -> list[str]:
    labels = ["O"]
    for tag in TAGS:
        labels.append(f"B-{tag}")
        labels.append(f"I-{tag}")
    return labels


LABEL_LIST = build_label_list()
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


def align_labels(examples: list[dict], tokenizer) -> Dataset:
    texts = [ex["text"] for ex in examples]
    all_entities = [ex["entities"] for ex in examples]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=64,
        return_offsets_mapping=True,
    )

    all_labels = []
    for i, entities in enumerate(all_entities):
        offsets = tokenized["offset_mapping"][i]
        word_ids = tokenized.word_ids(batch_index=i)
        labels = []
        prev_word_idx = None
        # For each entity, find the token index whose offsets start exactly at ent_start
        # (or the first token whose span overlaps ent_start) -> that's the "B" token.
        entity_first_token = {}
        for ent_idx, (ent_start, ent_end, tag) in enumerate(entities):
            first_tok = None
            for tok_idx, (start, end) in enumerate(offsets):
                if word_ids[tok_idx] is None or (start == 0 and end == 0):
                    continue
                if start < ent_end and end > ent_start:
                    first_tok = tok_idx
                    break
            if first_tok is not None:
                entity_first_token[first_tok] = tag

        for tok_idx, (start, end) in enumerate(offsets):
            if word_ids[tok_idx] is None:
                labels.append(-100)
                continue
            # continuation subword of the same word -> -100 (standard HF convention)
            if word_ids[tok_idx] == prev_word_idx:
                labels.append(-100)
                prev_word_idx = word_ids[tok_idx]
                continue
            prev_word_idx = word_ids[tok_idx]

            # find which entity (if any) this token's char span falls inside
            tag_for_token = None
            is_b = False
            for ent_start, ent_end, tag in entities:
                if start < ent_end and end > ent_start:
                    tag_for_token = tag
                    is_b = tok_idx in entity_first_token and entity_first_token[tok_idx] == tag
                    break
            if tag_for_token is None:
                labels.append(LABEL2ID["O"])
            elif is_b:
                labels.append(LABEL2ID[f"B-{tag_for_token}"])
            else:
                labels.append(LABEL2ID[f"I-{tag_for_token}"])

        all_labels.append(labels)

    tokenized["labels"] = all_labels
    tokenized.pop("offset_mapping")
    return Dataset.from_dict(dict(tokenized))


def compute_metrics_builder():
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [ID2LABEL[p] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [ID2LABEL[l] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        return {"f1": f1_score(true_labels, true_predictions)}

    return compute_metrics


def main():
    print(f"Loading data from {TRAIN_PATH} and {TEST_PATH}")
    train_examples = load_jsonl(TRAIN_PATH)
    test_examples = load_jsonl(TEST_PATH)
    print(f"Train: {len(train_examples)} examples | Test: {len(test_examples)} examples")

    print(f"Loading tokenizer/model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_dataset = align_labels(train_examples, tokenizer)
    test_dataset = align_labels(test_examples, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        use_cpu=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(),
        processing_class=tokenizer,
    )

    print("Starting fine-tuning...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    print("Running final evaluation on test set...")
    predictions_output = trainer.predict(test_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids

    true_predictions = [
        [ID2LABEL[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    report_str = classification_report(true_labels, true_predictions, digits=4)
    report_dict = classification_report(true_labels, true_predictions, digits=4, output_dict=True)

    print("\n=== Classification report (test set) ===")
    print(report_str)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "model_name": MODEL_NAME,
        "training_time_seconds": elapsed,
        "num_train_examples": len(train_examples),
        "num_test_examples": len(test_examples),
        "report": report_dict,
    }
    def to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    metrics_path = RESULTS_DIR / "bertimbau_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(metrics_payload), f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")

    def spans_from_bio(text: str, tokens_offsets, bio_labels):
        """Rebuild char-offset entity spans from a BIO tag sequence + matching offsets."""
        spans = []
        current = None
        for (start, end), label in zip(tokens_offsets, bio_labels):
            if label == "O":
                if current:
                    spans.append(current)
                    current = None
                continue
            prefix, tag = label.split("-", 1)
            if prefix == "B" or current is None or current[2] != tag:
                if current:
                    spans.append(current)
                current = [start, end, tag]
            else:
                current[1] = end
        if current:
            spans.append(current)
        return [(s, e, t, text[s:e]) for s, e, t in spans]

    import torch

    print("\n=== Qualitative examples ===")
    n_qual = min(8, len(test_examples))
    qual_examples = []
    model.eval()
    for i in range(n_qual):
        ex = test_examples[i]
        text = ex["text"]
        enc = tokenizer(text, truncation=True, max_length=64, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        word_ids = enc.word_ids(0)

        with torch.no_grad():
            logits = model(**{k: torch.tensor([v]) for k, v in enc.items()}).logits
        pred_ids = logits.argmax(-1)[0].tolist()

        # one label per word (from its first subword, mirroring training), but the
        # word's span must cover ALL its subwords, not just the first, or the
        # reconstructed entity text gets truncated (e.g. "Smartphone" -> "S").
        word_spans: dict[int, list[int]] = {}
        word_label: dict[int, str] = {}
        for idx, w in enumerate(word_ids):
            if w is None:
                continue
            start, end = offsets[idx]
            if w not in word_spans:
                word_spans[w] = [start, end]
                word_label[w] = ID2LABEL[pred_ids[idx]]
            else:
                word_spans[w][1] = end

        filtered_offsets = [tuple(word_spans[w]) for w in sorted(word_spans)]
        filtered_labels = [word_label[w] for w in sorted(word_spans)]

        pred_spans = spans_from_bio(text, filtered_offsets, filtered_labels)
        gold_spans = [(s, e, t, text[s:e]) for s, e, t in ex["entities"]]

        print(f"\nText: {text}")
        print(f"  Gold: {gold_spans}")
        print(f"  Pred: {pred_spans}")

        qual_examples.append({"text": text, "gold": gold_spans, "pred": pred_spans})

    qual_path = RESULTS_DIR / "bertimbau_qualitative.json"
    with qual_path.open("w", encoding="utf-8") as f:
        json.dump(qual_examples, f, ensure_ascii=False, indent=2)
    print(f"Saved qualitative examples to {qual_path}")


if __name__ == "__main__":
    main()
