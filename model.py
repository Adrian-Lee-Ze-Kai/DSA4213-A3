import os
import numpy as np
from typing import Tuple, Callable

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
)
import evaluate


def infer_lora_targets(model_name: str):
    name = model_name.lower()
    if "distilbert" in name:
        return ["q_lin", "v_lin"]
    if "bert" in name or "roberta" in name or "deberta" in name:
        return ["query", "value"]
    # generic fallback for some decoder-style blocks
    return ["q_proj", "v_proj"]


def load_tokenized_agnews(tokenizer, max_length: int):
    ds = load_dataset("ag_news")
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    num_labels = 4
    return ds, collator, num_labels


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def build_model(model_name: str, num_labels: int):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def make_compute_metrics() -> Callable:
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
    return compute_metrics


def count_trainable_params(model) -> Tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / total