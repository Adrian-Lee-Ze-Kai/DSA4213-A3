import numpy as np
from typing import Tuple, Callable
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate


def load_tokenized_dataset(dataset_name: str, text_column: str, tokenizer, max_length: int):
    ds = load_dataset(dataset_name)
    def tok(batch):
        return tokenizer(batch[text_column], truncation=True, max_length=max_length)
    ds = ds.map(tok, batched=True, remove_columns=[text_column])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return ds, collator


def make_compute_metrics(metrics: list) -> Callable:
    metric_loaders = {metric: evaluate.load(metric) for metric in metrics}
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        results = {}
        for metric, loader in metric_loaders.items():
            if metric == "f1":
                results["f1_macro"] = loader.compute(predictions=preds, references=labels, average="macro")["f1"]
            else:
                results[metric] = loader.compute(predictions=preds, references=labels)[metric]
        return results
    
    return compute_metrics


def count_trainable_params(model) -> Tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / total