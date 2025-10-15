import os
import time
import argparse
from typing import List

import torch
from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

from utilities import (
    load_tokenized_dataset,
    count_trainable_params,
)
from efficient_tuning import infer_lora_targets 
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt

def train_and_eval_lora(
    model_name: str,
    dataset_name: str,
    text_field: str,
    r: int,
    alpha: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    max_length: int,
    seed: int,
    base_output_dir: str,
):
    set_seed(seed)

    # Tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds, collator = load_tokenized_dataset(dataset_name, text_field, tokenizer, max_length)
    feat = ds["train"].features.get("labels") or ds["train"].features.get("label")
    num_labels = getattr(feat, "num_classes", len(getattr(feat, "names", [])) or 2)

    # Base model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # LoRA config
    target_modules = infer_lora_targets(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["classifier", "pre_classifier"],  # DistilBERT head
    )
    model = get_peft_model(model, lora_config)

    trainable, total, frac = count_trainable_params(model)

    # Output dir per r
    out_dir = os.path.join(base_output_dir, model_name.replace("/", "_"), f"r{r}")
    os.makedirs(out_dir, exist_ok=True)

    # Training args
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to=[],  # no wandb
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Train and time
    t0 = time.perf_counter()
    trainer.train()
    train_secs = time.perf_counter() - t0

    # Save adapter + tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Test evaluation
    preds = trainer.predict(ds["test"])
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=-1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    return {
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_frac": frac,
        "train_secs": train_secs,
        **metrics,
        "output_dir": out_dir,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--dataset", type=str, default="rotten_tomatoes")
    ap.add_argument("--text_field", type=str, default="text")
    ap.add_argument("--r_values", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--alpha_factor", type=float, default=2.0, help="alpha = int(r * alpha_factor)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default=r"./outputs/peft_ablation")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    results: List[dict] = []

    print(f"Running LoRA r ablation on {args.model_name} for {args.dataset}")
    for r in args.r_values:
        alpha = int(max(1, r * args.alpha_factor))
        print(f"\n=== r={r}, alpha={alpha}, dropout={args.dropout} ===")
        res = train_and_eval_lora(
            model_name=args.model_name,
            dataset_name=args.dataset,
            text_field=args.text_field,
            r=r,
            alpha=alpha,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
            base_output_dir=args.output_dir,
        )
        results.append(res)
        print(
            f"r={r:>2} | train_secs={res['train_secs']:.2f} | "
            f"acc={res['accuracy']:.4f} | f1_macro={res['f1_macro']:.4f} | "
            f"mcc={res['mcc']:.4f} | trainable={res['trainable_params']:,} "
            f"({res['trainable_frac']:.2%})"
        )

    rs = [r["r"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1s  = [r["f1_macro"] for r in results]
    mccs = [r["mcc"] for r in results]
    secs = [r["train_secs"] for r in results]

    def _bar(ax, xlabels, values, title, ylabel, fmt="{:.3f}"):
        x = np.arange(len(xlabels))
        bars = ax.bar(x, values, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in xlabels])
        ax.set_title(title)
        ax.set_xlabel("LoRA rank r")
        ax.set_ylabel(ylabel)
        for b, v in zip(bars, values):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _bar(axes[0,0], rs, accs, "Accuracy vs r", "Accuracy")
    _bar(axes[0,1], rs, f1s,  "F1 Macro vs r", "F1 Macro")
    _bar(axes[1,0], rs, mccs, "MCC vs r", "MCC")
    _bar(axes[1,1], rs, secs, "Training Time vs r", "Seconds", fmt="{:.1f}")

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "lora_r_ablation.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved bar plots to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
