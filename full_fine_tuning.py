import os
import argparse
import time
from transformers import Trainer, TrainingArguments, set_seed
from utilities import (
    load_tokenized_dataset,
    count_trainable_params,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name or path")
    ap.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    ap.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output_dir", type=str, default="./outputs/full", help="Directory to save the trained model")
    return ap.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Use any HF dataset with 'text' and 'label' (e.g., ag_news, rotten_tomatoes, emotion, imdb)
    ds, collator = load_tokenized_dataset("rotten_tomatoes", "text", tokenizer, args.max_length)
    feat = ds["train"].features.get("labels") or ds["train"].features.get("label")
    num_labels = getattr(feat, "num_classes", len(getattr(feat, "names", [])) or 2)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    trainable, total, frac = count_trainable_params(model)
    print(f"Full FT: trainable {trainable:,} / {total:,} ({frac:.2%})")

    out_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        dataloader_pin_memory=False,  # CPU-only: silence pin-memory warning
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],    # train only; no eval during training
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Train and time it
    t0 = time.perf_counter()
    trainer.train()
    total_secs = time.perf_counter() - t0
    print(f"Total training time (seconds): {total_secs:.2f}")

    # Save model + tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()