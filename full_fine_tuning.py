import os
import argparse
from transformers import Trainer, TrainingArguments, set_seed
from utilities import (
    load_tokenized_dataset,
    make_compute_metrics,
    count_trainable_params,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name or path")
    ap.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    ap.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output_dir", type=str, default="./outputs/full", help="Directory to save the trained model")
    return ap.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    # Load tokenizer dynamically based on the model name
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset and tokenize using the utility function
    ds, collator = load_tokenized_dataset("ag_news", "text", tokenizer, args.max_length)
    num_labels = len(ds["train"].features["label"].names)

    # Load model dynamically based on the model name
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # Count trainable parameters using the utility function
    trainable, total, frac = count_trainable_params(model)
    print(f"Full FT: trainable {trainable:,} / {total:,} ({frac:.2%})")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.model_name.replace("/", "_")),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(["accuracy", "f1"]),
    )

    # Train the model
    trainer.train()
    trainer.save_model(training_args.output_dir)
    print("Saved to:", training_args.output_dir)


if __name__ == "__main__":
    main()