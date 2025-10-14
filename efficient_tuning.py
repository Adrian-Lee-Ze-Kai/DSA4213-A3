import os
import argparse
import time
from transformers import Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, TaskType
from utilities import (
    load_tokenized_dataset,
    make_compute_metrics,
    count_trainable_params,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def infer_lora_targets(model_name: str):
     name = model_name.lower()
     if "distilbert" in name:
         return ["q_lin", "v_lin"]
     if any(k in name for k in ["bert", "roberta", "deberta"]):
         return ["query", "value"]
     # generic fallback for many decoder/LLMs
     return ["q_proj", "v_proj"]

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name or path")
    ap.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    ap.add_argument("--max_length", type=int, default=256, help="Maximum sequence length for tokenization")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--output_dir", type=str, default="./outputs/peft", help="Directory to save the trained model")
    ap.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    ap.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    ap.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    return ap.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    # Load tokenizer dynamically based on the model name
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset and tokenize using the utility function
    ds, collator = load_tokenized_dataset("rotten_tomatoes", "text", tokenizer, args.max_length)
    feat = ds["train"].features.get("labels") or ds["train"].features.get("label")
    num_labels = getattr(feat, "num_classes", len(getattr(feat, "names", [])) or 2)

    # Load model dynamically based on the model name
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    # Apply LoRA to the model
    target_modules = infer_lora_targets(args.model_name)
    lora_config = LoraConfig(
         task_type=TaskType.SEQ_CLS,  # Sequence classification task
         r=args.lora_r,
         lora_alpha=args.lora_alpha,
         lora_dropout=args.lora_dropout,
         target_modules=target_modules,  # Match the architecture
         bias="none",
     )
    model = get_peft_model(model, lora_config)

    # Count trainable parameters using the utility function
    trainable, total, frac = count_trainable_params(model)
    print(f"Efficient FT with LoRA: trainable {trainable:,} / {total:,} ({frac:.2%})")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.model_name.replace("/", "_")),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Train the model and time it

    t0 = time.perf_counter()
    trainer.train()
    total_secs = time.perf_counter() - t0
    print(f"Total training time (seconds): {total_secs:.2f}")

    # Save the model
    trainer.save_model(training_args.output_dir)
    print("Saved to:", training_args.output_dir)


if __name__ == "__main__":
    main()