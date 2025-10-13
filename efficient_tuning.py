import os
import argparse

from transformers import Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, TaskType

from Model import (
    build_tokenizer,
    load_tokenized_agnews,
    build_model,
    make_compute_metrics,
    count_trainable_params,
    infer_lora_targets,
)

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="./outputs/peft")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    return ap.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)

    tokenizer = build_tokenizer(args.model_name)
    ds, collator, num_labels = load_tokenized_agnews(tokenizer, args.max_length)
    base_model = build_model(args.model_name, num_labels)

    target_modules = infer_lora_targets(args.model_name)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)

    trainable, total, frac = count_trainable_params(model)
    print(f"PEFT (LoRA): trainable {trainable:,} / {total:,} ({frac:.2%})")
    model.print_trainable_parameters()

    out_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    training_args = TrainingArguments(
        output_dir=out_dir,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval:", metrics)
    # Saves LoRA adapters (and base config refs)
    trainer.save_model(out_dir)
    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()