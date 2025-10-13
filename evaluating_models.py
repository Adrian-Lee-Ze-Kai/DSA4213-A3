from model import load_tokenized_dataset, make_compute_metrics

def evaluate_model(model_dir, tokenizer_dir, dataset_name, max_length=256):
    """
    Evaluates a saved model on the specified dataset.

    Args:
        model_dir (str): Path to the saved model directory.
        tokenizer_dir (str): Path to the tokenizer directory.
        dataset_name (str): Name of the dataset to evaluate on.
        max_length (int): Maximum token length.

    Returns:
        dict: Evaluation metrics (accuracy, F1, MCC).
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Load and tokenize the dataset
    dataset, collator = load_tokenized_dataset(dataset_name, "text", tokenizer, max_length)
    dataset = dataset["test"]  # Use the test split

    # Prepare inputs for evaluation
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, collate_fn=collator
    )

    # Evaluate the model
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(model.device)
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    compute_metrics = make_compute_metrics(["accuracy", "f1", "matthews_correlation"])
    metrics = compute_metrics((np.array(all_preds), np.array(all_labels)))
    return metrics