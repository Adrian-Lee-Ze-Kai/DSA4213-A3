from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
from utilities import load_tokenized_dataset, make_compute_metrics

def evaluate_model(model_dir, tokenizer_dir, dataset_name, max_length=256):
    """
    Evaluates a saved model on the specified dataset and plots metrics over evaluation steps.

    Args:
        model_dir (str): Path to the saved model directory.
        tokenizer_dir (str): Path to the tokenizer directory.
        dataset_name (str): Name of the dataset to evaluate on.
        max_length (int): Maximum token length.

    Returns:
        dict: Final evaluation metrics (accuracy, F1, MCC).
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Load and tokenize the dataset
    dataset, collator = load_tokenized_dataset(dataset_name, "text", tokenizer, max_length)
    dataset = dataset["test"]  # Use the test split

    # Prepare dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collator)

    # Initialize metrics storage
    metrics_over_time = {"accuracy": [], "f1": [], "matthews_correlation": []}
    all_preds = []
    all_labels = []

    # Evaluation loop
    compute_metrics = make_compute_metrics(["accuracy", "f1", "matthews_correlation"])
    for batch in dataloader:
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1)

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

        # Compute metrics for this batch
        metrics = compute_metrics((np.array(all_preds), np.array(all_labels)))

        # Store metrics for plotting
        for metric_name, value in metrics.items():
            metrics_over_time[metric_name].append(value)

    # Plot metrics
    for metric_name, values in metrics_over_time.items():
        plt.plot(values, label=metric_name)
    
    plt.xlabel("Evaluation Step")
    plt.ylabel("Metric Value")
    plt.title("Metrics Over Evaluation Steps")
    plt.legend()
    plt.show()

    # Return final metrics
    return metrics


def compare_models(metrics_model1, metrics_model2, model1_name="Model 1", model2_name="Model 2"):
    """
    Compares the metrics of two models and plots them side by side.

    Args:
        metrics_model1 (dict): Metrics of the first model (e.g., {"accuracy": [...], "f1": [...]}).
        metrics_model2 (dict): Metrics of the second model (same structure as metrics_model1).
        model1_name (str): Name of the first model (for labeling).
        model2_name (str): Name of the second model (for labeling).
    """
    metric_names = metrics_model1.keys()
    num_metrics = len(metric_names)

    # Create subplots for each metric
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]  # Ensure axes is iterable for a single metric

    for ax, metric_name in zip(axes, metric_names):
        ax.plot(metrics_model1[metric_name], label=model1_name, color="blue")
        ax.plot(metrics_model2[metric_name], label=model2_name, color="orange")
        ax.set_title(metric_name.capitalize())
        ax.set_xlabel("Evaluation Step")
        ax.set_ylabel("Metric Value")
        ax.legend()

    plt.tight_layout()
    plt.show()