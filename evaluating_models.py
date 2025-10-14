import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from utilities import load_tokenized_dataset
import matplotlib.pyplot as plt

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


def _collect_preds(model, dataloader, device="cpu"):
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.get("labels", batch.get("label"))
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "label")}
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return y_true, y_pred


def _plot_metrics(metrics: dict, title: str):
    names = list(metrics.keys())
    vals = [float(v) for v in metrics.values()]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, vals, color=["steelblue", "darkorange", "seagreen"])
    # Dynamic y-limits (MCC can be < 0)
    y_min = min(0.0, min(vals) - 0.05)
    y_max = max(1.0, max(vals) + 0.05)
    plt.ylim(y_min, y_max)
    plt.ylabel("Score")
    plt.title(title)
    # Annotate bars with values
    for bar, val in zip(bars, vals):
        h = bar.get_height()
        offset = 4 if h >= 0 else -12
        va = "bottom" if h >= 0 else "top"
        plt.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
        )
    plt.tight_layout()
    plt.show()


def evaluate_model(model_dir, tokenizer_dir, dataset_name, max_length=256, batch_size=32, device=None, plot=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    ds, collator = load_tokenized_dataset(dataset_name, "text", tok, max_length)
    test_dl = DataLoader(ds["test"], batch_size=batch_size, collate_fn=collator)

    y_true, y_pred = _collect_preds(model, test_dl, device=device)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if plot:
        _plot_metrics(metrics, title=f"{dataset_name} — Full model")
    return metrics


def evaluate_peft_model(base_model_id, adapter_dir, dataset_name, max_length=256, batch_size=32, device=None, plot=True):
    assert PeftModel is not None, "peft not installed: pip install peft"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(adapter_dir)  # or base_model_id
    base = AutoModelForSequenceClassification.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base, adapter_dir)
    ds, collator = load_tokenized_dataset(dataset_name, "text", tok, max_length)
    test_dl = DataLoader(ds["test"], batch_size=batch_size, collate_fn=collator)
    y_true, y_pred = _collect_preds(model, test_dl, device=device)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if plot:
        _plot_metrics(metrics, title=f"{dataset_name} — PEFT (LoRA)")
    return metrics