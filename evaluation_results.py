import os
import matplotlib.pyplot as plt
from evaluating_models import evaluate_model, evaluate_peft_model

# Paths
base_model = "distilbert-base-uncased"
dataset    = "rotten_tomatoes"   # change if needed
max_len    = 256
batch_size = 16  # match your training batch size

full_dir   = r"outputs\full\distilbert-base-uncased"
peft_dir   = r"outputs\peft\distilbert-base-uncased"
abl_base   = r"outputs\peft_ablation\distilbert-base-uncased"
r_values   = [4, 8, 16]

results = []

# Full fine-tuned model
if os.path.isdir(full_dir):
    m = evaluate_model(full_dir, full_dir, dataset, max_length=max_len, batch_size=batch_size, plot=False)
    results.append(("Full FT", m))

# Baseline PEFT adapter
if os.path.isdir(peft_dir):
    m = evaluate_peft_model(base_model, peft_dir, dataset, max_length=max_len, batch_size=batch_size, plot=False)
    results.append(("PEFT (baseline)", m))

# LoRA-r adapters from ablation
for r in r_values:
    adapter_dir = os.path.join(abl_base, f"r{r}")
    if os.path.isdir(adapter_dir):
        m = evaluate_peft_model(base_model, adapter_dir, dataset, max_length=max_len, batch_size=batch_size, plot=False)
        results.append((f"LoRA r={r}", m))

# Print metrics
for name, m in results:
    print(f"{name:>14} | Acc={m['accuracy']:.4f}  F1_macro={m['f1_macro']:.4f}  MCC={m['mcc']:.4f}")

# Build plot data
labels = [name for name, _ in results]
accs   = [m["accuracy"] for _, m in results]
f1s    = [m["f1_macro"] for _, m in results]
mccs   = [m["mcc"] for _, m in results]

plt.ioff()  # make plt.show() block until the window is closed

def _plot_single(labels, values, title, ylabel):
    plt.figure(figsize=(6, 4))
    palette = ["steelblue", "darkorange", "seagreen"]
    bar_colors = [palette[i % len(palette)] for i in range(len(values))]
    bars = plt.bar(labels, values, color=bar_colors)
    plt.title(title)
    ymin = min(0.0, min(values) - 0.05)
    ymax = max(1.0, max(values) + 0.05)
    plt.ylim(ymin, ymax)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    for b, v in zip(bars, values):
        plt.annotate(f"{v:.3f}", (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                     textcoords="offset points")
    plt.tight_layout()
    plt.show()
    plt.close()

_plot_single(labels, accs, "Accuracy by Model", "Accuracy")
_plot_single(labels, f1s,  "F1 Macro by Model", "F1 Macro")
_plot_single(labels, mccs, "MCC by Model", "MCC")