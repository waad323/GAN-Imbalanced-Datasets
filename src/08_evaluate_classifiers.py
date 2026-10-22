"""
Step 8: Classification & Evaluation Pipeline.

Trains a CNN classifier under three experiment modes:
  1. Baseline  — original (imbalanced) data only
  2. Vanilla   — original + Vanilla GAN synthetic data
  3. WGAN      — original + WGAN synthetic data

Evaluates each experiment using Accuracy, Precision, Recall, and F1-Score.
Generates confusion matrices and a side-by-side metric comparison plot.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

# ── Configuration ──
image_size = 64
batch_size = 32
epochs = 10
lr = 1e-3
log_file = './results/Evaluation_Results.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('./results', exist_ok=True)


# ── Logging Utility ──
def log_print(message):
    """Prints to console and appends to log file."""
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")


# ── Visualization Helpers ──
def plot_confusion_matrix(cm, classes, filename, title='Confusion Matrix'):
    """Saves a confusion matrix heatmap as an image."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()
    log_print(f"Saved Confusion Matrix to results/{filename}")


def plot_generated_samples():
    """Saves a preview grid of generated samples from each GAN variant."""
    paths = {
        'Vanilla GAN': './vanilla/GeneratedData/NORMAL',
        'DCGAN': './dcgan/GeneratedData/NORMAL',
        'WGAN': './wgan/GeneratedData/NORMAL'
    }

    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for i, (name, path) in enumerate(paths.items()):
        if not os.path.exists(path):
            log_print(f"Warning: Path not found: {path}")
            continue

        files = [f for f in os.listdir(path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]

        for j, file in enumerate(files):
            from PIL import Image
            try:
                img = Image.open(os.path.join(path, file)).convert('L')
                img = img.resize((image_size, image_size))
                axes[i, j].imshow(img, cmap='gray')
            except Exception:
                pass
            axes[i, j].axis('off')
            if j == 2:
                axes[i, j].set_title(name, fontsize=12)

    plt.savefig("results/generated_samples_preview.png")
    plt.close()
    log_print("Saved generated samples preview to results/generated_samples_preview.png")


def plot_metric_comparison(results):
    """Saves a grouped bar chart comparing metrics across experiments."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    experiments = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, exp in enumerate(experiments):
        vals = [results[exp][m] for m in metrics]
        plt.bar(x + i * width, vals, width, label=exp)

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Baseline vs Vanilla GAN vs WGAN')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig("results/metrics_comparison.png")
    plt.close()
    log_print("Saved metric comparison plot to results/metrics_comparison.png")


# ── CNN Classifier ──
class SimpleCNN(nn.Module):
    """3-block CNN for binary classification (Normal vs Pneumonia)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 64 → 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 32 → 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),        # 16 → 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)       # Binary output
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Data Loading ──
def get_datasets(experiment_mode):
    """
    Returns train and test DataLoaders.
    - 'baseline': original data only
    - 'vanilla':  original + Vanilla GAN synthetic data
    - 'wgan':     original + WGAN synthetic data
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    original_train = datasets.ImageFolder(root='./chest_xray/train', transform=transform)
    test_set = datasets.ImageFolder(root='./chest_xray/test', transform=transform)

    if experiment_mode == 'baseline':
        log_print(f"[{experiment_mode}] Training on original data only "
                  f"({len(original_train)} images)")
        train_set = original_train
    else:
        syn_path = f'./{experiment_mode}/GeneratedData/'
        if not os.path.exists(syn_path):
            raise FileNotFoundError(f"Synthetic data not found: {syn_path}")

        synthetic_data = datasets.ImageFolder(root=syn_path, transform=transform)
        train_set = ConcatDataset([original_train, synthetic_data])
        log_print(f"[{experiment_mode}] Original ({len(original_train)}) + "
                  f"Synthetic ({len(synthetic_data)}) = {len(train_set)} images")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ── Training ──
def train_classifier(experiment_name):
    train_loader, test_loader = get_datasets(experiment_name)
    model = SimpleCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    log_print(f"--- Starting Training: {experiment_name} ---")

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    log_print(f"--- Finished Training: {experiment_name} ---")
    return model, test_loader


# ── Evaluation ──
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Pneumonia'],
        output_dict=True
    )

    return (acc,
            report['Normal']['precision'],
            report['Normal']['recall'],
            report['Normal']['f1-score'],
            cm)


# ── Main Pipeline ──
if __name__ == "__main__":
    # Clear previous log
    with open(log_file, "w") as f:
        f.write(f"Evaluation run: {datetime.datetime.now()}\n\n")

    # 1. Visualize generated samples
    plot_generated_samples()

    # 2. Run all three experiments
    experiments = ['baseline', 'vanilla', 'wgan']
    results = {}

    log_print("STEP 3: CLASSIFICATION & COMPARISON")

    for exp in experiments:
        try:
            model, test_loader = train_classifier(exp)
            acc, prec, rec, f1, cm = evaluate_model(model, test_loader)

            results[exp] = {
                'Accuracy': acc, 'Precision': prec,
                'Recall': rec, 'F1-Score': f1
            }
            plot_confusion_matrix(cm, ['Normal', 'Pneumonia'],
                                 f"cm_{exp}.png", title=f"Confusion Matrix — {exp}")

            log_print(f"Result {exp}: Accuracy={acc:.4f}, F1-Score (Normal)={f1:.4f}")
            log_print("-" * 30)

        except Exception as e:
            log_print(f"Skipping {exp} due to error: {e}")

    # 3. Comparison plot
    if results:
        plot_metric_comparison(results)

    # 4. Summary table
    log_print("\n")
    log_print("=" * 65)
    log_print("FINAL COMPARISON RESULTS (Metric: Normal Class)")
    log_print("=" * 65)
    log_print(f"{'Experiment':<15} | {'Accuracy':<10} | {'Precision':<10} | "
              f"{'Recall':<10} | {'F1-Score':<10}")
    log_print("-" * 65)

    for exp, metrics in results.items():
        log_print(f"{exp:<15} | {metrics['Accuracy']:.4f}     | "
                  f"{metrics['Precision']:.4f}     | {metrics['Recall']:.4f}    | "
                  f"{metrics['F1-Score']:.4f}")

    log_print("=" * 65)
