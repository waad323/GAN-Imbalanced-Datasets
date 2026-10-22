"""
Step 1: Visualize Class Imbalance in the Chest X-Ray Dataset.

Loads the training set and plots a bar chart showing the count of
NORMAL vs PNEUMONIA images to highlight the class imbalance problem.

Dataset: Chest X-Ray Images (Pneumonia) — Kaggle
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = './chest_xray/train'

try:
    dataset = datasets.ImageFolder(root=data_dir)
    labels = torch.tensor(dataset.targets)
    class_names = dataset.classes
    counts = torch.bincount(labels)

    print(f"Classes Found: {class_names}")
    print(f"Counts: {counts.tolist()}")

    plt.figure(figsize=(8, 6))

    ax = sns.barplot(
        x=class_names, y=counts.tolist(),
        hue=class_names, palette=['#1f77b4', '#d62728'], legend=False
    )

    plt.title('Class Distribution in Training Set', fontsize=14)
    plt.ylabel('Number of Images')
    plt.grid(True, axis='y', ls='--', alpha=0.7)

    for index, count in enumerate(counts):
        ax.text(index, count + 20, str(int(count)), ha='center', fontweight='bold')

    plt.savefig('results/Imbalance_Distribution.jpg')
    print("Distribution plot saved to results/Imbalance_Distribution.jpg")

except FileNotFoundError:
    print(f"Error: The directory '{data_dir}' was not found.")
    print("Please download the dataset from Kaggle and place it in ./chest_xray/")
