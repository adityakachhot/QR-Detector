import cv2
import matplotlib.pyplot as plt
import os

# Folder where GradCAM outputs are stored
heatmap_dir = "gradcam_outputs"

# Image pairs (Kaggle vs Balanced)
samples = [
    ("both_correct_benign"),
    ("both_correct_malicious"),
    ("kaggle_wrong_balanced_correct_benign"),
    ("kaggle_wrong_balanced_correct_malicious")
]

rows = len(samples)

plt.figure(figsize=(12, 10))

for i, sample in enumerate(samples):

    kaggle_path = os.path.join(heatmap_dir, f"{sample}_kaggle.png")
    balanced_path = os.path.join(heatmap_dir, f"{sample}_balanced.png")

    kaggle_img = cv2.cvtColor(cv2.imread(kaggle_path), cv2.COLOR_BGR2RGB)
    balanced_img = cv2.cvtColor(cv2.imread(balanced_path), cv2.COLOR_BGR2RGB)

    # Input image approximation (use balanced heatmap without overlay if needed)
    input_img = balanced_img

    # Column 1: Input
    plt.subplot(rows,3,i*3+1)
    plt.imshow(input_img)
    plt.axis("off")
    if i == 0:
        plt.title("Input QR")

    # Column 2: Kaggle
    plt.subplot(rows,3,i*3+2)
    plt.imshow(kaggle_img)
    plt.axis("off")
    if i == 0:
        plt.title("Kaggle Model GradCAM")

    # Column 3: Balanced
    plt.subplot(rows,3,i*3+3)
    plt.imshow(balanced_img)
    plt.axis("off")
    if i == 0:
        plt.title("Balanced Model GradCAM")

plt.tight_layout()
plt.savefig("gradcam_comparison_grid.png", dpi=300)
plt.show()