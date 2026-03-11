import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Folder where GradCAM images are stored
heatmap_dir = "gradcam_outputs/"

# Image names you want to compare
kaggle_img = "both_correct_malicious_kaggle.png"
balanced_img = "both_correct_malicious_balanced.png"


def load_heatmap(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Image not found: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0

    return gray


# Load heatmaps
kaggle_map = load_heatmap(os.path.join(heatmap_dir, kaggle_img))
balanced_map = load_heatmap(os.path.join(heatmap_dir, balanced_img))


# -------------------------------------
# 1️⃣ Activation Intensity Distribution
# -------------------------------------

plt.figure(figsize=(7,5))

plt.hist(kaggle_map.flatten(), bins=50, alpha=0.6, label="Kaggle Model")
plt.hist(balanced_map.flatten(), bins=50, alpha=0.6, label="Balanced Model")

plt.xlabel("Activation Intensity")
plt.ylabel("Pixel Count")
plt.title("Grad-CAM Activation Distribution")
plt.legend()

plt.show()


# -------------------------------------
# 2️⃣ Attention Spread Comparison
# -------------------------------------

threshold = 0.6

kaggle_spread = np.sum(kaggle_map > threshold) / kaggle_map.size
balanced_spread = np.sum(balanced_map > threshold) / balanced_map.size

models = ["Kaggle Model", "Balanced Model"]
values = [kaggle_spread, balanced_spread]

plt.figure(figsize=(6,4))

plt.bar(models, values, color=["#ff6b6b", "#4ecdc4"])

plt.ylabel("Activated Area Ratio")
plt.title("Grad-CAM Attention Spread")

plt.show()


# -------------------------------------
# 3️⃣ Activation Profile Curve
# -------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(curve, window=10):
    return np.convolve(curve, np.ones(window)/window, mode='same')

profile_kaggle = np.mean(kaggle_map, axis=0)
profile_balanced = np.mean(balanced_map, axis=0)

profile_kaggle = smooth_curve(profile_kaggle)
profile_balanced = smooth_curve(profile_balanced)

plt.figure(figsize=(8,4))

plt.plot(profile_kaggle, label="Kaggle Model", color="red", linewidth=2)
plt.plot(profile_balanced, label="Balanced Model", color="blue", linewidth=2)

plt.xlabel("QR Code Width")
plt.ylabel("Activation Intensity")
plt.title("Smoothed Grad-CAM Activation Profile")

plt.legend()
plt.grid(True)

plt.show()