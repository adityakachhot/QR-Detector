"""
Model Uncertainty Estimation
Uses MC-Dropout to estimate prediction uncertainty and confidence intervals.
Identifies high-confidence vs uncertain predictions for malicious QR detection.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================
# DEVICE
# =========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================
# LOAD MODEL
# =========================================

def load_model(weight_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    return model

model = load_model("efficientnet_kaggle.pth")

# =========================================
# ENABLE MC-DROPOUT
# =========================================

def enable_dropout(model):
    """Enable dropout layers during inference for MC-Dropout estimation."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def disable_dropout(model):
    """Disable dropout for standard evaluation."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()

# =========================================
# TRANSFORM
# =========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================
# MC-DROPOUT PREDICTION
# =========================================

def predict_with_uncertainty(image_path, num_samples=50):
    """
    Predict with uncertainty using MC-Dropout.
    Multiple forward passes with dropout enabled give different predictions.

    Args:
        image_path: Path to input image
        num_samples: Number of forward passes to perform

    Returns:
        Dict with mean prediction, std, confidence interval, and entropy
    """

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    predictions = []

    # Enable dropout
    enable_dropout(model)

    with torch.no_grad():
        for _ in range(num_samples):
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()
            predictions.append(prob)

    # Disable dropout
    disable_dropout(model)

    predictions = np.array(predictions)

    # Calculate statistics
    mean_pred = predictions.mean()
    std_pred = predictions.std()
    confidence_lower = np.percentile(predictions, 2.5)
    confidence_upper = np.percentile(predictions, 97.5)

    # Calculate entropy (higher entropy = more uncertain)
    # Use binary entropy: H = -p*log(p) - (1-p)*log(1-p)
    epsilon = 1e-10
    entropy = -mean_pred * np.log(mean_pred + epsilon) - (1 - mean_pred) * np.log(1 - mean_pred + epsilon)

    # Prediction class
    pred_class = "Malicious" if mean_pred > 0.5 else "Benign"

    # Confidence (distance from decision boundary)
    confidence_score = abs(mean_pred - 0.5) * 2

    return {
        "filepath": image_path,
        "mean_probability": mean_pred,
        "std_dev": std_pred,
        "ci_lower": confidence_lower,
        "ci_upper": confidence_upper,
        "entropy": entropy,
        "prediction": pred_class,
        "confidence_score": confidence_score,
        "num_samples": num_samples
    }

# =========================================
# ANALYZE TEST SET WITH UNCERTAINTY
# =========================================

def analyze_dataset_with_uncertainty(csv_file, output_csv, num_samples=50, sample_size=200):
    """Analyze dataset with uncertainty estimates (samples if too large)."""

    data = pd.read_csv(csv_file)

    # Sample if dataset is too large
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    results = []

    print(f"Analyzing {len(data)} images with MC-Dropout ({num_samples} samples each)...\n")

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            result = predict_with_uncertainty(row["filepath"], num_samples=num_samples)
            result["ground_truth"] = row.get("label", "unknown")
            results.append(result)
        except Exception as e:
            print(f"Error processing {row['filepath']}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    return results_df

# =========================================
# VISUALIZATION AND STATISTICS
# =========================================

def visualize_uncertainty_analysis(results_df, save_path="xai_outputs/uncertainty_analysis.png"):
    """Create comprehensive uncertainty analysis visualization."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Prediction confidence distribution
    axes[0, 0].hist(results_df["confidence_score"], bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Confidence Score (0-1)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Distribution of Model Confidence")
    axes[0, 0].axvline(results_df["confidence_score"].mean(), color="red", linestyle="--", label="Mean")
    axes[0, 0].legend()

    # 2. Uncertainty (std dev) vs mean prediction
    colors = ["green" if pred == "Benign" else "red" for pred in results_df["prediction"]]
    axes[0, 1].scatter(
        results_df["mean_probability"],
        results_df["std_dev"],
        c=colors,
        alpha=0.6,
        s=50
    )
    axes[0, 1].set_xlabel("Mean Probability")
    axes[0, 1].set_ylabel("Std Deviation (Uncertainty)")
    axes[0, 1].set_title("Uncertainty vs Prediction")
    axes[0, 1].axvline(0.5, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].legend(["Decision Boundary", "Benign", "Malicious"], loc="upper left")

    # 3. Entropy distribution
    axes[1, 0].hist(results_df["entropy"], bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[1, 0].set_xlabel("Entropy (Information Uncertainty)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Distribution of Prediction Entropy")
    axes[1, 0].axvline(np.log(2), color="red", linestyle="--", label="Max Entropy")
    axes[1, 0].legend()

    # 4. Confidence interval width
    ci_width = results_df["ci_upper"] - results_df["ci_lower"]
    axes[1, 1].hist(ci_width, bins=30, edgecolor="black", alpha=0.7, color="purple")
    axes[1, 1].set_xlabel("95% CI Width")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Distribution of Confidence Interval Width")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved uncertainty analysis: {save_path}")

def generate_uncertainty_summary(results_df, save_path="xai_outputs/uncertainty_summary.txt"):
    """Generate summary statistics."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL UNCERTAINTY ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Total predictions: {len(results_df)}\n")
        f.write(f"  Mean confidence: {results_df['confidence_score'].mean():.4f}\n")
        f.write(f"  Std dev of confidence: {results_df['confidence_score'].std():.4f}\n")
        f.write(f"  Mean uncertainty (std): {results_df['std_dev'].mean():.4f}\n")
        f.write(f"  Mean entropy: {results_df['entropy'].mean():.4f}\n\n")

        f.write("PREDICTION BREAKDOWN:\n")
        for pred_class in ["Benign", "Malicious"]:
            subset = results_df[results_df["prediction"] == pred_class]
            f.write(f"\n  {pred_class}:\n")
            f.write(f"    Count: {len(subset)}\n")
            f.write(f"    Mean probability: {subset['mean_probability'].mean():.4f}\n")
            f.write(f"    Mean confidence: {subset['confidence_score'].mean():.4f}\n")
            f.write(f"    Mean uncertainty: {subset['std_dev'].mean():.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("HIGH UNCERTAINTY PREDICTIONS (Top 10):\n")
        f.write("=" * 70 + "\n\n")

        high_uncertainty = results_df.nlargest(10, "entropy")
        for idx, row in high_uncertainty.iterrows():
            f.write(f"Image: {row['filepath']}\n")
            f.write(f"  Prediction: {row['prediction']} ({row['mean_probability']:.2%})\n")
            f.write(f"  Uncertainty (std): {row['std_dev']:.4f}\n")
            f.write(f"  Entropy: {row['entropy']:.4f}\n")
            f.write(f"  95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]\n\n")

    print(f"✓ Saved summary: {save_path}")

# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":
    os.makedirs("xai_outputs", exist_ok=True)

    print("\n" + "=" * 70)
    print("MC-DROPOUT UNCERTAINTY ESTIMATION")
    print("=" * 70 + "\n")

    # Analyze test set
    print("Step 1: Analyzing test set...\n")
    test_results = analyze_dataset_with_uncertainty(
        "exp3_test_unified.csv",
        "xai_outputs/test_set_uncertainty.csv",
        num_samples=20,
        sample_size=200
    )

    # Create visualizations
    print("\nStep 2: Creating visualizations...\n")
    visualize_uncertainty_analysis(test_results, "xai_outputs/uncertainty_analysis.png")

    # Generate summary
    print("\nStep 3: Generating summary statistics...\n")
    generate_uncertainty_summary(test_results, "xai_outputs/uncertainty_summary.txt")

    print("\n" + "=" * 70)
    print("✓ Uncertainty analysis complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - xai_outputs/test_set_uncertainty.csv (all predictions with uncertainty)")
    print(f"  - xai_outputs/uncertainty_analysis.png (visualizations)")
    print(f"  - xai_outputs/uncertainty_summary.txt (statistics)")
