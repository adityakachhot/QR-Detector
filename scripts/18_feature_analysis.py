"""
Feature Analysis & Statistical Comparison
Analyzes visual and statistical differences between malicious and benign QR codes.
Computes image statistics and uses attribution maps to identify key features.
"""

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =========================================
# IMAGE STATISTICS EXTRACTION
# =========================================

def extract_image_statistics(image_path):
    """Extract hand-crafted features from QR code image."""

    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Contrast and brightness
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    contrast = np.max(gray) - np.min(gray)

    # 2. Edge density (using Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(edge_magnitude > 50)

    # 3. Color distribution
    color_std_b = np.std(img[:, :, 0])
    color_std_g = np.std(img[:, :, 1])
    color_std_r = np.std(img[:, :, 2])
    color_variance = (color_std_b + color_std_g + color_std_r) / 3

    # 4. Histogram properties
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_entropy = -np.sum((hist / hist.sum()) * np.log2(hist / hist.sum() + 1e-10))

    # 5. Black and white pixel ratio
    black_pixels = np.sum(gray < 50) / gray.size
    white_pixels = np.sum(gray > 200) / gray.size
    bw_ratio = black_pixels / (white_pixels + 1e-10)

    # 6. Texture (using Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = np.var(laplacian)

    # 7. Symmetry (horizontal and vertical)
    h_flip = cv2.flip(gray, 1)
    v_flip = cv2.flip(gray, 0)
    h_symmetry = np.mean(gray == h_flip)
    v_symmetry = np.mean(gray == v_flip)

    return {
        "brightness_mean": mean_brightness,
        "brightness_std": std_brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "color_variance": color_variance,
        "histogram_entropy": hist_entropy,
        "black_pixel_ratio": black_pixels,
        "white_pixel_ratio": white_pixels,
        "bw_ratio": bw_ratio,
        "texture_variance": texture_variance,
        "horizontal_symmetry": h_symmetry,
        "vertical_symmetry": v_symmetry
    }

# =========================================
# ANALYZE DATASET
# =========================================

def analyze_features(csv_file, output_csv, label_column="label", sample_size=500):
    """Extract features from dataset images (samples if too large)."""

    data = pd.read_csv(csv_file)

    # Sample if dataset is too large
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    features_list = []

    print(f"Extracting features from {len(data)} images...\n")

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        try:
            stats_dict = extract_image_statistics(row["filepath"])
            if stats_dict:
                stats_dict["filepath"] = row["filepath"]
                stats_dict["label"] = row.get(label_column, "unknown")
                features_list.append(stats_dict)
        except Exception as e:
            print(f"Error processing {row['filepath']}: {e}")

    features_df = pd.DataFrame(features_list)
    features_df.to_csv(output_csv, index=False)

    return features_df

# =========================================
# STATISTICAL ANALYSIS
# =========================================

def statistical_comparison(features_df, save_path="xai_outputs/feature_statistics.txt"):
    """Compare features between malicious and benign QR codes."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    malicious = features_df[features_df["label"] == "malicious"]
    benign = features_df[features_df["label"] == "benign"]

    feature_cols = [col for col in features_df.columns if col not in ["filepath", "label"]]

    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FEATURE ANALYSIS: MALICIOUS vs BENIGN QR CODES\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset: {len(malicious)} Malicious, {len(benign)} Benign\n\n")

        f.write("-" * 80 + "\n")
        f.write("STATISTICAL COMPARISON (t-test)\n")
        f.write("-" * 80 + "\n\n")

        significant_features = []

        for feature in feature_cols:
            mal_vals = malicious[feature].dropna()
            ben_vals = benign[feature].dropna()

            # T-test
            t_stat, p_value = stats.ttest_ind(mal_vals, ben_vals)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((mal_vals.std()**2 + ben_vals.std()**2) / 2)
            cohens_d = (mal_vals.mean() - ben_vals.mean()) / pooled_std

            is_significant = p_value < 0.05

            if is_significant:
                significant_features.append({
                    "feature": feature,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "malicious_mean": mal_vals.mean(),
                    "benign_mean": ben_vals.mean()
                })

            f.write(f"\n{feature.upper()}\n")
            f.write(f"  Malicious: Mean={mal_vals.mean():.4f}, Std={mal_vals.std():.4f}\n")
            f.write(f"  Benign:    Mean={ben_vals.mean():.4f}, Std={ben_vals.std():.4f}\n")
            f.write(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.6f}\n")
            f.write(f"  Cohen's d: {cohens_d:.4f}")
            f.write(f"  {'*** SIGNIFICANT ***' if is_significant else ''}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("TOP DISCRIMINATIVE FEATURES (by Cohen's d)\n")
        f.write("=" * 80 + "\n\n")

        sig_df = pd.DataFrame(significant_features).sort_values("cohens_d", key=abs, ascending=False)

        for idx, row in sig_df.head(10).iterrows():
            direction = "Higher in Malicious" if row["cohens_d"] > 0 else "Higher in Benign"
            f.write(f"{row['feature']:.<30} d={row['cohens_d']:>7.4f} (p={row['p_value']:.6f})\n")
            f.write(f"  {direction}: Mal={row['malicious_mean']:.4f}, Ben={row['benign_mean']:.4f}\n\n")

    print(f"✓ Saved statistical analysis: {save_path}")

    return pd.DataFrame(significant_features)

# =========================================
# VISUALIZATION
# =========================================

def visualize_feature_comparison(features_df, save_path="xai_outputs/feature_comparison.png"):
    """Create comprehensive feature comparison visualization."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    malicious = features_df[features_df["label"] == "malicious"]
    benign = features_df[features_df["label"] == "benign"]

    feature_cols = [col for col in features_df.columns if col not in ["filepath", "label"]]

    # Select top features for visualization
    top_features = [
        "brightness_mean", "edge_density", "contrast",
        "bw_ratio", "histogram_entropy", "texture_variance"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(top_features[:6]):
        ax = axes[idx]

        # Violin plot
        data_to_plot = [benign[feature].dropna(), malicious[feature].dropna()]
        parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Benign", "Malicious"])
        ax.set_ylabel(feature.replace("_", " ").title())
        ax.set_title(f"{feature.replace('_', ' ').title()}")
        ax.grid(axis="y", alpha=0.3)

        # Perform t-test
        t_stat, p_val = stats.ttest_ind(malicious[feature].dropna(), benign[feature].dropna())
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        ax.text(0.5, 0.95, f"p={p_val:.4f} {significance}", transform=ax.transAxes,
                ha="center", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved feature comparison: {save_path}")

def plot_feature_importance_heatmap(features_df, save_path="xai_outputs/feature_correlation_heatmap.png"):
    """Plot correlation between features and labels."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create binary label
    features_df["is_malicious"] = (features_df["label"] == "malicious").astype(int)

    feature_cols = [col for col in features_df.columns if col not in ["filepath", "label", "is_malicious"]]

    # Calculate correlation with label
    correlations = []
    for feature in feature_cols:
        corr = features_df[feature].corr(features_df["is_malicious"])
        correlations.append(corr)

    # Sort by absolute correlation
    sorted_indices = np.argsort(np.abs(correlations))[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sorted_features = [feature_cols[i] for i in sorted_indices]
    sorted_corrs = [correlations[i] for i in sorted_indices]

    colors = ["red" if x > 0 else "blue" for x in sorted_corrs]
    ax.barh(range(len(sorted_features)), sorted_corrs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels([f.replace("_", " ").title() for f in sorted_features])
    ax.set_xlabel("Correlation with Malicious Label")
    ax.set_title("Feature Importance: Correlation with QR Code Maliciousness")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved correlation heatmap: {save_path}")

# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":
    os.makedirs("xai_outputs", exist_ok=True)

    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS: MALICIOUS vs BENIGN QR CODES")
    print("=" * 80 + "\n")

    # Step 1: Extract features
    print("Step 1: Extracting features from training set...\n")
    train_features = analyze_features(
        "exp3_train_unified.csv",
        "xai_outputs/train_set_features.csv"
    )

    print(f"\nSuccessfully extracted features from {len(train_features)} images")

    # Step 2: Statistical analysis
    print("\nStep 2: Performing statistical analysis...\n")
    statistical_comparison(train_features, "xai_outputs/feature_statistics.txt")

    # Step 3: Visualizations
    print("\nStep 3: Creating visualizations...\n")
    visualize_feature_comparison(train_features, "xai_outputs/feature_comparison.png")
    plot_feature_importance_heatmap(train_features, "xai_outputs/feature_correlation_heatmap.png")

    print("\n" + "=" * 80)
    print("✓ Feature analysis complete!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - xai_outputs/train_set_features.csv (extracted features)")
    print(f"  - xai_outputs/feature_statistics.txt (statistical comparison)")
    print(f"  - xai_outputs/feature_comparison.png (feature distributions)")
    print(f"  - xai_outputs/feature_correlation_heatmap.png (importance ranking)")
