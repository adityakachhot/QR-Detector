"""
LIME (Local Interpretable Model-agnostic Explanations)
Explains individual predictions by perturbing input images locally
and showing which superpixels are most important for the decision.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

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
    model.eval()
    return model

model = load_model("efficientnet_kaggle.pth")

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
# PREDICTION FUNCTION FOR LIME
# =========================================

def predict_fn(images):
    """
    Wrapper function for LIME that accepts numpy arrays and returns probabilities.
    LIME will perturb images and call this repeatedly.
    """
    predictions = []

    for img_array in images:
        # Convert numpy array to tensor
        if img_array.max() > 1:
            img_array = img_array / 255.0

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img_tensor)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).item()

        predictions.append([1 - prob, prob])  # [benign_prob, malicious_prob]

    return np.array(predictions)

# =========================================
# GENERATE LIME EXPLANATION
# =========================================

def generate_lime_explanation(image_path, save_path, label_name=""):
    """Generate LIME explanation marking important superpixels."""

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # Get model prediction
    pred_probs = predict_fn(np.array([image_array]))
    pred_class = np.argmax(pred_probs[0])
    confidence = pred_probs[0][pred_class]

    class_names = ["Benign", "Malicious"]

    print(f"Processing: {os.path.basename(image_path)}")
    print(f"  Prediction: {class_names[pred_class]} ({confidence:.2%})")

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation (this calls predict_fn many times)
    explanation = explainer.explain_instance(
        image_array,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=50  # Reduced for speed, increase for accuracy
    )

    # Get the explanation for the predicted class
    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=True,
        num_features=10,  # Top 10 important regions
        hide_rest=False
    )

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_array)
    axes[0].set_title(f"Original Image\n{label_name}")
    axes[0].axis("off")

    # Mask of important regions
    axes[1].imshow(mark_boundaries(image_array / 255.0, mask))
    axes[1].set_title(f"Important Regions (Top 10)\nPrediction: {class_names[pred_class]} ({confidence:.2%})")
    axes[1].axis("off")

    # Highlighted important superpixels
    axes[2].imshow(temp)
    axes[2].set_title("LIME Explanation\n(Green = Important for class)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved: {save_path}\n")

    return {
        "image": os.path.basename(image_path),
        "label": label_name,
        "prediction": class_names[pred_class],
        "confidence": f"{confidence:.4f}",
        "top_features": 10
    }

# =========================================
# PROCESS TEST SAMPLES
# =========================================

os.makedirs("xai_outputs/lime", exist_ok=True)

samples = pd.read_csv("gradcam_selected_samples.csv")

results = []

for idx, row in samples.iterrows():
    img_path = row["filepath"]
    name = row["type"]

    save_path = f"xai_outputs/lime/{idx}_{name}_lime_explanation.png"

    try:
        result = generate_lime_explanation(img_path, save_path, label_name=name)
        results.append(result)
    except Exception as e:
        print(f"Error processing {img_path}: {e}\n")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("xai_outputs/lime_results.csv", index=False)
print(f"\n✓ Saved {len(results)} LIME explanations")
print(f"✓ Results summary: xai_outputs/lime_results.csv")
