from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

matplotlib.use("Agg")
import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    print("Usage: python hf_segmentation_test.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])
output_dir = Path("LocalResults")
output_dir.mkdir(exist_ok=True)

# Use the GPU when available; otherwise use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Downloading/loading Hugging Face model...")

# First run downloads approximately 99 MB.
# Later runs use the locally cached copy.
checkpoint_path = hf_hub_download(
    repo_id="mokshhere/skin-lesion-segformer-gate2",
    filename="real_only/best.pt",
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

state_dict = checkpoint["model"]

# Remove the wrapper name used during training.
state_dict = {
    name.removeprefix("model."): weights
    for name, weights in state_dict.items()
}

# Recreate the architecture used during training.
model = smp.Segformer(
    encoder_name="mit_b2",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Load the image.
image_bgr = cv2.imread(str(image_path))

if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
original_height, original_width = image_rgb.shape[:2]

# Resize to the model's required size.
resized = cv2.resize(
    image_rgb,
    (512, 512),
    interpolation=cv2.INTER_AREA,
)

# Apply the same ImageNet normalization used during training.
image_array = resized.astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

image_array = (image_array - mean) / std

image_tensor = (
    torch.from_numpy(image_array)
    .permute(2, 0, 1)
    .unsqueeze(0)
    .to(device)
)

# Generate a lesion probability map.
with torch.inference_mode():
    logits = model(image_tensor)
    probability_512 = torch.sigmoid(logits)[0, 0].cpu().numpy()

# Restore the prediction to the original image dimensions.
probability = cv2.resize(
    probability_512,
    (original_width, original_height),
    interpolation=cv2.INTER_LINEAR,
)

# Pixels with at least 50% probability become lesion pixels.
mask = (probability >= 0.5).astype(np.uint8) * 255

# Create a visual overlay.
overlay = image_rgb.copy()
inside_lesion = mask > 0

overlay[inside_lesion] = (
    0.55 * image_rgb[inside_lesion]
    + 0.45 * np.array([255, 0, 0])
).astype(np.uint8)

# Draw the predicted boundary in green.
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

# Save outputs.
stem = image_path.stem

mask_path = output_dir / f"{stem}_hf_mask.png"
overlay_path = output_dir / f"{stem}_hf_overlay.png"
comparison_path = output_dir / f"{stem}_hf_comparison.png"

cv2.imwrite(str(mask_path), mask)
cv2.imwrite(
    str(overlay_path),
    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
)

figure, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image_rgb)
axes[0].set_title("Original")

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Hugging Face mask")

axes[2].imshow(overlay)
axes[2].set_title("Predicted lesion boundary")

for axis in axes:
    axis.axis("off")

figure.tight_layout()
figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
plt.close(figure)

lesion_percent = 100 * np.count_nonzero(mask) / mask.size

print(f"Lesion area: {lesion_percent:.2f}% of image")
print(f"Saved mask: {mask_path}")
print(f"Saved overlay: {overlay_path}")
print(f"Saved comparison: {comparison_path}")