# Melanoma-Detection
4730 Semester Project

How to run:

    1. Install dependencies
        pip install -r requirements.txt

    2. Run
        python main.py <path to image> [path to save result]

    3. Example
        python main.py Images/Malignant/ISIC_0000154.jpg Results/result.png

Notes:
- This repository's runnable entrypoint is the top-level `main.py` which
  delegates to the original pipeline in the `Code` directory. Run commands
  from the repository root so the example paths (e.g. `Images/...`) resolve.

shree-huggingface is using https://huggingface.co/DevBhuyan/Skin-Lesion-Segmentation model