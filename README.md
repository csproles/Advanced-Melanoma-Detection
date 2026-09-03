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

shree-huggingface is using https://huggingface.co/DevBhuyan/Skin-Lesion-Segmentation 

Installation required: 
VS Code extensions: Python, Pylance by Microsoft
Python packages: torch, torchvision, segmentation-models-pytorch, huggingface-hub, opencv-python, numpy, matplotlib, pillow

Installation commands (windows):
python -m venv C:\venvs\melanoma
& "C:\venvs\melanoma\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy opencv-python matplotlib pillow huggingface-hub "segmentation-models-pytorch>=0.5"

Verify installation:
python -c "import cv2, numpy, matplotlib, torch, segmentation_models_pytorch, huggingface_hub; print('All packages are ready')"
