"""Top-level runner wrapper to make the project runnable from repository root.

This inserts `Code` into `sys.path` so the original `Code/main.py` can be
invoked using the same CLI described in the README.
"""
import sys
import os

ROOT = os.path.dirname(__file__)
CODE_DIR = os.path.join(ROOT, "Code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from main import run_pipeline


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python main.py <image_path> [output_path]")
        print("Example: python main.py Images/Benign/ISIC_0000005.jpg Results/result.png")
        sys.exit(1)
    run_pipeline(sys.argv[1], output_path=sys.argv[2] if len(sys.argv) > 2 else None)
