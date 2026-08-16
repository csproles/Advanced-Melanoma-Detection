# Melanoma Detection

## What it does

Upload a photo of a skin lesion and the app runs it through a classical (non-ML)
image-processing pipeline that mirrors the ABCDE rule dermatologists use for a
quick visual melanoma screen:

- **A**symmetry, **B**order irregularity, **C**olor variation, **D**iameter,
  **E**volving (always reported as "not assessable" — a single photo can't show
  change over time)

Each gets a 0–10 score from real image measurements (contour geometry, k-means
color clustering, etc.), combined into one 0–100 risk indicator. The pipeline
also runs denoising, hair removal, and lesion segmentation, and the UI shows
each intermediate image alongside the scores.

There are two parts:

| Component | Tech | Port |
|---|---|---|
| `MelanomaDetection.Web` | Blazor Server (.NET 10) | `https://localhost:7001` |
| `MelanomaDetection.Python` | Flask + OpenCV | `http://localhost:5002` |

The Blazor app is the UI; it calls the Flask API to do the actual image
processing.

## Setup (quick)

Prerequisites: [.NET 10 SDK](https://dotnet.microsoft.com/download) and Python 3.

**1. Start the Python API:**

```powershell
cd MelanomaDetection.Python
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# venv\Scripts\activate.bat    # Windows Command Prompt
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python main.py
```

If PowerShell blocks the activation script with an execution-policy error, run
`Set-ExecutionPolicy -Scope Process RemoteSigned` first (session-only, doesn't
change any system setting).

Leave this running — it serves the API on `http://localhost:5002`. Confirm it's
up:

```bash
curl http://localhost:5002/health
# {"status":"healthy"}
```

**2. Start the Blazor app** (in a second terminal):

```bash
cd MelanomaDetection.Web
dotnet run --launch-profile https
```

Open `https://localhost:7001` in a browser. Both processes need to stay running
— the Blazor app calls the Flask API over HTTP for every image analysis.

## How to use

1. Go to `https://localhost:7001` and click **Start Analysis** (or go straight
   to `/upload`).
2. Select a lesion photo — JPEG, PNG, BMP, or WebP, up to 5MB.
3. Confirm the preview looks right, then click **Analyze Image**.
4. You're redirected to `/results/{id}`, showing:
   - The processed images (original, bilateral-filtered, noise-removed, edges)
   - Each ABCDE sub-score (0–10) with a color-coded progress bar
   - An overall risk score (0–100) with a color-coded alert and a clinical
     recommendation
5. Click **Analyze Another Image** to go again.

No test images on hand? A labeled set from the ISIC archive lives in
`../Images/Benign/` and `../Images/Malignant/` at the repo root.

## Medical disclaimer

**Educational use only. This is NOT a substitute for professional medical
diagnosis. Consult a dermatologist.** The risk score is a heuristic derived
from classical image-processing measurements, not a validated diagnostic tool,
and has not been cleared or approved by any regulatory body. If you notice a
new, changing, or unusual mole, see a dermatologist regardless of what this
tool reports.

## Limitations

- **Measured accuracy: 5/10 (50%) on a small labeled ISIC sample** — see
  `MelanomaDetection.Python/validate_accuracy.py`. This is a real, honest
  ceiling of the current heuristic approach, not a rounding error.
- **No trained ML model.** Every score comes from hand-tuned classical CV
  thresholds (Otsu segmentation, contour compactness, k-means color
  clustering), not a model trained on labeled data. Two of the four scored
  ABCDE letters (border, color) are known to saturate near their maximum for
  many real images, which limits how well they discriminate benign from
  malignant.
- **"Evolving" is never scored.** A single static photo can't show change over
  time, so this always reports as not assessable — by design, not a bug.
- **Diameter is relative, not physical.** There's no camera calibration, so the
  diameter score reflects the lesion's size relative to the photo frame, not
  an actual millimeter measurement.
- **In-memory results storage.** The Flask API keeps processed results in a
  plain Python dict — restarting `main.py` clears all history, and this isn't
  meant to run as a shared multi-user service.
- **No persistence/database.** Nothing is saved between sessions; there's no
  history of past uploads.
- **Local-only, unauthenticated.** Both servers assume localhost and have no
  auth — don't expose either port to the public internet as-is.

## Future enhancements

- Replace or augment the heuristic scorer with a trained ML model (the
  original design explicitly deferred this — see `image_processor.py`'s class
  docstring)
- Recalibrate ABCDE thresholds against a much larger labeled dataset than the
  10-image validation set used so far
- Camera/scale calibration for a real physical diameter measurement
- Persist results to a database instead of in-memory storage, with a history
  view
- User accounts and authentication if this ever needs to run beyond a single
  local machine
- Batch upload / compare multiple lesions over time (would also make
  "Evolving" finally assessable)
