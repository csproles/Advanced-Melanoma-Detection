"""
LLM Explanation Layer — v1
Takes ABCDE pipeline output (dict/JSON) -> plain-language explanation + next steps.

Usage:
    pip install openai python-dotenv
    1. Create a file named ".env" in this same folder
    2. Put one line in it:  OPENAI_API_KEY=your_key_here
    3. python llm_explainer.py

The .env file is excluded from git via .gitignore — never commit it.

Swap `HARDCODED_ABCDE_OUTPUT` for the real pipeline's output dict once the
CV/model side is wired up — `explain_findings()` doesn't care where it came from.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env in this folder and loads it into os.environ

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o"  # swap to "gpt-4o" if you want a stronger model

# ---------------------------------------------------------------------------
# SAFETY-CONSTRAINED SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a plain-language explainer for a skin lesion screening
tool used in a hackathon prototype (not a diagnostic medical device).

You will receive structured ABCDE analysis output (Asymmetry, Border, Color,
Diameter, Evolution) from an image analysis pipeline, expressed as scores or
flags. Your job is to translate that into a calm, clear explanation a
non-expert can understand, plus concrete next steps.

STRICT RULES — follow every one of these, no exceptions:

1. NEVER say or imply "you have [condition]", "this is/isn't cancer", "this is
   benign", or any diagnostic conclusion. You are describing what the ABCDE
   FEATURES show, not what they mean medically.
   - Wrong: "You have an irregular mole that could be melanoma."
   - Right: "The analysis flagged some border irregularity and color
     variation in this spot — these are features a dermatologist typically
     looks at closely."

2. Tone is calm and matter-of-fact. No alarming language ("dangerous",
   "urgent warning", "high risk of cancer"). No false reassurance either
   ("don't worry, it's probably fine"). Just state what was observed.

3. ALWAYS include a recommendation to see a licensed dermatologist or
   healthcare provider for any actual evaluation — regardless of whether the
   scores look "low" or "high". This tool flags patterns; it does not
   evaluate them.

4. Explain each ABCDE component that was flagged in one plain-language
   sentence (what it measures, why it's tracked). Do NOT mention components
   where "flagged" is false — not even briefly, not even as a reassuring
   aside. If a component wasn't flagged, act as if it wasn't mentioned in
   the input at all.

4a. The "D" (diameter) component may be entirely absent from the input if
    the pipeline couldn't detect hair to calibrate its pixel-to-millimeter
    scale. If diameter is missing, say plainly that size could not be
    measured for this image rather than guessing or omitting the gap
    silently.

4b. Component C (color) is calibrated to the person's own skin tone from
    the image itself, not a generic skin-tone default — if useful, this can
    be mentioned as a reason the color assessment is specific to this photo.

5. End with 2-4 concrete, doable next steps (e.g., "photograph the spot
   monthly to track changes", "bring this analysis to a dermatology
   appointment", "note if it itches, bleeds, or changes size").

6. Never mention specific probabilities, percentages, or risk levels unless
   they are provided directly in the input data — do not invent confidence
   numbers.

7. If the input data is incomplete or a component is missing/unclear, say so
   plainly rather than filling in a guess.

Output format: return plain text with exactly two labeled sections, in this
exact order, using these exact headers on their own line:

What the analysis noticed:
[your explanation here]

Suggested next steps:
[your next steps here]

No markdown symbols (no #, no **, no bullet characters) — this is a
screening prototype, not a report from a clinician, and needs to render as
plain text.

EXAMPLE — showing rule 4 in action:

Given this input:
{
  "asymmetry": {"score": 0.34, "flagged": true},
  "border": {"score": 0.62, "flagged": true},
  "diameter_mm": {"value": 8.4, "flagged": false, "measured": true}
}

A CORRECT response mentions only asymmetry and border, and says nothing
about diameter at all — not even "diameter was measured but not flagged."
Silence on an unflagged field is correct; describing it, even neutrally,
is not.
"""

USER_PROMPT_TEMPLATE = """Here is the ABCDE analysis output from the image pipeline:

{abcde_json}

Explain this to the person who uploaded the photo, following your system
instructions exactly.
"""

# ---------------------------------------------------------------------------
# HARDCODED EXAMPLE INPUT — matches Callie's actual pipeline output schema
# (Asymmetry/Border/Color = overlap+circularity+color-spread scores;
#  Diameter = hair-calibrated mm measurement, may be absent; Evolution =
#  not yet implemented in the CV pipeline, always empty for now)
# Thresholds per Callie's slides: A flags > 0.20, B flags > 0.50,
# C flags on high spread OR a dangerous color >8% (forced HIGH if >50%),
# D flags > 10mm (her calibrated threshold, not the standard 6mm figure)
# ---------------------------------------------------------------------------
HARDCODED_ABCDE_OUTPUT = {
    "asymmetry": {"score": 0.34, "flagged": True},
    "border": {"score": 0.62, "flagged": True},
    "color": {
        "spread_high": True,
        "dangerous_color_detected": "blue-gray",
        "dangerous_color_coverage_pct": 12,
        "flagged": True,
    },
    "diameter_mm": {"value": 8.4, "flagged": False, "measured": True},
    "evolution": {"flagged": False, "note": "not yet implemented — no prior-image comparison available"},
}


def explain_findings(abcde_output: dict) -> str:
    """Send ABCDE JSON to the LLM layer and return a plain-language explanation."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    abcde_json=json.dumps(abcde_output, indent=2)
                ),
            },
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    result = explain_findings(HARDCODED_ABCDE_OUTPUT)
    print(result)
