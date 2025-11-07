# model.py
#
# Truss entrypoint for Feather-Judge / Flow-Judge style evaluation on Baseten.
# - Uses a custom "Story Continuity" metric.
# - Uses an HF-backed model wrapper (no legacy Baseten adapter).
# - Exposes a simple JSON API:
#       {
#           "previous_text": "...",
#           "current_text": "..."
#       }
#   ->
#       {
#           "score": <int|float>,
#           "feedback": "<string>"
#       }

from typing import Any, Dict
import threading

from flow_judge import EvalInput, FlowJudge  # from your feather-judge fork
from flow_judge.metrics import CustomMetric, RubricItem
from flow_judge.models.huggingface import Hf  # HF backend, safe & local to the container

# If your fork exposes Hf somewhere slightly different, update this import accordingly.


# ---------- Global singleton ----------

_JUDGE = None
_JUDGE_LOCK = threading.Lock()


# ---------- Metric / rubric ----------

def _build_continuity_metric() -> CustomMetric:
    return CustomMetric(
        name="Story Continuity",
        criteria=(
            "Evaluate how well the current text continues a story from the previous text. "
            "Refer to the output as current text and the input as previous text. "
            "Do not refer to an input or output."
        ),
        rubric=[
            RubricItem(
                score=1,
                description=(
                    "No continuity. Very different in theme, tone, and content. "
                    "New elements do not make sense in the context of the story."
                ),
            ),
            RubricItem(
                score=2,
                description=(
                    "Poor continuity. Somewhat different in theme, tone, and content. "
                    "New elements do not make sense in the context of the story."
                ),
            ),
            RubricItem(
                score=3,
                description=(
                    "Some continuity. Somewhat aligned and somewhat different in theme, tone, "
                    "and content. New elements make sense in the context of the story."
                ),
            ),
            RubricItem(
                score=4,
                description=(
                    "Good continuity. Aligned in theme, tone, and content. "
                    "New elements make sense in the context of the story."
                ),
            ),
            RubricItem(
                score=5,
                description=(
                    "Excellent continuity. Very aligned in theme, tone, and content. "
                    "New elements make sense in the context of the story."
                ),
            ),
        ],
        required_inputs=["previous_text"],
        required_output="current_text",
    )


# ---------- Model backend (no Baseten adapter) ----------

def _build_model():
    """
    Construct the underlying judge model WITHOUT using flow_judge's Baseten adapter.

    We use the Hugging Face backend here so that:
    - Truss can download & serve it normally on Baseten infra.
    - We avoid the stale Baseten integration in the original library.
    """

    # Pick a model compatible with your environment.
    # For the actual Flow-Judge weights, adapt to your fork if needed:
    # e.g. "flowaicom/Flow-Judge-v0.1" or your own fine-tuned judge.
    #
    # Keep generation params conservative & deterministic-ish for evaluation.
    model = Hf(
        model_name="flowaicom/Flow-Judge-v0.1",
        # Depending on your fork's Hf signature, you might pass:
        # dtype="bfloat16",
        # device_map="auto",
        # trust_remote_code=True,
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
    )

    metric = _build_continuity_metric()

    judge = FlowJudge(
        metric=metric,
        model=model,
    )

    return judge


# ---------- Truss hooks ----------

def load_model():
    """
    Called once when the Baseten/Truss container starts.

    We build and cache a global FlowJudge instance so subsequent predict()
    calls are cheap.
    """
    global _JUDGE
    with _JUDGE_LOCK:
        if _JUDGE is None:
            _JUDGE = _build_model()
    return _JUDGE


def predict(model, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Truss predict() entrypoint.

    Expected input JSON:
        {
          "previous_text": "USER INPUT 1",
          "current_text": "USER INPUT 2"
        }

    Returns:
        {
          "score": <numeric>,
          "feedback": "<text explanation from the judge model>"
        }
    """

    # Basic validation
    if not isinstance(request, dict):
        raise ValueError("Request body must be a JSON object.")

    previous_text = request.get("previous_text")
    current_text = request.get("current_text")

    if not previous_text or not current_text:
        raise ValueError(
            "Both 'previous_text' and 'current_text' must be provided in the request."
        )

    # Map into EvalInput following your metric's required schema.
    eval_input = EvalInput(
        inputs=[{"previous_text": previous_text}],
        output={"current_text": current_text},
    )

    result = model.evaluate(eval_input, save_results=False)

    # FlowJudge result usually has .score and .feedback; adjust if your fork differs.
    return {
        "score": result.score,
        "feedback": result.feedback,
    }
