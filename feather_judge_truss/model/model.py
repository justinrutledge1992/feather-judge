import threading
from typing import Any, Dict

from flow_judge import EvalInput, FlowJudge
from flow_judge.metrics import CustomMetric, RubricItem
from flow_judge.models.huggingface import Hf  # adjust if your fork places it elsewhere

_JUDGE = None
_JUDGE_LOCK = threading.Lock()


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
                    "Some continuity. Somewhat aligned and somewhat different in theme, "
                    "tone, and content. New elements make sense in the context of the story."
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


def _build_judge() -> FlowJudge:
    """
    Build the FlowJudge instance using an HF-backed model.
    Runs inside the Baseten container. No legacy Baseten adapter.
    """

    backend = Hf(
        model_name="flowaicom/Flow-Judge-v0.1-AWQ",  # or your own checkpoint
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
    )

    metric = _build_continuity_metric()

    return FlowJudge(
        metric=metric,
        model=backend,
    )


def load_model():
    """
    Called once at container startup.
    Returns a cached FlowJudge instance.
    """
    global _JUDGE
    with _JUDGE_LOCK:
        if _JUDGE is None:
            print("[FeatherJudge] Initializing FlowJudge...")
            _JUDGE = _build_judge()
    return _JUDGE


def predict(model: FlowJudge, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected request JSON:
        {
          "previous_text": "...",
          "current_text": "..."
        }

    Returns:
        {
          "score": <numeric>,
          "feedback": "<string>"
        }
    """
    if not isinstance(request, dict):
        raise ValueError("Request body must be a JSON object.")

    previous_text = request.get("previous_text")
    current_text = request.get("current_text")

    if not previous_text or not current_text:
        raise ValueError(
            "Both 'previous_text' and 'current_text' must be provided in the request."
        )

    eval_input = EvalInput(
        inputs=[{"previous_text": previous_text}],
        output={"current_text": current_text},
    )

    result = model.evaluate(eval_input, save_results=False)

    return {
        "score": result.score,
        "feedback": result.feedback,
    }
