import threading
from typing import Any, Dict

from flow_judge import EvalInput, FlowJudge
from flow_judge.metrics import CustomMetric, RubricItem
from flow_judge.models.huggingface import Hf

# Global singleton + lock so the model loads once per container
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
    Build the FlowJudge instance using an HF backend.
    No Baseten adapter. This runs inside the Baseten container.
    """

    # Pick your evaluation model here.
    # You can swap this to your own HF checkpoint or the original Flow-Judge weights.
    # Make sure the corresponding requirement is in requirements.txt.
    model = Hf(
        model_name="flowaicom/Flow-Judge-v0.1-AWQ",
        max_new_tokens=256,
        temperature=0.1,
        top_p=0.9,
    )

    metric = _build_continuity_metric()

    return FlowJudge(
        metric=metric,
        model=model,
    )


def load_model():
    """
    Called once at container startup.
    Returns the FlowJudge instance (cached globally).
    """
    global _JUDGE
    with _JUDGE_LOCK:
        if _JUDGE is None:
            _JUDGE = _build_judge()
    return _JUDGE


def predict(model: FlowJudge, request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baseten/Truss entrypoint.
    Expects:
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

    # Adjust attribute names if your fork differs
    return {
        "score": result.score,
        "feedback": result.feedback,
    }
