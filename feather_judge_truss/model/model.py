# model.py
from flow_judge import EvalInput, FlowJudge
from flow_judge.metrics import CustomMetric, RubricItem


class Model:
    def __init__(self, **kwargs):
        """
        This runs once when the model container is initialized.
        Avoid heavy loading here; Baseten will call `load()` next.
        """
        self.judge = None
        self.metric = None

    def load(self):
        """
        Called once at startup. Initialize your metric and FlowJudge here.
        You can also load weights or any other setup resources.
        """
        # Define your custom metric
        continuity_metric = CustomMetric(
            name="Story Continuity",
            criteria=(
                "Evaluate how well the current text continues a story from the previous text. "
                "Refer to the output as current text and the input as previous text. "
                "Do not refer to an input or output."
            ),
            rubric=[
                RubricItem(score=1, description="No continuity. Very different in theme, tone, and content. New elements do not make sense in the context of the story."),
                RubricItem(score=2, description="Poor continuity. Somewhat different in theme, tone, and content. New elements do not make sense in the context of the story."),
                RubricItem(score=3, description="Some continuity. Somewhat aligned and somewhat different in theme, tone, and content. New elements make sense in the context of the story."),
                RubricItem(score=4, description="Good continuity. Aligned in theme, tone, and content. New elements make sense in the context of the story."),
                RubricItem(score=5, description="Excellent continuity. Very aligned in theme, tone, and content. New elements make sense in the context of the story."),
            ],
            required_inputs=["previous_text"],
            required_output="current_text",
        )

        # Initialize FlowJudge (using default local inference, not Baseten adapter)
        self.judge = FlowJudge(metric=continuity_metric)
        self.metric = continuity_metric

        print("✅ FlowJudge model loaded successfully in Baseten environment.")

    def predict(self, model_input):
        """
        Called for each inference request.
        Expects a JSON payload like:
        {
            "previous_text": "...",
            "current_text": "..."
        }
        Returns a dictionary with feedback and score.
        """
        if not self.judge:
            raise RuntimeError("Model not loaded. Ensure `load()` was called before prediction.")

        # Extract user input
        previous_text = model_input.get("previous_text", "")
        current_text = model_input.get("current_text", "")

        # Build evaluation input
        eval_input = EvalInput(
            inputs=[{"previous_text": previous_text}],
            output={"current_text": current_text},
        )

        # Run the evaluation
        result = self.judge.evaluate(eval_input, save_results=False)

        # Return feedback and score
        return {
            "feedback": result.feedback,
            "score": result.score,
        }
