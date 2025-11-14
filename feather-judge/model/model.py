import os
import uuid

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

MAX_LENGTH = 1024
TEMPERATURE = 0.1
TOP_P = 0.95
TOP_K = 40
DO_SAMPLE = True
DEFAULT_STREAM = False

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Model:
    """Baseten model class for deployment."""

    def __init__(self, **kwargs):
        """Initialize Baseten model deployment class."""
        self._config = kwargs["config"]
        self.model_id = None
        self.llm_engine = None
        self.model_args = None
        self.hf_secret_token = kwargs["secrets"].get("hf_access_token", None)
        self.vllm_base_url = None
        self._model_metadata = kwargs.get(
            "model_metadata",
            self._config.get("model_metadata", {})  # fallback to config if needed
        )
        os.environ["HF_TOKEN"] = self.hf_secret_token

    def load(self):
        """Load the model."""
        self._model_repo_id = self._model_metadata["repo_id"]
        self._vllm_config = self._model_metadata["vllm_config"]
        if self._vllm_config is None:
            self._vllm_config = {}

        """Load non-OpenAI compatible model."""
        self.model_args = AsyncEngineArgs(model=self._model_repo_id, **self._vllm_config)
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args=self.model_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_repo_id)

    async def predict(self, model_input):
        """Generate output based on the input."""
        if "messages" not in model_input and "prompt" not in model_input:
            raise ValueError("Prompt or messages must be provided")

        stream = model_input.get("stream", False)

        """Generate output for non-OpenAI compatible model."""
        # SamplingParams does not take/use argument 'model'
        if "model" in model_input:
            model_input.pop("model")
        if "prompt" in model_input:
            prompt = model_input.pop("prompt")
            sampling_params = SamplingParams(**model_input)
            idx = str(uuid.uuid4().hex)
            messages = [
                {"role": "user", "content": prompt},
            ]
            # templatize the input to the model
            input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif "messages" in model_input:
            messages = model_input.pop("messages")
            sampling_params = SamplingParams(**model_input)
            idx = str(uuid.uuid4().hex)
            # templatize the input to the model
            input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
        # since we accept any valid vllm sampling parameters, we can just pass it through
        vllm_generator = self.llm_engine.generate(input, sampling_params, idx)

        async def generator():
            full_text = ""
            async for output in vllm_generator:
                text = output.outputs[0].text
                delta = text[len(full_text) :]
                full_text = text
                yield delta

        if stream:
            return generator()
        else:
            full_text = ""
            async for delta in generator():
                full_text += delta
            return {"text": full_text}

