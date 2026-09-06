import json
from copy import deepcopy
from pathlib import Path
from langchain_core.runnables import RunnableLambda
from src.safety import SYSTEM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def greedy_config(config):
    config = deepcopy(config)
    config.do_sample = False
    config.temperature = 1.0
    config.top_p = 1.0
    config.top_k = 50
    return config

class LocalLLM:
    def __init__(self, adapter=None, model_id=MODEL_ID, revision="main", quantized=False):
        if adapter:
            manifest_path = Path(adapter) / "training_manifest.json"
            if not (Path(adapter) / "adapter_config.json").is_file():
                raise ValueError("Adapter ausente; execute o notebook de treinamento.")
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                model_id = manifest["model"]
                revision = manifest.get("resolved_revision") or manifest["requested_revision"]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.identity = {"model": model_id, "revision": revision, "adapter": adapter}
        kwargs = {"revision": revision, "dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        if quantized:
            if not torch.cuda.is_available():
                raise ValueError("Quantização requer GPU CUDA.")
            kwargs.update(device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel
            if not (Path(adapter) / "adapter_config.json").is_file():
                raise ValueError("Adapter ausente; execute o notebook de treinamento.")
            self.model = PeftModel.from_pretrained(self.model, adapter)
        if not quantized:
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def generate(self, messages, max_new_tokens=256):
        import torch
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")
        if inputs.input_ids.shape[1] > 6000:
            raise ValueError("Contexto excede o limite de 6000 tokens.")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         generation_config=greedy_config(self.model.generation_config),
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

def build_chain(llm):
    def prompt(state):
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
            json.dumps({"question": state["question"], "patient": state["patient"],
                        "protocol": state["protocol"]}, ensure_ascii=False)}]
    return RunnableLambda(prompt) | RunnableLambda(llm.generate)

class DemoLLM:
    identity = {"model": "DEMO deterministic", "adapter": None}
    def generate(self, messages):
        return "Demonstração sem LLM: revise as observações e os exames pendentes apresentados. A decisão cabe ao profissional responsável."
