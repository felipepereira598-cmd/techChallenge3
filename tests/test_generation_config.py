from types import SimpleNamespace
from src.llm import greedy_config


def test_greedy_config_preserves_model_tokens_and_original():
    original = SimpleNamespace(do_sample=True, temperature=0.7, top_p=0.8,
                               top_k=20, eos_token_id=[1, 2], repetition_penalty=1.1)
    config = greedy_config(original)
    assert (config.do_sample, config.temperature, config.top_p, config.top_k) == (False, 1.0, 1.0, 50)
    assert config.eos_token_id == [1, 2]
    assert config.repetition_penalty == 1.1
    assert original.do_sample and original.temperature == 0.7
    config.eos_token_id.append(3)
    assert original.eos_token_id == [1, 2]
