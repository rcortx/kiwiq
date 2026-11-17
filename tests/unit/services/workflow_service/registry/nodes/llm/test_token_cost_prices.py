"""
Token prices used by the LLM node and prompt compaction (llm_prices.compat.tokencost).

The node prices a call with calculate_cost_by_tokens before falling back to the model
metadata prices in config.py, so the library and the metadata must agree wherever the
metadata lists a price.
"""

import pytest

from llm_prices.compat.tokencost import calculate_cost_by_tokens

from workflow_service.registry.nodes.llm.config import (
    AnthropicModels,
    AWSBedrockModels,
    LLMModelProvider,
    OpenAIModels,
    PerplexityModels,
)

# Same prefixes as LLMNode.map_model_to_tokencost_format.
PREFIX = {LLMModelProvider.AWS_BEDROCK: "bedrock/", LLMModelProvider.PERPLEXITY: "perplexity/"}


def per_million(model: str, token_type: str) -> float:
    return float(calculate_cost_by_tokens(1_000_000, model, token_type))


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    [m for enum in (OpenAIModels, AnthropicModels) for m in enum if m.metadata.input_token_price_per_M > 0],
    ids=lambda m: m.value,
)
def test_library_price_matches_model_metadata(model):
    assert per_million(model.value, "input") == pytest.approx(model.metadata.input_token_price_per_M)
    assert per_million(model.value, "output") == pytest.approx(model.metadata.output_token_price_per_M)


@pytest.mark.unit
def test_claude_3_5_haiku_is_priced_at_its_list_price():
    # tokencost 0.1.26 priced claude-3-5-haiku-latest at $1 / $5, above the metadata's $0.80 / $4.
    assert per_million(AnthropicModels.CLAUDE_3_5_HAIKU.value, "input") == pytest.approx(0.8)
    assert per_million(AnthropicModels.CLAUDE_3_5_HAIKU.value, "output") == pytest.approx(4.0)


@pytest.mark.unit
@pytest.mark.parametrize("model", ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"])
def test_claude_4_5_cache_reads_are_priced(model):
    # tokencost 0.1.26 raised KeyError for both, so every call fell back to the metadata.
    assert per_million(model, "cached") == pytest.approx(per_million(model, "input") / 10)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    [m for enum in (AWSBedrockModels, PerplexityModels) for m in enum if m.value != "r1-1776"],
    ids=lambda m: m.value,
)
def test_bedrock_and_perplexity_ids_are_priced(model):
    tokencost_model = PREFIX[model.metadata.provider] + model.value
    assert per_million(tokencost_model, "input") > 0
    assert per_million(tokencost_model, "output") > 0
