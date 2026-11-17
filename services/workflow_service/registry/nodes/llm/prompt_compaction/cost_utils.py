"""
Cost calculation utilities for prompt compaction.

Provides consistent LLM cost calculation aligned with the main LLM node's billing logic.
Supports OpenAI and Anthropic providers (the only models used for compaction).

Uses tokencost library with model_metadata fallback for accurate pricing.
Handles cached token pricing properly for both providers.
"""

from typing import Dict, Any

from workflow_service.registry.nodes.llm.config import ModelMetadata, LLMModelProvider


def calculate_compaction_llm_cost(
    token_usage: Dict[str, Any],
    provider: LLMModelProvider,
    model_name: str,
    model_metadata: ModelMetadata,
) -> float:
    """
    Calculate actual LLM cost for compaction calls (OpenAI/Anthropic only).
    
    Uses tokencost library with model_metadata fallback for accurate pricing.
    Handles cached tokens for both providers following the same logic as
    the main LLM node's _calculate_actual_cost() method.
    
    This is a simplified version of llm_node.py's _calculate_actual_cost() (lines 3228-3328)
    tailored for compaction use cases - no Perplexity, reasoning tokens, citation tokens,
    or web search costs since compaction only uses OpenAI and Anthropic models.
    
    Args:
        token_usage: Token usage dict with input_tokens, output_tokens, cached_tokens
        provider: LLM provider (OpenAI or Anthropic)
        model_name: Model name used for the LLM call
        model_metadata: Model metadata with pricing information
    
    Returns:
        Cost in USD (pre-markup, markup is applied in billing.py)
    
    Raises:
        Exception: If cost calculation fails completely
    
    Note:
        This returns the pre-markup cost. The billing layer applies
        settings.LLM_TOKEN_COST_MARKUP_FACTOR before charging credits.
    """
    from llm_prices.compat.tokencost import calculate_cost_by_tokens
    
    # Extract token counts from usage metadata
    input_tokens = token_usage.get("input_tokens", 0)
    output_tokens = token_usage.get("output_tokens", 0)
    cached_tokens = token_usage.get("cached_tokens", 0)
    # cache_creation_tokens = token_usage.get("cache_creation_input_tokens", 0)
    
    # OpenAI and Anthropic both use model name as-is for tokencost library
    # (no special prefixes needed for these providers)
    tokencost_model = model_name
    
    try:
        # Calculate input tokens cost (excluding cached and cache creation tokens)
        # Cached tokens (cache_read) are priced cheaper
        # Cache creation tokens (Anthropic only) are priced more expensive
        input_tokens_uncached = input_tokens - cached_tokens  #  - cache_creation_tokens
        
        # Use tokencost library for accurate, up-to-date pricing
        input_tokens_cost = float(
            calculate_cost_by_tokens(input_tokens_uncached, tokencost_model, "input")
        )
        output_tokens_cost = float(
            calculate_cost_by_tokens(output_tokens, tokencost_model, "output")
        )
        
        # Calculate cached tokens cost with fallback to model metadata
        cached_tokens_cost = 0.0
        if cached_tokens > 0:
            try:
                # Try tokencost first
                cached_tokens_cost = float(
                    calculate_cost_by_tokens(cached_tokens, tokencost_model, "cached")
                )
            except Exception:
                # Fallback to model metadata pricing if tokencost doesn't have cached pricing
                if model_metadata.cached_token_price_per_M > 0:
                    cached_tokens_cost = (
                        model_metadata.cached_token_price_per_M * cached_tokens / 1_000_000
                    )
        
        # Calculate cache creation tokens cost (Anthropic only)
        # These are tokens used to create a new cache (more expensive than regular input)
        cache_creation_cost = 0.0
        # if cache_creation_tokens > 0:
        #     # tokencost library may not have "cache_creation" pricing, use model metadata
        #     if hasattr(model_metadata, 'cache_creation_token_price_per_M') and model_metadata.cache_creation_token_price_per_M > 0:
        #         cache_creation_cost = (
        #             model_metadata.cache_creation_token_price_per_M * cache_creation_tokens / 1_000_000
        #         )
        #     else:
        #         # Fallback: treat as regular input tokens if no specific pricing
        #         cache_creation_cost = float(
        #             calculate_cost_by_tokens(cache_creation_tokens, tokencost_model, "input")
        #         )
        
        total_cost = input_tokens_cost + output_tokens_cost + cached_tokens_cost + cache_creation_cost
        return float(total_cost)
        
    except Exception as e:
        # Fallback to model metadata pricing if tokencost library fails
        # This ensures we can still calculate costs even if tokencost doesn't recognize the model
        input_tokens_uncached = input_tokens - cached_tokens  #  - cache_creation_tokens
        
        input_cost = model_metadata.input_token_price_per_M * input_tokens_uncached / 1_000_000
        output_cost = model_metadata.output_token_price_per_M * output_tokens / 1_000_000
        
        # Handle cached token pricing
        cached_cost = 0.0
        if cached_tokens > 0 and model_metadata.cached_token_price_per_M > 0:
            cached_cost = model_metadata.cached_token_price_per_M * cached_tokens / 1_000_000
        
        # Handle cache creation token pricing (Anthropic)
        cache_creation_cost = 0.0
        # if cache_creation_tokens > 0:
        #     if hasattr(model_metadata, 'cache_creation_token_price_per_M') and model_metadata.cache_creation_token_price_per_M > 0:
        #         cache_creation_cost = model_metadata.cache_creation_token_price_per_M * cache_creation_tokens / 1_000_000
        #     else:
        #         # Fallback: treat as regular input tokens
        #         cache_creation_cost = model_metadata.input_token_price_per_M * cache_creation_tokens / 1_000_000
        
        total_cost = input_cost + output_cost + cached_cost + cache_creation_cost
        return float(total_cost)

