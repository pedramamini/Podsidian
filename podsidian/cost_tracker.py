"""
Cost tracking module for Podsidian.

This module provides functionality to track and calculate costs for AI API calls.
"""

import logging
import threading
import requests
import json
import os
from typing import Dict, Optional, Any, List
from decimal import Decimal
from functools import lru_cache

logger = logging.getLogger(__name__)

# Thread-local storage for cost tracking
_local = threading.local()

# Model pricing cache
_model_pricing = {
    'whisper/large-v3': {
        'prompt': Decimal('0.0'),  # Free since it's local
        'completion': Decimal('0.0')  # Free since it's local
    }
}

@lru_cache(maxsize=32)
def get_model_pricing(model_id: str) -> Dict[str, Decimal]:
    """
    Get the pricing information for a model from OpenRouter.
    
    Args:
        model_id: The model ID to get pricing for
        
    Returns:
        A dictionary with 'prompt' and 'completion' pricing per token
    """
    # Default pricing if we can't get it from OpenRouter
    default_pricing = {
        'prompt': Decimal('0.000005'),  # $0.000005 per token
        'completion': Decimal('0.000015')  # $0.000015 per token
    }
    
    # Check if we already have the pricing in our cache
    if model_id in _model_pricing:
        return _model_pricing[model_id]
    
    # Try to get the pricing from OpenRouter
    try:
        response = requests.get('https://openrouter.ai/api/v1/models')
        if response.status_code == 200:
            models_data = response.json().get('data', [])
            for model_data in models_data:
                if model_data['id'] == model_id:
                    pricing = model_data.get('pricing', {})
                    _model_pricing[model_id] = {
                        'prompt': Decimal(str(pricing.get('prompt', default_pricing['prompt']))),
                        'completion': Decimal(str(pricing.get('completion', default_pricing['completion'])))
                    }
                    return _model_pricing[model_id]
    except Exception as e:
        logger.warning(f"Failed to get model pricing from OpenRouter: {str(e)}")
    
    # If we couldn't get the pricing, use the default
    _model_pricing[model_id] = default_pricing
    return default_pricing

def init_cost_tracker():
    """Initialize the cost tracker for the current thread."""
    _local.costs = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cost": Decimal("0.0"),
        "completion_cost": Decimal("0.0"),
        "total_cost": Decimal("0.0"),
        "api_calls": 0,
        "models_used": {},
        "audio_seconds": 0,
        "audio_cost": Decimal("0.0")
    }

def get_costs() -> Dict[str, Any]:
    """Get the current costs."""
    if not hasattr(_local, "costs"):
        init_cost_tracker()
    return _local.costs

def track_api_call(response_data: Dict[str, Any], model: str, audio_seconds: Optional[float] = None):
    """
    Track the cost of an API call.
    
    Args:
        response_data: The response data from the API call
        model: The model used for the API call
    """
    if not hasattr(_local, "costs"):
        init_cost_tracker()
    
    # Extract usage information from the response
    usage = response_data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    
    # If usage information is not available, estimate based on response content
    if not usage:
        # Rough estimation if usage is not provided
        choices = response_data.get("choices", [])
        if choices and len(choices) > 0:
            completion_content = choices[0].get("message", {}).get("content", "")
            # Rough estimate: 1 token ≈ 4 characters
            completion_tokens = len(completion_content) // 4
            # Assume prompt tokens based on completion (typically prompt is 1-2x completion)
            prompt_tokens = completion_tokens * 2
            total_tokens = prompt_tokens + completion_tokens
    
    # Get pricing for this model
    pricing = get_model_pricing(model)
    
    # Calculate costs
    prompt_cost = Decimal(prompt_tokens) * pricing['prompt']
    completion_cost = Decimal(completion_tokens) * pricing['completion']
    total_cost = prompt_cost + completion_cost
    
    # Update the cost tracker
    _local.costs["prompt_tokens"] += prompt_tokens
    _local.costs["completion_tokens"] += completion_tokens
    _local.costs["total_tokens"] += total_tokens
    _local.costs["prompt_cost"] += prompt_cost
    _local.costs["completion_cost"] += completion_cost
    _local.costs["total_cost"] += total_cost
    _local.costs["api_calls"] += 1
    
    # Track usage by model
    if model not in _local.costs["models_used"]:
        _local.costs["models_used"][model] = {
            "calls": 0,
            "tokens": 0,
            "cost": Decimal("0.0")
        }
    _local.costs["models_used"][model]["calls"] += 1
    _local.costs["models_used"][model]["tokens"] += total_tokens
    _local.costs["models_used"][model]["cost"] += total_cost
    
    # Track audio duration and cost for Whisper
    if audio_seconds is not None and model.startswith('whisper/'):
        _local.costs["audio_seconds"] += audio_seconds
        # Whisper is free since we run it locally
        _local.costs["audio_cost"] += Decimal("0.0")
        logger.debug(f"Whisper transcription: {audio_seconds:.1f} seconds of audio processed")
    else:
        logger.debug(f"API call to {model}: {prompt_tokens} prompt tokens, {completion_tokens} completion tokens, ${total_cost:.6f}")

def format_cost_summary() -> str:
    """
    Format the cost summary as a string.
    
    Returns:
        A formatted string with the cost summary
    """
    if not hasattr(_local, "costs"):
        return "No API calls tracked"
    
    costs = _local.costs
    summary = [
        "AI Cost Summary:",
        f"  API Calls: {costs['api_calls']}",
        f"  Prompt Tokens: {costs['prompt_tokens']}",
        f"  Completion Tokens: {costs['completion_tokens']}",
        f"  Total Tokens: {costs['total_tokens']}",
        f"  Audio Processed: {costs['audio_seconds']:.1f} seconds",
        f"  Total Cost: ${costs['total_cost']:.6f}"
    ]
    
    # Add model-specific information if available
    if costs['models_used']:
        summary.append("\nCost by Model:")
        for model, model_costs in costs['models_used'].items():
            summary.append(f"  {model}:")
            summary.append(f"    Calls: {model_costs['calls']}")
            summary.append(f"    Tokens: {model_costs['tokens']}")
            summary.append(f"    Cost: ${model_costs['cost']:.6f}")
    
    return "\n".join(summary)
