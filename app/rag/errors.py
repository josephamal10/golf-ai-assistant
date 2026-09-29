class RateLimitedError(RuntimeError):
    """An upstream API (embeddings or LLM) refused the call for rate-limit or quota reasons."""
