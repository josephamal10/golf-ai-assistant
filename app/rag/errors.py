class RateLimitedError(RuntimeError):
    """An upstream API (embeddings or LLM) refused the call for rate-limit or quota reasons."""


class UpstreamUnavailableError(RuntimeError):
    """An upstream LLM kept failing with server errors (e.g. 503 high demand) after retries."""
