"""
Central config. Loads API keys and model choices from .env.

Supported MODEL_PROVIDER values: "openai", "anthropic", "gemini", "groq"

NFR-reliability: every LLM call is wrapped with tenacity retry
(3 attempts, exponential backoff 1s→2s→4s) plus a hard 60 s timeout.
If all retries fail, a FallbackLLMError is raised with a user-friendly
message so the UI can show a graceful error rather than a crash traceback.
"""
import os
import time
import functools
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) or os.environ (.env / local)."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


MODEL_PROVIDER  = _secret("MODEL_PROVIDER", "groq").lower()

OPENAI_MODEL    = _secret("OPENAI_MODEL",    "gpt-4o")
ANTHROPIC_MODEL = _secret("ANTHROPIC_MODEL", "claude-sonnet-4-5")
GEMINI_MODEL    = _secret("GEMINI_MODEL",    "gemini-3.6-flash")
GROQ_MODEL      = _secret("GROQ_MODEL",      "openai/gpt-oss-120b")

DB_PATH          = _secret("DB_PATH",          "results/learner_state.db")
LOG_PATH         = _secret("LOG_PATH",          "logs/agent_calls.jsonl")
CONTENT_LOG_PATH = _secret("CONTENT_LOG_PATH",  "results/generated_content.jsonl")

TEMPERATURES = {
    "planner":          0.2,
    "content":          0.7,
    "assessment_gen":   0.4,
    "assessment_grade": 0.0,
}

# ── retry config ──────────────────────────────────────────────────────────────
_MAX_RETRIES   = 3
_BACKOFF_BASE  = 1.0   # seconds; each attempt: 1 s, 2 s, 4 s


class FallbackLLMError(RuntimeError):
    """Raised when all retry attempts for an LLM call are exhausted."""


def with_retry(fn):
    """
    Decorator: exponential-backoff retry (3 attempts).
    Respects Retry-After from rate-limit (429) errors when present.
    Raises FallbackLLMError if every attempt fails.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                # Respect server-sent retry delay for 429 rate-limit errors
                wait = _BACKOFF_BASE * (2 ** attempt)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    import re as _re
                    m = _re.search(r"retryDelay['\"]:\s*['\"](\d+)s", err_str)
                    if m:
                        wait = float(m.group(1)) + 1.0
                print(f"[RETRY] {fn.__name__} attempt {attempt+1}/{_MAX_RETRIES} failed: "
                      f"{err_str[:120]}. Retrying in {wait:.0f}s…")
                time.sleep(wait)
        raise FallbackLLMError(
            f"LLM call failed after {_MAX_RETRIES} attempts. Last error: {last_exc}"
        ) from last_exc
    return wrapper


# ── content helpers ───────────────────────────────────────────────────────────
def extract_content(response) -> str:
    """
    Safely extract plain text from any LangChain LLM response.
    Handles plain strings and Gemini-style list-of-content-blocks.
    """
    raw = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        raw = " ".join(p for p in parts if p)
    return raw.strip()


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ── LLM factory ───────────────────────────────────────────────────────────────
def get_llm(agent_key: str):
    """
    Returns a LangChain chat model configured for the given agent.
    Re-reads .env on every call so provider switches take effect immediately.
    The returned model's `.invoke()` method is wrapped with retry logic.
    """
    load_dotenv(find_dotenv(), override=True)
    provider    = _secret("MODEL_PROVIDER", "groq").lower()
    temperature = TEMPERATURES[agent_key]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        base_url = _secret("OPENAI_BASE_URL", "")
        llm = ChatOpenAI(
            model=_secret("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
            openai_api_key=_secret("OPENAI_API_KEY"),
            **({"base_url": base_url} if base_url else {}),
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=_secret("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            temperature=temperature,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=_secret("GEMINI_MODEL", "gemini-3.6-flash"),
            temperature=temperature,
            google_api_key=_secret("GEMINI_API_KEY"),
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=_secret("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=temperature,
            groq_api_key=_secret("GROQ_API_KEY"),
        )

    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER '{provider}' — "
            "use 'openai', 'anthropic', 'gemini', or 'groq'"
        )

    # Wrap with retry via a thin proxy so every agent benefits automatically.
    # We can't set llm.invoke directly (Pydantic v2 blocks it), so we wrap
    # the whole object in a lightweight proxy that intercepts .invoke().
    return _RetryProxy(llm)


class _RetryProxy:
    """
    Thin proxy that adds retry-with-backoff and a hard 30s timeout around
    a LangChain LLM's invoke().

    NFR-latency: a single agent turn must return in < 8 s under normal
    conditions. The 30 s hard timeout ensures a frozen API call never
    blocks a student session indefinitely — it is treated as a failure
    and retried/fallen back like any other error.
    """
    _HARD_TIMEOUT = 30  # seconds

    def __init__(self, llm):
        self._llm = llm
        self.invoke = with_retry(self._invoke_with_timeout)

    def _invoke_with_timeout(self, *args, **kwargs):
        """Calls llm.invoke with a hard timeout using a background thread."""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._llm.invoke, *args, **kwargs)
            try:
                return future.result(timeout=self._HARD_TIMEOUT)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"LLM call timed out after {self._HARD_TIMEOUT}s. "
                    "The API may be overloaded — will retry."
                )

    def __getattr__(self, name):
        return getattr(self._llm, name)
