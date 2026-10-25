"""Shared configuration for the existing Foundry OpenAI v1 deployment."""

import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
REASONING_TAG = re.compile(
    r"<\s*/?\s*(?:think|thinking|analysis|reasoning)\b", re.IGNORECASE
)


def configuration(env_file: Path | None = None) -> tuple[str, str, str]:
    """Load local settings while preserving exported environment overrides."""
    load_dotenv(
        env_file if env_file is not None else ROOT / ".env",
        override=False,
        interpolate=False,
    )
    names = ("FOUNDRY_BASE_URL", "FOUNDRY_MODEL", "FOUNDRY_API_KEY")
    values = [os.getenv(name, "").strip() for name in names]
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise ValueError("Missing configuration: " + ", ".join(missing))
    base_url, model, api_key = values
    url = urlsplit(base_url)
    if (
        url.scheme != "https"
        or not (url.hostname or "").endswith(
            (".services.ai.azure.com", ".openai.azure.com")
        )
        or url.path.rstrip("/") != "/openai/v1"
        or url.query
        or url.fragment
        or url.username
        or url.password
        or url.port not in (None, 443)
    ):
        raise ValueError(
            "FOUNDRY_BASE_URL must be an HTTPS Foundry resource URL ending in "
            "/openai/v1/, without a query, fragment, or embedded credentials. "
            "Use the deployed model's OpenAI-compatible endpoint, not a project URL."
        )
    return base_url.rstrip("/") + "/", model, api_key
