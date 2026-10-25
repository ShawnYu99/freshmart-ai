"""Explicit live connection test for the configured Foundry deployment."""

import logging
from pathlib import Path
import sys

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.foundry import REASONING_TAG, configuration as foundry_configuration

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def configuration() -> tuple[str, str, str]:
    """Read local settings without overwriting exported environment variables."""
    return foundry_configuration(ENV_FILE)


def error(message: str) -> int:
    print("Foundry connection test failed: " + message, file=sys.stderr)
    return 1


def main() -> int:
    # SDK debug logs and raw errors can contain request/response data.
    for name in ("openai", "httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        base_url, model, api_key = configuration()
    except ValueError:
        return error(
            "Check FOUNDRY_BASE_URL, FOUNDRY_MODEL, and FOUNDRY_API_KEY in .env "
            "or the exported environment. The URL must use HTTPS and the "
            "/openai/v1/ path on a Foundry resource, without URL credentials "
            "or query parameters."
        )
    except OSError:
        return error("Unable to read the local .env configuration file.")

    print("Sending one request to the configured Foundry deployment...", flush=True)
    try:
        with OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=60.0,
            max_retries=0,
        ) as client:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Give only a brief final answer. Do not include "
                            "reasoning, analysis, or internal thinking."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Say hello in one short sentence for the fictional "
                            "FreshMart AI retail support demo."
                        ),
                    },
                ],
            )
    except AuthenticationError:
        return error(
            "Authentication failed (HTTP 401). Check or rotate the resource API "
            "key and verify that it belongs to the configured Foundry resource."
        )
    except PermissionDeniedError:
        return error(
            "Access denied (HTTP 403). Check whether key authentication is "
            "enabled and whether network or resource access policies permit access."
        )
    except NotFoundError:
        return error(
            "Endpoint or deployment not found (HTTP 404). Compare "
            "FOUNDRY_BASE_URL and FOUNDRY_MODEL with the deployment's code sample; "
            "the model value must be the exact deployment name."
        )
    except RateLimitError:
        return error(
            "Rate limited or quota exhausted (HTTP 429). Check the deployment's "
            "quota and usage, then wait before retrying. No automatic retry was made."
        )
    except BadRequestError:
        return error(
            "Request rejected (HTTP 400). Verify that the configured deployment "
            "supports OpenAI v1 Chat Completions."
        )
    except APITimeoutError:
        return error("The request timed out. Check deployment availability and network access.")
    except APIConnectionError:
        return error(
            "Unable to reach Foundry. Check DNS, TLS certificates, firewall, "
            "and private-endpoint access. TLS verification remains enabled."
        )
    except APIStatusError as exc:
        return error(
            f"The service returned HTTP {exc.status_code}. Check resource health "
            "and deployment availability. Raw service details were not printed."
        )

    if not completion.choices:
        return error("The service returned no completion choices.")
    choice = completion.choices[0]
    content = choice.message.content
    if choice.finish_reason != "stop" or not content or not content.strip():
        return error(
            "The service returned no complete final answer. The response may "
            "have been filtered or truncated; hidden reasoning was not printed."
        )
    if REASONING_TAG.search(content) or api_key in content:
        return error("The returned content was withheld to avoid exposing sensitive output.")

    print("Response:")
    print(content.strip())
    if completion.usage is not None:
        usage = completion.usage
        print(
            f"Token usage: prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}, total={usage.total_tokens}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
