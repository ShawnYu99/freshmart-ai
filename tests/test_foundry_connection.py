"""Offline tests only: importing or collecting the live script never calls Azure."""

import json

import httpx2
import pytest
from openai import OpenAI

from scripts import test_foundry


@pytest.fixture
def configured(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FOUNDRY_BASE_URL=https://unit-test.services.ai.azure.com/openai/v1\n"
        "FOUNDRY_MODEL=DeepSeek-V4-Flash\n"
        "FOUNDRY_API_KEY=unit-test-key\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(test_foundry, "ENV_FILE", env_file)
    for name in ("FOUNDRY_BASE_URL", "FOUNDRY_MODEL", "FOUNDRY_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def stub_client(monkeypatch, handler):
    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == 60.0
        return OpenAI(
            **kwargs,
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        )

    monkeypatch.setattr(test_foundry, "OpenAI", factory)


def response_body(content="Offline fixture answer.", *, reason="stop", usage=True):
    body = {
        "id": "offline-fixture",
        "object": "chat.completion",
        "created": 0,
        "model": "DeepSeek-V4-Flash",
        "choices": [{
            "index": 0,
            "finish_reason": reason,
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning_content": "PRIVATE REASONING FIXTURE",
            },
        }],
    }
    if usage:
        body["usage"] = {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        }
    return body


def test_reads_env_and_normalizes_base_url(configured, monkeypatch):
    assert test_foundry.configuration() == (
        "https://unit-test.services.ai.azure.com/openai/v1/",
        "DeepSeek-V4-Flash", "unit-test-key",
    )
    monkeypatch.setenv("FOUNDRY_API_KEY", "exported-key")
    assert test_foundry.configuration()[2] == "exported-key"


@pytest.mark.parametrize("url", [
    "http://unit-test.services.ai.azure.com/openai/v1",
    "https://untrusted.example/openai/v1",
    "https://unit-test.services.ai.azure.com/api/projects/test",
    "https://unit-test.services.ai.azure.com/models",
    "https://unit-test.services.ai.azure.com/openai/v1/chat/completions",
    "https://unit-test.services.ai.azure.com/openai/v1?api-key=do-not-print",
    "https://unit-test.services.ai.azure.com/openai/v1#fragment",
    "https://user:do-not-print@unit-test.services.ai.azure.com/openai/v1",
    "https://unit-test.services.ai.azure.com:invalid/openai/v1",
])
def test_invalid_endpoint_never_sends_request(configured, monkeypatch, capsys, url):
    monkeypatch.setenv("FOUNDRY_BASE_URL", url)
    monkeypatch.setattr(
        test_foundry, "OpenAI",
        lambda **kwargs: pytest.fail("Invalid configuration must not send a request"),
    )
    assert test_foundry.main() == 1
    output = capsys.readouterr()
    assert "unit-test-key" not in output.out + output.err
    assert "do-not-print" not in output.out + output.err


@pytest.mark.parametrize("name", ["FOUNDRY_BASE_URL", "FOUNDRY_MODEL", "FOUNDRY_API_KEY"])
def test_missing_configuration(configured, monkeypatch, name, capsys):
    monkeypatch.setenv(name, "")
    assert test_foundry.main() == 1
    assert "Check FOUNDRY_BASE_URL" in capsys.readouterr().err


def test_request_path_deployment_and_final_answer_only(configured, monkeypatch, capsys):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/openai/v1/chat/completions"
        assert not request.url.query
        assert request.headers["authorization"] == "Bearer unit-test-key"
        body = json.loads(request.content)
        assert body["model"] == "DeepSeek-V4-Flash"
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        assert "unit-test-key" not in request.content.decode()
        return httpx2.Response(200, json=response_body())

    stub_client(monkeypatch, handler)
    assert test_foundry.main() == 0
    output = capsys.readouterr()
    assert len(requests) == 1
    assert "Offline fixture answer." in output.out
    assert "prompt=10, completion=5, total=15" in output.out
    assert "PRIVATE REASONING FIXTURE" not in output.out + output.err
    assert "unit-test-key" not in output.out + output.err


@pytest.mark.parametrize(("status", "expected"), [
    (400, "Request rejected"), (401, "Authentication failed"),
    (403, "Access denied"), (404, "Endpoint or deployment not found"),
    (429, "Rate limited"), (500, "HTTP 500"),
])
def test_service_errors_are_safe_and_not_retried(
    configured, monkeypatch, capsys, status, expected
):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(
            status, json={"error": {"message": "PRIVATE SERVER DATA unit-test-key"}}
        )

    stub_client(monkeypatch, handler)
    assert test_foundry.main() == 1
    output = capsys.readouterr()
    assert len(requests) == 1
    assert expected in output.err
    assert "PRIVATE SERVER DATA" not in output.out + output.err
    assert "unit-test-key" not in output.out + output.err


@pytest.mark.parametrize("exception", [httpx2.ConnectError, httpx2.ReadTimeout])
def test_network_errors(configured, monkeypatch, capsys, exception):
    def handler(request):
        raise exception("PRIVATE NETWORK DETAILS", request=request)

    stub_client(monkeypatch, handler)
    assert test_foundry.main() == 1
    assert "PRIVATE NETWORK DETAILS" not in capsys.readouterr().err


@pytest.mark.parametrize(("content", "reason"), [
    (None, "stop"), ("", "stop"), (" ", "stop"),
    ("<think>hidden thoughts</think>Final text.", "stop"),
    ("<analysis>hidden thoughts</analysis>", "stop"),
    ("Accidental secret: unit-test-key", "stop"),
    ("Incomplete answer.", "length"),
    (None, "content_filter"),
])
def test_unsafe_or_incomplete_content_is_not_printed(
    configured, monkeypatch, capsys, content, reason
):
    stub_client(
        monkeypatch,
        lambda request: httpx2.Response(200, json=response_body(content, reason=reason)),
    )
    assert test_foundry.main() == 1
    output = capsys.readouterr()
    assert "Response:" not in output.out
    assert "hidden thoughts" not in output.out + output.err
    assert "unit-test-key" not in output.out + output.err


def test_usage_is_optional(configured, monkeypatch, capsys):
    stub_client(
        monkeypatch,
        lambda request: httpx2.Response(200, json=response_body(usage=False)),
    )
    assert test_foundry.main() == 0
    assert "Token usage" not in capsys.readouterr().out


def test_empty_choices(configured, monkeypatch, capsys):
    body = response_body()
    body["choices"] = []
    stub_client(monkeypatch, lambda request: httpx2.Response(200, json=body))
    assert test_foundry.main() == 1
    assert "no completion choices" in capsys.readouterr().err
