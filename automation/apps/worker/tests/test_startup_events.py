import asyncio
from datetime import timedelta

import mistralai.client.models
from mistralai.workflows.testing import create_test_worker

import worker.workflows.startup_events as se

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=30)

FAKE_MARKDOWN = "# Startup-Events\nStand: 20.11.2026\n\n## 💻 Software\n| Datum | Event |\n|---|---|\n| 25.11.2026 | AI Tinkerers |\n"


def _text_chunk(text: str) -> mistralai.client.models.TextChunk:
    return mistralai.client.models.TextChunk(type="text", text=text)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeGitHubClient:
    def __init__(self, *args, **kwargs) -> None:
        self.requests: list[tuple] = []

    async def __aenter__(self) -> "_FakeGitHubClient":
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    async def get(self, url, params=None, headers=None) -> _FakeResponse:
        self.requests.append(("GET", url, params, headers))
        return _FakeResponse(200, {"sha": "existing-sha"})

    async def put(self, url, json=None, headers=None) -> _FakeResponse:
        self.requests.append(("PUT", url, json, headers))
        assert json["sha"] == "existing-sha"
        assert json["branch"] == "main"
        return _FakeResponse(201, {"commit": {"sha": "new-commit-sha"}})


async def _fake_runner_run(agent, inputs, session, max_turns=None):
    return [_text_chunk(FAKE_MARKDOWN)]


async def test_commit_activity_dry_run_skips_http(monkeypatch) -> None:
    class _Boom:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("HTTP darf im dry_run nicht aufgerufen werden")

    monkeypatch.setattr(se.httpx, "AsyncClient", _Boom)
    result = await se.commit_events_file("o/r", "main", "startup-events.md", FAKE_MARKDOWN, True)
    assert result["dry_run"] is True
    assert result["commit_sha"] is None
    assert result["content_bytes"] == len(FAKE_MARKDOWN.encode("utf-8"))


async def test_commit_activity_puts_file(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    fake = _FakeGitHubClient()
    monkeypatch.setattr(se.httpx, "AsyncClient", lambda *a, **k: fake)
    result = await se.commit_events_file("o/r", "main", "startup-events.md", FAKE_MARKDOWN, False)
    assert result["commit_sha"] == "new-commit-sha"
    put = [r for r in fake.requests if r[0] == "PUT"][0]
    assert "Bearer fake-token" in put[3]["Authorization"]
    import base64

    assert base64.b64decode(put[2]["content"]).decode("utf-8") == FAKE_MARKDOWN


async def test_workflow_end_to_end_dry_run(temporal_env, monkeypatch) -> None:
    monkeypatch.setattr(se.workflows_mistralai.Runner, "run", _fake_runner_run)

    async with create_test_worker(
        temporal_env,
        workflows=[se.StartupEventsWorkflow],
        activities=[se.research_events_markdown, se.commit_events_file],
    ):
        handle = await temporal_env.client.start_workflow(
            "startup-events-bonn",
            {"dry_run": True},
            id="test-startup-events-dry",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=30)
        assert result["dry_run"] is True
        assert result["commit_sha"] is None
        assert FAKE_MARKDOWN[:100] in result["markdown_preview"]


async def test_workflow_end_to_end_commit(temporal_env, monkeypatch) -> None:
    monkeypatch.setattr(se.workflows_mistralai.Runner, "run", _fake_runner_run)
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    fake = _FakeGitHubClient()
    monkeypatch.setattr(se.httpx, "AsyncClient", lambda *a, **k: fake)

    async with create_test_worker(
        temporal_env,
        workflows=[se.StartupEventsWorkflow],
        activities=[se.research_events_markdown, se.commit_events_file],
    ):
        handle = await temporal_env.client.start_workflow(
            "startup-events-bonn",
            {"dry_run": False},
            id="test-startup-events-commit",
            task_queue="test-task-queue",
            execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
        )
        result = await asyncio.wait_for(handle.result(), timeout=30)
        assert result["dry_run"] is False
        assert result["commit_sha"] == "new-commit-sha"
