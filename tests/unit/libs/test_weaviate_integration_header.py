"""Unit tests for the X-Weaviate-Client-Integration header sent by WeaviateBaseClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from global_config.settings import global_settings
from weaviate_client import base_client
from weaviate_client.base_client import INTEGRATION_HEADER, WeaviateBaseClient, with_integration_header


def test_adds_kiwiq_name_and_app_version():
    headers = with_integration_header({"X-Openai-Api-Key": "key"})

    assert headers == {
        "X-Openai-Api-Key": "key",
        INTEGRATION_HEADER: f"kiwiq/{global_settings.VERSION}",
    }


def test_handles_missing_headers():
    assert with_integration_header(None) == {INTEGRATION_HEADER: f"kiwiq/{global_settings.VERSION}"}


def test_caller_header_wins_regardless_of_case():
    caller = {"x-weaviate-client-integration": "custom/1.0"}

    assert with_integration_header(caller) == caller


def test_does_not_mutate_caller_headers():
    caller = {"X-Openai-Api-Key": "key"}

    with_integration_header(caller)

    assert caller == {"X-Openai-Api-Key": "key"}


def test_reports_unknown_when_version_is_empty(monkeypatch):
    monkeypatch.setattr(global_settings, "VERSION", "")

    assert with_integration_header({})[INTEGRATION_HEADER] == "kiwiq/unknown"


@pytest.mark.asyncio
async def test_connect_sends_header_on_async_and_sync_clients(monkeypatch):
    async_client = MagicMock(connect=AsyncMock())
    use_async = MagicMock(return_value=async_client)
    connect_sync = MagicMock()
    monkeypatch.setattr(base_client.weaviate, "use_async_with_local", use_async)
    monkeypatch.setattr(base_client.weaviate, "connect_to_local", connect_sync)

    client = WeaviateBaseClient(host="localhost", additional_headers={"X-Openai-Api-Key": "key"})
    client.api_key = None  # force the local-connection path regardless of the environment
    await client.connect()

    expected = {"X-Openai-Api-Key": "key", INTEGRATION_HEADER: f"kiwiq/{global_settings.VERSION}"}
    assert use_async.call_args.kwargs["headers"] == expected
    assert connect_sync.call_args.kwargs["headers"] == expected
    assert client.additional_headers == {"X-Openai-Api-Key": "key"}
