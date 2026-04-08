"""Tests for version module — single source of truth for all version strings."""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Version format tests
# ---------------------------------------------------------------------------


class TestVersionFormat:
    """Test ENGINE_VERSION matches CalVer+SemVer pattern."""

    def test_engine_version_matches_pattern(self):
        from quantagent.version import ENGINE_VERSION

        pattern = r"^\d{4}\.\d{2}\.\d+\.\d+\.\d+(-\w+\.\d+)?(\+.+)?$"
        assert re.match(pattern, ENGINE_VERSION), (
            f"ENGINE_VERSION '{ENGINE_VERSION}' doesn't match "
            f"YYYY.MM.MAJOR.MINOR.PATCH[-prerelease][+build]"
        )

    def test_api_version_format(self):
        from quantagent.version import API_VERSION

        assert API_VERSION.startswith("v")
        assert API_VERSION == "v1"

    def test_engine_version_is_string(self):
        from quantagent.version import ENGINE_VERSION

        assert isinstance(ENGINE_VERSION, str)

    def test_engine_version_starts_with_year(self):
        from quantagent.version import ENGINE_VERSION

        year = int(ENGINE_VERSION.split(".")[0])
        assert 2024 <= year <= 2030


# ---------------------------------------------------------------------------
# PROMPT_VERSIONS completeness tests
# ---------------------------------------------------------------------------


class TestPromptVersions:
    """Test that PROMPT_VERSIONS covers all agents."""

    def test_prompt_versions_has_all_llm_agents(self):
        from quantagent.version import PROMPT_VERSIONS

        required_agents = [
            "indicator_agent",
            "pattern_agent",
            "trend_agent",
            "conviction_agent",
            "decision_agent",
            "reflection_agent",
        ]
        for agent in required_agents:
            assert agent in PROMPT_VERSIONS, f"Missing PROMPT_VERSIONS entry for {agent}"

    def test_prompt_versions_values_are_strings(self):
        from quantagent.version import PROMPT_VERSIONS

        for agent, version in PROMPT_VERSIONS.items():
            assert isinstance(version, str), f"{agent} version should be str, got {type(version)}"

    def test_prompt_versions_not_empty(self):
        from quantagent.version import PROMPT_VERSIONS

        assert len(PROMPT_VERSIONS) >= 6

    def test_ml_model_versions_exist(self):
        from quantagent.version import ML_MODEL_VERSIONS

        assert "direction_model" in ML_MODEL_VERSIONS
        assert "regime_model" in ML_MODEL_VERSIONS
        assert "anomaly_detector" in ML_MODEL_VERSIONS


# ---------------------------------------------------------------------------
# __version__ export tests
# ---------------------------------------------------------------------------


class TestVersionExport:
    """Test that quantagent.__version__ equals ENGINE_VERSION."""

    def test_package_version_matches_engine(self):
        import quantagent
        from quantagent.version import ENGINE_VERSION

        assert quantagent.__version__ == ENGINE_VERSION

    def test_root_version_re_exports(self):
        """The root version.py re-exports from quantagent.version."""
        from version import ENGINE_VERSION as root_ver
        from quantagent.version import ENGINE_VERSION as pkg_ver

        assert root_ver == pkg_ver

    def test_root_version_all_exports(self):
        from version import API_VERSION, ENGINE_VERSION, PROMPT_VERSIONS, ML_MODEL_VERSIONS

        assert ENGINE_VERSION
        assert API_VERSION
        assert PROMPT_VERSIONS
        assert ML_MODEL_VERSIONS is not None


# ---------------------------------------------------------------------------
# Health endpoint version test
# ---------------------------------------------------------------------------


class TestHealthEndpointVersion:
    """Test that /health returns engine_version."""

    @pytest.fixture
    def client(self):
        os.environ.setdefault("API_KEYS", "test-key-123")

        from api.app import create_app
        from api.dependencies import get_bot_repo, get_health_tracker
        from tracking.health import HealthTracker

        app = create_app()

        @asynccontextmanager
        async def noop_lifespan(app):
            yield

        app.router.lifespan_context = noop_lifespan

        # Mock dependencies
        mock_bot_repo = MagicMock()
        mock_bot_repo.get_bot = MagicMock(return_value=None)
        # Make it awaitable
        import asyncio
        async def async_get_bot(bot_id):
            return None
        mock_bot_repo.get_bot = async_get_bot

        app.dependency_overrides[get_bot_repo] = lambda: mock_bot_repo
        app.dependency_overrides[get_health_tracker] = lambda: HealthTracker()

        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_health_includes_engine_version(self, client):
        from quantagent.version import ENGINE_VERSION

        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "engine_version" in data
        assert data["engine_version"] == ENGINE_VERSION

    def test_health_version_format(self, client):
        resp = client.get("/v1/health")
        version = resp.json()["engine_version"]
        pattern = r"^\d{4}\.\d{2}\.\d+\.\d+\.\d+(-\w+\.\d+)?$"
        assert re.match(pattern, version)


# ---------------------------------------------------------------------------
# API app version test
# ---------------------------------------------------------------------------


class TestAPIAppVersion:
    """Test that FastAPI app uses ENGINE_VERSION."""

    def test_app_version_contains_engine_version(self):
        from api.app import create_app
        from quantagent.version import ENGINE_VERSION

        app = create_app()
        assert ENGINE_VERSION in app.version

    def test_app_version_contains_api_version(self):
        from api.app import create_app
        from quantagent.version import API_VERSION

        app = create_app()
        assert API_VERSION in app.version


# ---------------------------------------------------------------------------
# LLM trace metadata tests
# ---------------------------------------------------------------------------


class TestTraceMetadata:
    """Test that ClaudeProvider includes version in trace metadata."""

    def test_get_engine_version_helper(self):
        from llm.claude import ClaudeProvider
        from quantagent.version import ENGINE_VERSION
        # Compare against the imported constant so the test doesn't
        # have to be touched on every CalVer/SemVer bump.
        assert ClaudeProvider._get_engine_version() == ENGINE_VERSION

    def test_get_prompt_version_known_agent(self):
        from llm.claude import ClaudeProvider
        from quantagent.version import PROMPT_VERSIONS
        version = ClaudeProvider._get_prompt_version("indicator_agent")
        assert version == PROMPT_VERSIONS["indicator_agent"]

    def test_get_prompt_version_unknown_agent(self):
        from llm.claude import ClaudeProvider
        version = ClaudeProvider._get_prompt_version("nonexistent_agent")
        assert version == "unknown"
