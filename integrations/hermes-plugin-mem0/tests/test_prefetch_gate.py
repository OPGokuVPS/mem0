"""Regressions for the recall-prefetch context gate.

A cron job or subagent run is autonomous: it has no user question for recall to
answer, yet the prefetched block lands in the transcript and is re-sent on every
later call of that session (measured upstream: 32 prefetch dumps / ~187k tokens
across cron sessions). These tests pin the gate that skips the recall round-trip
in those contexts, and pin every way the gate could be turned off by accident.
"""

from unittest.mock import Mock


def _provider(plugin, monkeypatch, config=None, backend=None):
    monkeypatch.setattr(plugin, "_load_config", lambda: dict(config or {"api_key": "test-key"}))
    provider = plugin.Mem0MemoryProvider()
    backend = backend if backend is not None else Mock()
    backend.search.return_value = [{"memory": "Prefers Python"}]
    monkeypatch.setattr(provider, "_create_backend", lambda: backend)
    return provider, backend


def test_cron_context_skips_prefetch(plugin, monkeypatch):
    """The regression: a cron run must not spend a recall round-trip per turn."""
    provider, backend = _provider(plugin, monkeypatch)
    provider.initialize("cron-session", agent_context="cron")
    try:
        provider.on_turn_start(1, "run the nightly ingest")
        assert provider.prefetch("run the nightly ingest") == ""
        backend.search.assert_not_called()
    finally:
        provider.shutdown()


def test_subagent_context_skips_prefetch(plugin, monkeypatch):
    provider, backend = _provider(plugin, monkeypatch)
    provider.initialize("sub-session", agent_context="subagent")
    try:
        provider.on_turn_start(1, "summarize this file")
        assert provider.prefetch("summarize this file") == ""
        backend.search.assert_not_called()
    finally:
        provider.shutdown()


def test_primary_context_still_prefetches(plugin, monkeypatch):
    """Interactive sessions are unchanged: the gate must not cost recall there."""
    provider, backend = _provider(plugin, monkeypatch)
    provider.initialize("primary-session", agent_context="primary")
    try:
        provider.on_turn_start(1, "what do I prefer?")
        assert "Prefers Python" in provider.prefetch("what do I prefer?")
        assert backend.search.call_count == 1
    finally:
        provider.shutdown()


def test_absent_context_fails_open_to_prefetch(plugin, monkeypatch):
    """A caller that passes no agent_context keeps the old behaviour."""
    provider, backend = _provider(plugin, monkeypatch)
    provider.initialize("legacy-session")
    try:
        assert "Prefers Python" in provider.prefetch("what do I prefer?")
        assert backend.search.call_count == 1
    finally:
        provider.shutdown()


def test_context_name_is_normalized(plugin, monkeypatch):
    """A future caller handing us "Cron" must not quietly re-enable the round-trip."""
    provider, backend = _provider(plugin, monkeypatch)
    provider.initialize("cron-session", agent_context=" Cron ")
    try:
        assert provider.prefetch("anything") == ""
        backend.search.assert_not_called()
    finally:
        provider.shutdown()


def test_config_none_re_enables_prefetch_everywhere(plugin, monkeypatch):
    """The documented opt-out: prefetch_skip_contexts = none."""
    provider, backend = _provider(plugin, monkeypatch, config={"api_key": "test-key", "prefetch_skip_contexts": "none"})
    provider.initialize("cron-session", agent_context="cron")
    try:
        assert "Prefers Python" in provider.prefetch("what do I prefer?")
        assert backend.search.call_count == 1
    finally:
        provider.shutdown()


def test_misspelled_config_keeps_the_gate_on(plugin, monkeypatch):
    """A typo must never silently switch the gate OFF."""
    provider, backend = _provider(plugin, monkeypatch, config={"api_key": "test-key", "prefetch_skip_contexts": "crom"})
    provider.initialize("cron-session", agent_context="cron")
    try:
        assert provider.prefetch("anything") == ""
        backend.search.assert_not_called()
    finally:
        provider.shutdown()


def test_blank_config_keeps_the_builtin_default(plugin, monkeypatch):
    provider, backend = _provider(plugin, monkeypatch, config={"api_key": "test-key", "prefetch_skip_contexts": "  "})
    provider.initialize("cron-session", agent_context="cron")
    try:
        assert provider.prefetch("anything") == ""
        backend.search.assert_not_called()
    finally:
        provider.shutdown()
