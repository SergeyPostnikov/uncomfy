"""
Unit tests for StubServer — no GPU, no ComfyUI imports needed.
"""

import asyncio
import pytest
from unittest.mock import MagicMock

# Import only the stub, not the full engine (avoids comfy bootstrap)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uncomfy.engine import StubServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server() -> StubServer:
    return StubServer()


@pytest.fixture
def progress_queue() -> asyncio.Queue:
    return asyncio.Queue(maxsize=100)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_has_client_id(self, server):
        assert hasattr(server, "client_id")
        assert server.client_id is None

    def test_has_last_node_id(self, server):
        assert hasattr(server, "last_node_id")
        assert server.last_node_id is None

    def test_has_sockets_metadata(self, server):
        assert hasattr(server, "sockets_metadata")
        assert isinstance(server.sockets_metadata, dict)

    def test_send_sync_callable(self, server):
        assert callable(server.send_sync)

    def test_queue_updated_callable(self, server):
        assert callable(server.queue_updated)


# ---------------------------------------------------------------------------
# Event collection
# ---------------------------------------------------------------------------

class TestEventCollection:
    def test_send_sync_stores_event(self, server):
        server.send_sync("execution_start", {"prompt_id": "abc"})
        assert len(server.events) == 1
        assert server.events[0]["event"] == "execution_start"
        assert server.events[0]["data"]["prompt_id"] == "abc"

    def test_send_sync_multiple_events(self, server):
        server.send_sync("execution_start", {})
        server.send_sync("progress", {"value": 1, "max": 10})
        server.send_sync("execution_success", {})
        assert len(server.events) == 3

    def test_clear_resets_events(self, server):
        server.send_sync("execution_start", {})
        server.clear()
        assert server.events == []

    def test_send_sync_accepts_int_event(self, server):
        server.send_sync(1, b"binary_data")
        assert server.events[0]["event"] == 1

    def test_queue_updated_does_not_raise(self, server):
        server.queue_updated()  # must be a no-op


# ---------------------------------------------------------------------------
# last_error helper
# ---------------------------------------------------------------------------

class TestLastError:
    def test_no_error_returns_none(self, server):
        server.send_sync("execution_start", {})
        assert server.last_error() is None

    def test_returns_last_error_data(self, server):
        server.send_sync("execution_start", {})
        server.send_sync("execution_error", {"exception_message": "OOM", "node_id": "5"})
        err = server.last_error()
        assert err is not None
        assert err["exception_message"] == "OOM"

    def test_returns_most_recent_error(self, server):
        server.send_sync("execution_error", {"exception_message": "first"})
        server.send_sync("execution_error", {"exception_message": "second"})
        assert server.last_error()["exception_message"] == "second"


# ---------------------------------------------------------------------------
# Progress queue
# ---------------------------------------------------------------------------

class TestProgressQueue:
    def test_progress_events_pushed_to_queue(self, server, progress_queue):
        server.attach_progress_queue(progress_queue)
        server.send_sync("progress", {"value": 3, "max": 10})
        assert not progress_queue.empty()
        item = progress_queue.get_nowait()
        assert item["event"] == "progress"

    def test_non_progress_events_not_pushed(self, server, progress_queue):
        server.attach_progress_queue(progress_queue)
        # int event (binary preview) should not be pushed
        server.send_sync(99, b"data")
        assert progress_queue.empty()

    def test_all_progress_event_types_pushed(self, server, progress_queue):
        server.attach_progress_queue(progress_queue)
        interesting = [
            "progress", "executing", "execution_start",
            "execution_cached", "execution_success",
            "execution_error", "execution_interrupted",
        ]
        for ev in interesting:
            server.send_sync(ev, {})
        assert progress_queue.qsize() == len(interesting)

    def test_detach_stops_pushing(self, server, progress_queue):
        server.attach_progress_queue(progress_queue)
        server.detach_progress_queue()
        server.send_sync("progress", {"value": 1})
        assert progress_queue.empty()

    def test_full_queue_does_not_raise(self, server):
        tiny_q: asyncio.Queue = asyncio.Queue(maxsize=1)
        server.attach_progress_queue(tiny_q)
        server.send_sync("progress", {"value": 1})
        server.send_sync("progress", {"value": 2})  # queue full — must not raise

    def test_events_still_collected_when_queue_full(self, server):
        tiny_q: asyncio.Queue = asyncio.Queue(maxsize=1)
        server.attach_progress_queue(tiny_q)
        server.send_sync("progress", {"value": 1})
        server.send_sync("progress", {"value": 2})
        assert len(server.events) == 2  # both stored even if queue dropped one


# ---------------------------------------------------------------------------
# InferenceEngine unit (no comfy imports)
# ---------------------------------------------------------------------------

class TestInferenceEngineUnit:
    def test_not_ready_before_initialize(self):
        from uncomfy.engine import InferenceEngine
        engine = InferenceEngine()
        assert not engine.ready

    @pytest.mark.asyncio
    async def test_run_before_initialize_raises(self):
        from uncomfy.engine import InferenceEngine
        engine = InferenceEngine()
        with pytest.raises(RuntimeError, match="not initialised"):
            await engine.run({})

    def test_output_images_empty_result(self):
        """output_images on empty result returns empty list — no comfy needed."""
        from uncomfy.engine import InferenceEngine
        import unittest.mock as mock

        engine = InferenceEngine()
        # Patch folder_paths so we don't need ComfyUI installed
        with mock.patch.dict("sys.modules", {"folder_paths": mock.MagicMock(
            get_output_directory=lambda: r"C:\ComfyUI\output"
        )}):
            result = engine.output_images({"outputs": {}})
        assert result == []

    def test_output_images_parses_correctly(self):
        from uncomfy.engine import InferenceEngine
        import unittest.mock as mock

        engine = InferenceEngine()
        history = {
            "outputs": {
                "13": {"images": [{"filename": "test_00001_.png", "subfolder": "", "type": "output"}]}
            }
        }
        with mock.patch.dict("sys.modules", {"folder_paths": mock.MagicMock(
            get_output_directory=lambda: r"C:\ComfyUI\output"
        )}):
            paths = engine.output_images(history)

        assert len(paths) == 1
        assert paths[0].name == "test_00001_.png"
