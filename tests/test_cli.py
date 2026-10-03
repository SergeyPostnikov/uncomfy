"""
Unit tests for CLI — mock httpx, no real server needed.
"""
from __future__ import annotations

import json
import sys
import uuid
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import httpx

# Make module importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from uncomfy import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, body) -> httpx.Response:
    content = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    return httpx.Response(status_code, content=content)


def _job(status="pending", **kw) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "status": status,
        "workflow": "img2img",
        "prompt": None,
        "params": None,
        "input_path": None,
        "output_path": None,
        "error": None,
        "created_at": "2026-10-01T12:00:00",
        "started_at": None,
        "finished_at": None,
        "duration_sec": None,
        **kw,
    }


# ---------------------------------------------------------------------------
# _check
# ---------------------------------------------------------------------------

class TestCheck:
    def test_success_returns_json(self):
        r = _make_response(200, {"id": "abc"})
        assert cli._check(r) == {"id": "abc"}

    def test_error_exits(self):
        r = _make_response(404, {"detail": "not found"})
        with pytest.raises(SystemExit):
            cli._check(r)

    def test_error_message_shown(self, capsys):
        r = _make_response(500, {"detail": "server error"})
        with pytest.raises(SystemExit):
            cli._check(r)
        assert "500" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

class TestSubmit:
    def _args(self, **kw):
        ns = MagicMock()
        ns.workflow = "img2img"
        ns.prompt = "test"
        ns.image = None
        ns.params = None
        ns.verbose = False
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_prints_job_id(self, capsys):
        job = _job()
        mock_r = _make_response(202, job)
        with patch("uncomfy.cli._client") as mc:
            mc.return_value.__enter__.return_value.post.return_value = mock_r
            cli.cmd_submit(self._args())
        assert job["id"] in capsys.readouterr().out

    def test_verbose_prints_details(self, capsys):
        job = _job(prompt="test prompt", workflow="img2img")
        mock_r = _make_response(202, job)
        with patch("uncomfy.cli._client") as mc:
            mc.return_value.__enter__.return_value.post.return_value = mock_r
            cli.cmd_submit(self._args(verbose=True))
        out = capsys.readouterr().out
        assert "img2img" in out

    def test_invalid_params_json_exits(self, capsys):
        args = self._args(params="not-json")
        with pytest.raises(SystemExit):
            cli.cmd_submit(args)

    def test_missing_image_exits(self, capsys, tmp_path):
        args = self._args(image=str(tmp_path / "nonexistent.png"))
        with pytest.raises(SystemExit):
            cli.cmd_submit(args)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_prints_job_info(self, capsys):
        job = _job(status="done", duration_sec=15.3)
        mock_r = _make_response(200, job)
        args = MagicMock(); args.job_id = job["id"]
        with patch("uncomfy.cli._client") as mc:
            mc.return_value.__enter__.return_value.get.return_value = mock_r
            cli.cmd_status(args)
        out = capsys.readouterr().out
        assert "done" in out
        assert "15.3s" in out

    def test_404_exits(self):
        mock_r = _make_response(404, {"detail": "not found"})
        args = MagicMock(); args.job_id = "bad-id"
        with patch("uncomfy.cli._client") as mc:
            mc.return_value.__enter__.return_value.get.return_value = mock_r
            with pytest.raises(SystemExit):
                cli.cmd_status(args)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

class TestList:
    def test_prints_table(self, capsys):
        jobs = [_job(status="done", duration_sec=5.0) for _ in range(3)]
        mock_r = _make_response(200, jobs)
        args = MagicMock(); args.status = None; args.limit = 20
        with patch("uncomfy.cli._client") as mc:
            mc.return_value.__enter__.return_value.get.return_value = mock_r
            cli.cmd_list(args)
        out = capsys.readouterr().out
        assert "done" in out
        assert "img2img" in out

    def test_empty_list_message(self, capsys):
        mock_r = _make_response(200, [])
        args = MagicMock(); args.status = None; args.limit = 20
        with patch("uncomfy.cli._client") as mc:
            mc.return_value.__enter__.return_value.get.return_value = mock_r
            cli.cmd_list(args)
        assert "no jobs" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

class TestOutput:
    def test_saves_file(self, tmp_path, capsys):
        job = _job(status="done", output_path=r"C:\ComfyUI\output\test_001.png")
        job_r = _make_response(200, job)
        file_r = httpx.Response(200, content=b"PNG_DATA")

        save_path = tmp_path / "out.png"
        args = MagicMock()
        args.job_id = job["id"]
        args.save = str(save_path)

        with patch("uncomfy.cli._client") as mc:
            inst = mc.return_value.__enter__.return_value
            inst.get.side_effect = [job_r, file_r]
            cli.cmd_output(args)

        assert save_path.read_bytes() == b"PNG_DATA"

    def test_not_done_exits(self):
        job = _job(status="running")
        mock_r = _make_response(200, job)
        args = MagicMock(); args.job_id = job["id"]; args.save = None
        with patch("uncomfy.cli._client") as mc:
            mc.return_value.__enter__.return_value.get.return_value = mock_r
            with pytest.raises(SystemExit):
                cli.cmd_output(args)


# ---------------------------------------------------------------------------
# tqdm watch — import smoke test
# ---------------------------------------------------------------------------

class TestWatchImport:
    def test_tqdm_importable(self):
        from tqdm import tqdm
        assert tqdm is not None

    def test_watch_command_registered(self):
        p = cli._build_parser()
        a = p.parse_args(["watch", "some-job-id"])
        assert a.command == "watch"
        assert a.job_id == "some-job-id"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestParser:
    def _parse(self, *args):
        return cli._build_parser().parse_args(args)

    def test_submit_defaults(self):
        a = self._parse("submit", "--prompt", "hello")
        assert a.workflow == "img2img"
        assert a.prompt == "hello"
        assert not a.verbose

    def test_list_status_filter(self):
        a = self._parse("list", "--status", "done")
        assert a.status == "done"

    def test_list_invalid_status_exits(self):
        with pytest.raises(SystemExit):
            self._parse("list", "--status", "bogus")

    def test_watch_parses_job_id(self):
        a = self._parse("watch", "abc-123")
        assert a.job_id == "abc-123"

    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            self._parse()
