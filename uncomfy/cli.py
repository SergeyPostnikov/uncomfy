from __future__ import annotations

"""
uncomfy CLI — submit, status, list, output, watch

Usage:
    python -m uncomfy.cli submit --workflow img2img --prompt "..." [--image path]
    python -m uncomfy.cli status <job-id>
    python -m uncomfy.cli list [--status pending|running|done|failed]
    python -m uncomfy.cli output <job-id> [--save path]
    python -m uncomfy.cli watch <job-id>
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

BASE_URL = os.getenv("UNCOMFY_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=30)


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _check(r: httpx.Response) -> dict:
    if not r.is_success:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        _die(f"HTTP {r.status_code}: {detail}")
    return r.json()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    with _client() as c:
        if args.image:
            image_path = Path(args.image)
            if not image_path.exists():
                _die(f"image not found: {image_path}")
            with image_path.open("rb") as fh:
                r = c.post(
                    "/jobs",
                    params={"workflow": args.workflow, "prompt": args.prompt or ""},
                    files={"image": (image_path.name, fh, "image/png")},
                )
        else:
            payload: dict = {"workflow": args.workflow}
            if args.prompt:
                payload["prompt"] = args.prompt
            if args.params:
                try:
                    payload["params"] = json.loads(args.params)
                except json.JSONDecodeError as e:
                    _die(f"--params is not valid JSON: {e}")
            r = c.post("/jobs/json", json=payload)

    job = _check(r)
    print(job["id"])
    if args.verbose:
        _print_job(job)


def cmd_status(args: argparse.Namespace) -> None:
    with _client() as c:
        r = c.get(f"/jobs/{args.job_id}")
    job = _check(r)
    _print_job(job)


def cmd_list(args: argparse.Namespace) -> None:
    params: dict = {"limit": args.limit}
    if args.status:
        params["status"] = args.status
    with _client() as c:
        r = c.get("/jobs", params=params)
    jobs = _check(r)
    if not jobs:
        print("(no jobs)")
        return
    _print_table(jobs)


def cmd_output(args: argparse.Namespace) -> None:
    # Resolve filename from job if we got an ID
    job_id = args.job_id
    with _client() as c:
        job_r = c.get(f"/jobs/{job_id}")
    job = _check(job_r)
    if job["status"] != "done":
        _die(f"job {job_id[:8]} is not done (status={job['status']})")

    output_path = job.get("output_path") or ""
    if not output_path:
        _die("job has no output_path")

    filename = Path(output_path).name

    with _client() as c:
        r = c.get(f"/output/{filename}")
    if not r.is_success:
        _die(f"HTTP {r.status_code}: {r.text}")

    save_path = Path(args.save) if args.save else Path(filename)
    save_path.write_bytes(r.content)
    print(str(save_path.resolve()))


def cmd_watch(args: argparse.Namespace) -> None:
    asyncio.run(_watch(args.job_id))


async def _watch(job_id: str) -> None:
    url = BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{url}/ws/{job_id}"
    try:
        import websockets  # optional dep
    except ImportError:
        _die("websockets not installed — run: uv pip install websockets")

    print(f"watching job {job_id[:8]}…")
    async with websockets.connect(ws_url) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            event = msg.get("event", "")
            data = msg.get("data", {})

            if event == "ping":
                continue
            if event == "status":
                print(f"  status: {data.get('status')}")
            elif event == "progress":
                val = data.get("value", 0)
                mx = data.get("max", 1)
                bar = _bar(val, mx)
                print(f"\r  {bar} {val}/{mx}", end="", flush=True)
            elif event == "executing":
                node = data.get("node")
                if node:
                    print(f"\n  executing node {node}")
            elif event == "execution_success":
                print("\n  done")
                break
            elif event in ("execution_error", "execution_interrupted"):
                print(f"\n  {event}: {data}")
                break
            elif event == "error":
                print(f"  error: {data.get('message')}")
                break


def _bar(val: int, mx: int, width: int = 20) -> str:
    filled = int(width * val / max(mx, 1))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_job(job: dict) -> None:
    lines = [
        f"id:       {job['id']}",
        f"status:   {job['status']}",
        f"workflow: {job['workflow']}",
    ]
    if job.get("prompt"):
        lines.append(f"prompt:   {job['prompt'][:80]}")
    if job.get("duration_sec") is not None:
        lines.append(f"duration: {job['duration_sec']:.1f}s")
    if job.get("output_path"):
        lines.append(f"output:   {job['output_path']}")
    if job.get("error"):
        lines.append(f"error:    {job['error'][:120]}")
    print("\n".join(lines))


def _print_table(jobs: list[dict]) -> None:
    fmt = "{:<36}  {:<10}  {:<12}  {}"
    print(fmt.format("ID", "STATUS", "DURATION", "WORKFLOW"))
    print("-" * 72)
    for j in jobs:
        dur = f"{j['duration_sec']:.1f}s" if j.get("duration_sec") else "-"
        print(fmt.format(j["id"], j["status"], dur, j["workflow"]))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m uncomfy.cli",
        description="uncomfy — ComfyUI inference API client",
    )
    p.add_argument("--url", metavar="URL", help="API base URL (overrides UNCOMFY_URL)")
    sub = p.add_subparsers(dest="command", required=True)

    # submit
    s = sub.add_parser("submit", help="enqueue a new job")
    s.add_argument("--workflow", default="img2img", help="workflow name")
    s.add_argument("--prompt", default=None)
    s.add_argument("--image", default=None, metavar="PATH", help="input image")
    s.add_argument("--params", default=None, metavar="JSON", help='e.g. \'{"steps":20}\'')
    s.add_argument("-v", "--verbose", action="store_true")

    # status
    s = sub.add_parser("status", help="show job status")
    s.add_argument("job_id")

    # list
    s = sub.add_parser("list", help="list jobs")
    s.add_argument("--status", default=None, choices=["pending", "running", "done", "failed", "cancelled"])
    s.add_argument("--limit", type=int, default=20)

    # output
    s = sub.add_parser("output", help="download output image")
    s.add_argument("job_id")
    s.add_argument("--save", default=None, metavar="PATH")

    # watch
    s = sub.add_parser("watch", help="stream live progress for a job")
    s.add_argument("job_id")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.url:
        global BASE_URL
        BASE_URL = args.url.rstrip("/")

    dispatch = {
        "submit": cmd_submit,
        "status": cmd_status,
        "list": cmd_list,
        "output": cmd_output,
        "watch": cmd_watch,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
