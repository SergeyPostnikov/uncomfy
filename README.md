# uncomfy

ComfyUI inference engine without the UI — FastAPI + TaskIQ + SQLAlchemy wrapper.

## Based on

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous et al., licensed under GPL-3.0.

This project wraps ComfyUI's `PromptExecutor` with a clean async Python interface.
ComfyUI must be installed separately at `C:\ComfyUI` (or configured via `COMFY_PATH`).

## Stack

- **FastAPI** — REST API
- **TaskIQ** — async task queue
- **SQLAlchemy** — job persistence
- **ComfyUI** — inference engine (runtime dependency)

## License

GPL-3.0 — see [LICENSE](LICENSE).
