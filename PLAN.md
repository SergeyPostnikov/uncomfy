# uncomfy — план реализации

ComfyUI без UI. FastAPI + TaskIQ + SQLAlchemy поверх голого inference engine.

## Цель

Выдрать `PromptExecutor` из ComfyUI, завернуть в нормальный async backend:
- REST API вместо aiohttp + вебсокет-UI
- TaskIQ вместо внутренней очереди ComfyUI
- SQLAlchemy для персистентности jobs/логов
- CLI для запуска без браузера

---

## Структура

```
uncomfy/
├── engine.py        # реализует ExecutionServer, оборачивает PromptExecutor
├── tasks.py         # TaskIQ задачи: generate, colorize
├── models.py        # SQLAlchemy: Job, Output
├── database.py      # async engine, session factory
├── schemas.py       # Pydantic: JobCreate, JobStatus, JobResult
├── main.py          # FastAPI app, роуты
├── cli.py           # CLI: submit / status / list
├── config.py        # pydantic-settings: пути, брокер, БД
└── workflows/       # JSON workflow файлы
    ├── img2img.json
    └── t2i.json
```

---

## Фазы

### Фаза 1 — engine.py
Реализовать `ExecutionServer` протокол:
```python
class StubServer:
    client_id = None
    last_node_id = None
    sockets_metadata = {}
    events: list  # сюда пишем send_sync события

    def send_sync(self, event, data, sid=None):
        self.events.append({"event": event, "data": data})

    def queue_updated(self):
        pass
```
Обернуть `PromptExecutor` в `InferenceEngine`:
- `__init__`: инициализирует executor, загружает ноды
- `async run(workflow, prompt_id)`: выполняет один промпт, возвращает output paths

Зависимости: только `sys.path` указывает на `C:\ComfyUI`

---

### Фаза 2 — database.py + models.py
```python
# models.py
class Job(Base):
    id: str          # UUID
    status: str      # pending | running | done | failed
    workflow: str    # имя workflow файла
    prompt: str      # текстовый промпт
    params: JSON     # denoise, steps, seed, etc.
    input_path: str  # путь к входному изображению
    output_path: str # путь к результату
    error: str       # traceback если failed
    created_at: datetime
    started_at: datetime
    finished_at: datetime
    duration_sec: float
```
Миграции через **Alembic**.

---

### Фаза 3 — tasks.py
```python
# TaskIQ задача
@broker.task
async def run_job(job_id: str):
    # 1. достать Job из БД
    # 2. обновить status → running
    # 3. engine.run(workflow, job_id)
    # 4. обновить status → done / failed, записать output_path, duration
```
Broker:
- `InMemoryBroker` для локального запуска
- `RedisBroker` для production

---

### Фаза 4 — main.py (FastAPI)
```
POST   /jobs              — создать job (prompt + workflow + файл)
GET    /jobs/{id}         — статус + результат
GET    /jobs              — список с фильтром по статусу
GET    /output/{filename} — скачать картинку
DELETE /jobs/{id}         — отменить pending job
GET    /health            — статус воркера
WS     /ws/{job_id}       — прогресс шагов в реальном времени
```

---

### Фаза 5 — cli.py
```bash
python -m uncomfy submit img2img.json --input page.png --prompt "colorize manga"
python -m uncomfy status <job_id>
python -m uncomfy list --status done
python -m uncomfy output <job_id>   # скачать результат
```

---

### Фаза 6 — WebSocket прогресс
`engine.py` перехватывает `send_sync` события (`progress`, `executing`) и
пушит их в asyncio.Queue. FastAPI WS endpoint читает очередь и шлёт клиенту:
```json
{"event": "progress", "value": 5, "max": 15, "node": "KSampler"}
{"event": "done", "output": "/output/qwen_i2i_00006_.png"}
```

---

## Зависимости

```toml
[project]
dependencies = [
    "fastapi",
    "uvicorn[standard]",
    "taskiq",
    "taskiq-redis",       # если Redis брокер
    "sqlalchemy[asyncio]",
    "aiosqlite",          # SQLite async драйвер
    "alembic",
    "pydantic-settings",
    "httpx",
    "python-multipart",
]
```

ComfyUI engine подключается через `sys.path` — не как пакет, а как сосед:
```python
import sys
sys.path.insert(0, r"C:\ComfyUI")
```

---

## Что НЕ берём из ComfyUI

- `server.py` — весь aiohttp сервер
- `main.py` — argparse, запуск
- `app/` — веб UI
- `api_server/` — их незаконченный API
- `blueprints/`, `middleware/` — aiohttp роуты
- `latent_preview.py` — превью для UI

---

## Порядок запуска

```bash
# 1. воркер
python -m taskiq worker uncomfy.tasks:broker

# 2. API сервер
uvicorn uncomfy.main:app --host 0.0.0.0 --port 8000 --reload

# 3. отправить задачу
python -m uncomfy submit img2img.json --input manga.png
```
