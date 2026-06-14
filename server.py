import os
import asyncio
import base64
import httpx
import subprocess
import tempfile
import shutil
import io
import json
import re
import logging
import time
from pathlib import Path
from PIL import Image
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

logger = logging.getLogger("pandocr")
logging.basicConfig(level=os.getenv("PANDOCR_LOG_LEVEL", "INFO"))


def parse_csv_env(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


PADDLE_SERVICE_URL = os.getenv("PADDLE_SERVICE_URL", "http://localhost:8081/layout-parsing")
PADDLEOCR_VL_MODEL_NAME = os.getenv("PADDLEOCR_VL_MODEL_NAME", "PaddleOCR-VL-1.6-0.9B")
PADDLE_OCR_SERVICE_URL = os.getenv("PADDLE_OCR_SERVICE_URL", "http://localhost:8082/ocr")
PPOCR_V6_MODEL_NAME = os.getenv("PPOCR_V6_MODEL_NAME", "PP-OCRv6_medium")
PADDLE_REQUEST_TIMEOUT = float(os.getenv("PADDLE_REQUEST_TIMEOUT", "3600"))
TASK_DATA_DIR = Path(os.getenv("PANDOCR_TASK_DATA_DIR", "data/tasks")).resolve()
MAX_REQUEST_BYTES = int(float(os.getenv("PANDOCR_MAX_UPLOAD_MB", "512")) * 1024 * 1024)
PANDOCR_HOST = os.getenv("PANDOCR_HOST", "0.0.0.0")
PANDOCR_PORT = int(os.getenv("PANDOCR_PORT", "8000"))
MODEL_CONTROL_MODE = os.getenv("PANDOCR_MODEL_CONTROL", "docker").strip().lower()
MODEL_RUNTIME_STARTUP = os.getenv("PANDOCR_ACTIVE_MODEL_ON_START", "paddleocr-vl-1.6").strip()
DOCKER_SOCKET_PATH = os.getenv("PANDOCR_DOCKER_SOCKET", "/var/run/docker.sock")
MODEL_SWITCH_TIMEOUT = float(os.getenv("PANDOCR_MODEL_SWITCH_TIMEOUT", "1200"))
CORS_ORIGINS = parse_csv_env(
    "PANDOCR_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
)

MODEL_RUNTIME_CONFIG = {
    "paddleocr-vl-1.6": {
        "containers": ["paddleocr-vlm-server", "paddleocr-vl-api"],
        "start_order": ["paddleocr-vlm-server", "paddleocr-vl-api"],
        "stop_order": ["paddleocr-vl-api", "paddleocr-vlm-server"],
        "health_url": PADDLE_SERVICE_URL.rsplit("/", 1)[0] + "/health",
    },
    "pp-ocrv6": {
        "containers": ["paddleocr-ocr-api"],
        "start_order": ["paddleocr-ocr-api"],
        "stop_order": ["paddleocr-ocr-api"],
        "health_url": PADDLE_OCR_SERVICE_URL.rsplit("/", 1)[0] + "/health",
    },
}
DEFAULT_RUNTIME_MODEL_ID = MODEL_RUNTIME_STARTUP if MODEL_RUNTIME_STARTUP in MODEL_RUNTIME_CONFIG else "paddleocr-vl-1.6"

model_runtime_lock = asyncio.Lock()
model_runtime_operation = {
    "targetModelId": DEFAULT_RUNTIME_MODEL_ID,
    "state": "idle",
    "message": "",
    "startedAt": None,
    "updatedAt": None,
}
model_runtime_task: asyncio.Task | None = None


class ModelSwitchRequest(BaseModel):
    modelId: str


def model_catalog() -> list[dict]:
    return [
        {
            "id": "paddleocr-vl-1.6",
            "name": PADDLEOCR_VL_MODEL_NAME,
            "label": "PaddleOCR-VL 1.6",
            "kind": "document_parsing",
            "endpoint": "/api/paddleocr-vl-1.6",
        },
        {
            "id": "pp-ocrv6",
            "name": PPOCR_V6_MODEL_NAME,
            "label": "PP-OCRv6",
            "kind": "text_ocr",
            "endpoint": "/api/pp-ocrv6",
        },
    ]


def model_control_available() -> bool:
    return MODEL_CONTROL_MODE == "docker" and Path(DOCKER_SOCKET_PATH).exists()


async def docker_api_request(method: str, path: str, *, timeout: float = 30) -> httpx.Response:
    transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET_PATH)
    async with httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=timeout) as client:
        return await client.request(method, path)


async def inspect_container(name: str) -> dict:
    if not model_control_available():
        return {
            "name": name,
            "exists": False,
            "running": False,
            "state": "unknown",
            "health": "unknown",
        }

    response = await docker_api_request("GET", f"/containers/{name}/json")
    if response.status_code == 404:
        return {
            "name": name,
            "exists": False,
            "running": False,
            "state": "missing",
            "health": "missing",
        }
    response.raise_for_status()
    payload = response.json()
    state = payload.get("State") or {}
    health = state.get("Health") or {}
    return {
        "name": name,
        "exists": True,
        "running": bool(state.get("Running")),
        "state": state.get("Status") or "unknown",
        "health": health.get("Status") or "none",
    }


async def docker_container_action(name: str, action: str) -> None:
    if not model_control_available():
        raise RuntimeError("Docker model control is not available")
    if action == "stop":
        response = await docker_api_request("POST", f"/containers/{name}/stop?t=20", timeout=45)
        if response.status_code in {204, 304, 404}:
            return
    elif action == "start":
        response = await docker_api_request("POST", f"/containers/{name}/start", timeout=45)
        if response.status_code in {204, 304}:
            return
    else:
        raise ValueError(f"Unsupported container action: {action}")
    if response.status_code >= 400:
        raise RuntimeError(f"Docker {action} failed for {name}: {response.text}")


async def check_http_health(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url)
        return 200 <= response.status_code < 300
    except Exception:
        return False


async def model_runtime_status(model_id: str) -> dict:
    config = MODEL_RUNTIME_CONFIG[model_id]
    containers = [await inspect_container(name) for name in config["containers"]]
    if not model_control_available():
        health_ok = await check_http_health(config["health_url"])
        return {
            "id": model_id,
            "containers": containers,
            "running": health_ok,
            "ready": health_ok,
            "state": "ready" if health_ok else "unknown",
            "healthUrl": config["health_url"],
        }

    any_running = any(container["running"] for container in containers)
    all_running = all(container["running"] for container in containers)
    any_missing = any(not container["exists"] for container in containers)
    health_ok = await check_http_health(config["health_url"]) if all_running else False

    if any_missing:
        state = "missing"
    elif health_ok:
        state = "ready"
    elif any_running:
        state = "starting" if all_running else "partial"
    else:
        state = "stopped"

    return {
        "id": model_id,
        "containers": containers,
        "running": any_running,
        "ready": health_ok,
        "state": state,
        "healthUrl": config["health_url"],
    }


async def build_model_runtime_payload() -> dict:
    models = {
        model_id: await model_runtime_status(model_id)
        for model_id in MODEL_RUNTIME_CONFIG
    }
    ready_models = [model_id for model_id, status in models.items() if status["ready"]]
    running_models = [model_id for model_id, status in models.items() if status["running"]]
    active_model = ready_models[0] if ready_models else (running_models[0] if running_models else None)
    return {
        "controlMode": MODEL_CONTROL_MODE,
        "controlAvailable": model_control_available(),
        "activeModelId": active_model,
        "defaultModelId": DEFAULT_RUNTIME_MODEL_ID,
        "operation": dict(model_runtime_operation),
        "models": models,
    }


def set_model_runtime_operation(state: str, message: str = "", target_model_id: str | None = None) -> None:
    now = time.time()
    if target_model_id:
        model_runtime_operation["targetModelId"] = target_model_id
    model_runtime_operation["state"] = state
    model_runtime_operation["message"] = message
    model_runtime_operation["updatedAt"] = now
    if state == "switching":
        model_runtime_operation["startedAt"] = now


async def wait_model_ready(model_id: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await model_runtime_status(model_id)
        if status["ready"]:
            return
        await asyncio.sleep(3)
    raise TimeoutError(f"Timed out waiting for {model_id} to become ready")


async def wait_container_runtime_ready(container_name: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await inspect_container(container_name)
        if not status["exists"]:
            raise RuntimeError(f"Docker container {container_name} is missing. Run docker compose up --no-start first.")
        if status["running"] and status["health"] in {"healthy", "none"}:
            return
        await asyncio.sleep(3)
    raise TimeoutError(f"Timed out waiting for Docker container {container_name} to become healthy")


async def activate_model_runtime(model_id: str) -> None:
    if model_id not in MODEL_RUNTIME_CONFIG:
        raise ValueError(f"Unknown model id: {model_id}")
    if not model_control_available():
        raise RuntimeError("Docker model control is not available")

    async with model_runtime_lock:
        set_model_runtime_operation("switching", f"Switching to {model_id}", model_id)
        switch_started_at = time.monotonic()
        try:
            for other_model_id, config in MODEL_RUNTIME_CONFIG.items():
                if other_model_id == model_id:
                    continue
                for container_name in config["stop_order"]:
                    await docker_container_action(container_name, "stop")

            for container_name in MODEL_RUNTIME_CONFIG[model_id]["start_order"]:
                remaining_timeout = max(3, MODEL_SWITCH_TIMEOUT - (time.monotonic() - switch_started_at))
                await docker_container_action(container_name, "start")
                await wait_container_runtime_ready(container_name, remaining_timeout)

            remaining_timeout = max(3, MODEL_SWITCH_TIMEOUT - (time.monotonic() - switch_started_at))
            await wait_model_ready(model_id, remaining_timeout)
            set_model_runtime_operation("ready", f"{model_id} is ready", model_id)
        except Exception as err:
            logger.exception("Model runtime switch failed")
            set_model_runtime_operation("error", str(err), model_id)


def schedule_model_runtime_activation(model_id: str) -> None:
    global model_runtime_task
    if model_id not in MODEL_RUNTIME_CONFIG:
        raise HTTPException(status_code=400, detail="Unknown model id")
    if not model_control_available():
        raise HTTPException(status_code=503, detail="Docker model control is not available")
    if model_runtime_task and not model_runtime_task.done():
        model_runtime_task.cancel()
    set_model_runtime_operation("switching", f"Switching to {model_id}", model_id)
    model_runtime_task = asyncio.create_task(activate_model_runtime(model_id))

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def autostart_default_model_runtime():
    if model_control_available():
        schedule_model_runtime_activation(DEFAULT_RUNTIME_MODEL_ID)


@app.middleware("http")
async def reject_oversized_requests(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"} and MAX_REQUEST_BYTES > 0:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    max_mb = MAX_REQUEST_BYTES / 1024 / 1024
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request body is too large. Max upload size is {max_mb:.0f} MB."},
                    )
            except ValueError:
                pass
    return await call_next(request)


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


@app.get("/api/models")
async def get_models():
    """Return OCR models available through this proxy."""
    return {
        "default": DEFAULT_RUNTIME_MODEL_ID,
        "data": model_catalog(),
    }


@app.get("/api/model-runtime")
async def get_model_runtime():
    return await build_model_runtime_payload()


@app.post("/api/model-runtime/switch")
async def switch_model_runtime(request: ModelSwitchRequest):
    schedule_model_runtime_activation(request.modelId)
    return await build_model_runtime_payload()


def safe_task_id(task_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,80}", task_id or ""):
        raise HTTPException(status_code=400, detail="Invalid task id")
    return task_id


def task_file_path(task_id: str) -> Path:
    return TASK_DATA_DIR / safe_task_id(task_id) / "task.json"


def task_dir_path(task_id: str) -> Path:
    return TASK_DATA_DIR / safe_task_id(task_id)


def task_source_path(task_id: str) -> Path:
    return task_dir_path(task_id) / "source.bin"


def task_source_url(task_id: str) -> str:
    return f"/api/tasks/{safe_task_id(task_id)}/source"


def sanitize_task_for_storage(task: dict) -> dict:
    """Keep task JSON metadata lightweight even if an older client sends heavy fields."""
    task_id = task.get("id")
    source_url = task.get("sourceUrl")
    has_external_source = bool(source_url) or (isinstance(task_id, str) and task_source_path(task_id).exists())

    stored = dict(task)
    stored.pop("detailLoaded", None)
    if has_external_source:
        stored["sourceUrl"] = source_url or task_source_url(task_id)
        stored.pop("sourceDataUrl", None)

    batches = stored.get("batches") if isinstance(stored.get("batches"), list) else []
    compact_batches = []
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        compact = dict(batch)
        compact.pop("payloadDataUrl", None)
        compact_batches.append(compact)
    stored["batches"] = compact_batches
    return stored


def task_summary(task: dict) -> dict:
    batches = task.get("batches") if isinstance(task.get("batches"), list) else []
    completed_pages = sum(
        int(batch.get("pageCount") or 0)
        for batch in batches
        if isinstance(batch, dict) and batch.get("status") == "completed"
    )
    return {
        "id": task.get("id"),
        "name": task.get("name"),
        "originalName": task.get("originalName"),
        "sourceKind": task.get("sourceKind"),
        "mimeType": task.get("mimeType"),
        "size": task.get("size"),
        "createdAt": task.get("createdAt"),
        "updatedAt": task.get("updatedAt"),
        "status": task.get("status"),
        "pageCount": task.get("pageCount"),
        "pdfBatchSize": task.get("pdfBatchSize"),
        "sourceUrl": task.get("sourceUrl"),
        "modelId": task.get("modelId"),
        "modelName": task.get("modelName"),
        "error": task.get("error"),
        "completedPages": completed_pages,
        "batchCount": len(batches),
        "hasMarkdown": bool(task.get("markdown")),
        "hasOcrResults": bool(task.get("ocrResults")),
        "detailLoaded": False,
    }


def read_task_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        task = json.load(f)
    if not isinstance(task, dict):
        raise ValueError("Task file must contain a JSON object")
    return task


def write_task_file(path: Path, task: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(task, f, ensure_ascii=False)
    temp_path.replace(path)


def list_task_summaries() -> list[dict]:
    TASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for path in TASK_DATA_DIR.glob("*/task.json"):
        try:
            tasks.append(task_summary(read_task_file(path)))
        except (OSError, ValueError, json.JSONDecodeError) as err:
            logger.warning("Skipping invalid task file %s: %s", path, err)
    tasks.sort(key=lambda item: item.get("updatedAt") or 0, reverse=True)
    return tasks


def remove_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def extract_pdf_pages(source_path: Path, start_page: int, end_page: int) -> bytes:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(source_path))
    total_pages = len(reader.pages)
    if total_pages <= 0:
        raise ValueError("Source PDF has no pages")
    if start_page < 1 or end_page < start_page or start_page > total_pages:
        raise ValueError(f"Invalid page range {start_page}-{end_page} for {total_pages} pages")

    end_page = min(end_page, total_pages)
    writer = PdfWriter()
    for page_index in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_index])

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@app.post("/api/tasks/{task_id}/source")
async def upload_task_source(task_id: str, file: UploadFile = File(...)):
    """Persist the original uploaded source outside task.json."""
    source_path = task_source_path(task_id)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = source_path.with_suffix(".tmp")
    with temp_path.open("wb") as buffer:
        await run_in_threadpool(shutil.copyfileobj, file.file, buffer)
    temp_path.replace(source_path)
    return {
        "ok": True,
        "url": task_source_url(task_id),
        "size": source_path.stat().st_size,
        "filename": Path(file.filename or "source").name,
        "contentType": file.content_type or "application/octet-stream",
    }


@app.get("/api/tasks/{task_id}/source")
async def get_task_source(task_id: str):
    """Return the original uploaded source file for previewing or resumable parsing."""
    source_path = task_source_path(task_id)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Task source not found")

    media_type = "application/octet-stream"
    filename = "source"
    task_path = task_file_path(task_id)
    if task_path.exists():
        try:
            task = await run_in_threadpool(read_task_file, task_path)
            media_type = task.get("mimeType") or media_type
            filename = task.get("originalName") or task.get("name") or filename
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return FileResponse(source_path, media_type=media_type, filename=filename)


@app.get("/api/tasks/{task_id}/source/pages")
async def get_task_source_pages(
    task_id: str,
    start_page: int = Query(..., ge=1),
    end_page: int = Query(..., ge=1),
):
    """Return a compact PDF containing only a page range from the source PDF."""
    source_path = task_source_path(task_id)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Task source not found")
    if end_page < start_page:
        raise HTTPException(status_code=400, detail="end_page must be greater than or equal to start_page")

    try:
        pdf_content = await run_in_threadpool(extract_pdf_pages, source_path, start_page, end_page)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("Failed to extract PDF pages")
        raise HTTPException(status_code=500, detail=f"Failed to extract PDF pages: {err}") from err

    return Response(content=pdf_content, media_type="application/pdf")


@app.get("/api/tasks")
async def list_tasks():
    """List locally persisted document parsing task summaries."""
    tasks = await run_in_threadpool(list_task_summaries)
    return {"tasks": tasks}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Return one full locally persisted task."""
    path = task_file_path(task_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        task = await run_in_threadpool(read_task_file, path)
    except (OSError, ValueError, json.JSONDecodeError) as err:
        logger.warning("Failed to read task file %s: %s", path, err)
        raise HTTPException(status_code=500, detail="Failed to read task")
    if task_source_path(task_id).exists() and not task.get("sourceUrl"):
        task["sourceUrl"] = task_source_url(task_id)
    task["detailLoaded"] = True
    return task


@app.put("/api/tasks/{task_id}")
async def save_task(task_id: str, request: Request):
    """Persist one task to the local project data directory."""
    task = await request.json()
    if not isinstance(task, dict):
        raise HTTPException(status_code=400, detail="Task payload must be a JSON object")
    if task.get("id") != task_id:
        raise HTTPException(status_code=400, detail="Task id mismatch")

    stored_task = sanitize_task_for_storage(task)
    path = task_file_path(task_id)
    await run_in_threadpool(write_task_file, path, stored_task)
    return {"ok": True, "task": task_summary(stored_task)}


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete one locally persisted task."""
    path = task_file_path(task_id)
    await run_in_threadpool(remove_path, path.parent)
    return {"ok": True}


@app.delete("/api/tasks")
async def clear_tasks():
    """Delete all locally persisted tasks."""
    await run_in_threadpool(remove_path, TASK_DATA_DIR)
    await run_in_threadpool(TASK_DATA_DIR.mkdir, parents=True, exist_ok=True)
    return {"ok": True}


@app.post("/api/convert/to-pdf")
async def convert_to_pdf(file: UploadFile = File(...)):
    """Convert PPT/PPTX/DOC/DOCX to PDF using LibreOffice."""
    logger.info("Received conversion request for: %s", file.filename)

    if not shutil.which("soffice"):
        raise HTTPException(
            status_code=500,
            detail="LibreOffice (soffice) not found on server. Please install it to support Office conversion.",
        )

    filename = Path(file.filename or "upload").name
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".ppt", ".pptx", ".doc", ".docx"]:
        raise HTTPException(status_code=400, detail="Only .ppt, .pptx, .doc, and .docx files are supported.")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, filename)

            with open(input_path, "wb") as buffer:
                await run_in_threadpool(shutil.copyfileobj, file.file, buffer)

            cmd = [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                temp_dir,
                input_path,
            ]

            logger.info("Running conversion command: %s", " ".join(cmd))
            result = await run_in_threadpool(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )

            if result.returncode != 0:
                logger.warning("Conversion failed: %s", result.stderr)
                raise HTTPException(status_code=500, detail=f"Conversion failed: {result.stderr}")

            pdfs = [f for f in os.listdir(temp_dir) if f.lower().endswith(".pdf")]
            if not pdfs:
                raise HTTPException(status_code=500, detail="PDF file not generated")

            pdf_path = os.path.join(temp_dir, pdfs[0])
            logger.info("Conversion successful, sending back: %s", pdf_path)

            with open(pdf_path, "rb") as f:
                pdf_content = await run_in_threadpool(f.read)

            return Response(content=pdf_content, media_type="application/pdf")

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="File conversion timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during conversion")
        raise HTTPException(status_code=500, detail=str(e))


class OCRRequest(BaseModel):
    image: Optional[str] = None
    fileType: Optional[int] = None
    useLayoutDetection: bool = True
    useDocUnwarping: bool = False
    useDocOrientationClassify: bool = False
    useTextlineOrientation: bool = False
    useChartRecognition: bool = False
    useSealRecognition: bool = True
    formatBlockContent: bool = True
    showFormulaNumber: bool = True
    markdownIgnoreLabels: List[str] = Field(default_factory=list)
    layoutThreshold: Optional[float] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[float] = None
    layoutMergeBboxesMode: Optional[str] = None
    repetitionPenalty: Optional[float] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    minPixels: Optional[int] = None
    maxPixels: Optional[int] = None
    visualize: Optional[bool] = None


RawOCRInput = Union[bytes, str]


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_optional_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def parse_optional_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def parse_optional_string(value) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def parse_markdown_ignore_labels(value) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [text]


async def parse_ocr_input(request: Request) -> tuple[OCRRequest, RawOCRInput]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if not upload or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="Missing multipart field: file")

        file_bytes = await upload.read()
        ocr_request = OCRRequest(
            fileType=parse_optional_int(form.get("fileType")),
            useLayoutDetection=parse_bool(form.get("useLayoutDetection"), True),
            useDocUnwarping=parse_bool(form.get("useDocUnwarping"), False),
            useDocOrientationClassify=parse_bool(form.get("useDocOrientationClassify"), False),
            useTextlineOrientation=parse_bool(form.get("useTextlineOrientation"), False),
            useChartRecognition=parse_bool(form.get("useChartRecognition"), False),
            useSealRecognition=parse_bool(form.get("useSealRecognition"), True),
            formatBlockContent=parse_bool(form.get("formatBlockContent"), True),
            showFormulaNumber=parse_bool(form.get("showFormulaNumber"), True),
            markdownIgnoreLabels=parse_markdown_ignore_labels(form.get("markdownIgnoreLabels")),
            layoutThreshold=parse_optional_float(form.get("layoutThreshold")),
            layoutNms=parse_bool(form.get("layoutNms")) if form.get("layoutNms") is not None else None,
            layoutUnclipRatio=parse_optional_float(form.get("layoutUnclipRatio")),
            layoutMergeBboxesMode=parse_optional_string(form.get("layoutMergeBboxesMode")),
            repetitionPenalty=parse_optional_float(form.get("repetitionPenalty")),
            temperature=parse_optional_float(form.get("temperature")),
            topP=parse_optional_float(form.get("topP")),
            minPixels=parse_optional_int(form.get("minPixels")),
            maxPixels=parse_optional_int(form.get("maxPixels")),
            visualize=parse_bool(form.get("visualize")) if form.get("visualize") is not None else None,
        )
        return ocr_request, file_bytes

    payload = await request.json()
    ocr_request = OCRRequest(**payload)
    if not ocr_request.image:
        raise HTTPException(status_code=400, detail="Missing JSON field: image")
    return ocr_request, ocr_request.image


def normalize_raw_input_to_base64(raw_input: RawOCRInput) -> str:
    if isinstance(raw_input, bytes):
        return base64.b64encode(raw_input).decode("utf-8")
    if "base64," in raw_input:
        return raw_input.split("base64,")[1]
    return raw_input


def raw_input_to_bytes(raw_input: RawOCRInput) -> bytes:
    if isinstance(raw_input, bytes):
        return raw_input
    normalized = raw_input.split("base64,")[1] if "base64," in raw_input else raw_input
    try:
        return base64.b64decode(normalized, validate=True)
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid base64 input") from err


def prepare_service_input(ocr_request: OCRRequest, raw_input: RawOCRInput) -> tuple[str, int]:
    base64_data = normalize_raw_input_to_base64(raw_input)
    file_type = ocr_request.fileType

    if file_type is None:
        if isinstance(raw_input, bytes):
            if raw_input.startswith(b"%PDF-"):
                file_type = 0
                logger.info("Auto-detected PDF input")
            else:
                file_type = 1
                logger.info("Auto-detected Image input")
        elif base64_data.startswith("JVBERi0"):
            file_type = 0
            logger.info("Auto-detected PDF input")
        else:
            file_type = 1
            logger.info("Auto-detected Image input")

    if file_type == 1:
        try:
            img_bytes = raw_input_to_bytes(raw_input)
            img = Image.open(io.BytesIO(img_bytes))
            if img.format == "GIF":
                logger.info("GIF detected, converting to static JPEG for OCR")
                img.seek(0)
                rgb_img = img.convert("RGB")
                buffer = io.BytesIO()
                rgb_img.save(buffer, format="JPEG", quality=95)
                base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                logger.info("GIF conversion successful")
        except Exception as gif_err:
            logger.info("GIF conversion skipped: %s", gif_err)

    return base64_data, file_type


def build_pipeline_payload(request: OCRRequest, base64_data: str, file_type: int) -> dict:
    payload = {
        "file": base64_data,
        "fileType": file_type,
        "useLayoutDetection": request.useLayoutDetection,
        "useDocUnwarping": request.useDocUnwarping,
        "useDocOrientationClassify": request.useDocOrientationClassify,
        "useChartRecognition": request.useChartRecognition,
        "useSealRecognition": request.useSealRecognition,
        "formatBlockContent": request.formatBlockContent,
        "showFormulaNumber": request.showFormulaNumber,
        "prettifyMarkdown": True,
    }
    optional_params = [
        "markdownIgnoreLabels",
        "layoutThreshold",
        "layoutNms",
        "layoutUnclipRatio",
        "layoutMergeBboxesMode",
        "repetitionPenalty",
        "temperature",
        "topP",
        "minPixels",
        "maxPixels",
        "visualize",
    ]
    for param in optional_params:
        val = getattr(request, param)
        if val is not None:
            payload[param] = val
    return payload


def build_ppocr_payload(request: OCRRequest, base64_data: str, file_type: int) -> dict:
    payload = {
        "file": base64_data,
        "fileType": file_type,
        "useDocOrientationClassify": request.useDocOrientationClassify,
        "useDocUnwarping": request.useDocUnwarping,
        "useTextlineOrientation": request.useTextlineOrientation,
    }
    if request.visualize is not None:
        payload["visualize"] = request.visualize
    return payload


def parse_pipeline_response(data: dict, image_prefix: str = "") -> dict:
    if "result" not in data or "layoutParsingResults" not in data["result"]:
        logger.warning("Unexpected pipeline response format: %s", data)
        raise HTTPException(status_code=500, detail="Unexpected response format from Pipeline")

    results = data["result"]["layoutParsingResults"]
    full_markdown = ""
    all_images = {}

    for res in results:
        if "markdown" in res and "text" in res["markdown"]:
            md_text = res["markdown"]["text"]
            md_images = res["markdown"].get("images", {})
            if md_images:
                for img_path, img_base64 in md_images.items():
                    key = f"{image_prefix}_{img_path}" if image_prefix else img_path
                    all_images[key] = img_base64
            full_markdown += md_text + "\n\n"

    return {
        "markdown": full_markdown,
        "images": all_images,
        "layoutParsingResults": results,
    }


def as_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: as_jsonable(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    return value


def pick_indexed_value(values, index):
    if isinstance(values, list) and index < len(values):
        return as_jsonable(values[index])
    return None


def extract_ppocr_lines(pruned_result: dict) -> list[dict]:
    texts = pruned_result.get("rec_texts") if isinstance(pruned_result.get("rec_texts"), list) else []
    scores = pruned_result.get("rec_scores") if isinstance(pruned_result.get("rec_scores"), list) else []
    boxes = pruned_result.get("rec_boxes")
    polys = pruned_result.get("rec_polys")
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(polys, "tolist"):
        polys = polys.tolist()

    lines = []
    for index, text in enumerate(texts):
        line = {
            "text": str(text),
            "score": pick_indexed_value(scores, index),
        }
        box = pick_indexed_value(boxes, index)
        poly = pick_indexed_value(polys, index)
        if box is not None:
            line["box"] = box
        if poly is not None:
            line["poly"] = poly
        lines.append(line)
    return lines


def parse_ppocr_response(data: dict) -> dict:
    if "result" not in data or "ocrResults" not in data["result"]:
        logger.warning("Unexpected PP-OCR response format: %s", data)
        raise HTTPException(status_code=500, detail="Unexpected response format from PP-OCR service")

    pages = []
    full_markdown_parts = []
    for page_index, page_result in enumerate(data["result"]["ocrResults"]):
        pruned = page_result.get("prunedResult") if isinstance(page_result, dict) else {}
        if not isinstance(pruned, dict):
            pruned = {}
        pruned = as_jsonable(pruned)
        lines = extract_ppocr_lines(pruned)
        markdown_text = "\n".join(line["text"] for line in lines if line.get("text"))
        if markdown_text:
            full_markdown_parts.append(markdown_text)

        pages.append(
            {
                "model": PPOCR_V6_MODEL_NAME,
                "parser": "pp-ocrv6",
                "page_index": pruned.get("page_index", page_index),
                "pageImage": page_result.get("inputImage") if isinstance(page_result, dict) else None,
                "markdown": {
                    "text": markdown_text,
                    "images": {},
                },
                "ocrLines": lines,
                "prunedResult": pruned,
            }
        )

    return {
        "markdown": "\n\n".join(full_markdown_parts),
        "images": {},
        "layoutParsingResults": pages,
    }


async def run_ocr_request(ocr_request: OCRRequest, raw_input: RawOCRInput) -> dict:
    runtime = await model_runtime_status("paddleocr-vl-1.6")
    if not runtime["ready"]:
        raise HTTPException(status_code=503, detail="PaddleOCR-VL service is not ready. Switch to this model and wait for it to become ready.")

    base64_data, file_type = prepare_service_input(ocr_request, raw_input)

    payload = build_pipeline_payload(ocr_request, base64_data, file_type)

    logger.info("Sending request to Pipeline Service at %s", PADDLE_SERVICE_URL)
    timeout = PADDLE_REQUEST_TIMEOUT if PADDLE_REQUEST_TIMEOUT > 0 else None
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            PADDLE_SERVICE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if resp.status_code != 200:
            logger.warning("Service Error (HTTP %s): %s", resp.status_code, resp.text)
            if resp.status_code == 422:
                logger.warning("Validation Error Details: %s", resp.json())
            raise HTTPException(status_code=resp.status_code, detail=f"Upstream error: {resp.text}")

        return parse_pipeline_response(resp.json())


async def run_ppocrv6_request(ocr_request: OCRRequest, raw_input: RawOCRInput) -> dict:
    runtime = await model_runtime_status("pp-ocrv6")
    if not runtime["ready"]:
        raise HTTPException(status_code=503, detail="PP-OCRv6 service is not ready. Switch to this model and wait for it to become ready.")

    base64_data, file_type = prepare_service_input(ocr_request, raw_input)
    payload = build_ppocr_payload(ocr_request, base64_data, file_type)

    logger.info("Sending request to PP-OCR service at %s", PADDLE_OCR_SERVICE_URL)
    timeout = PADDLE_REQUEST_TIMEOUT if PADDLE_REQUEST_TIMEOUT > 0 else None
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            PADDLE_OCR_SERVICE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if resp.status_code != 200:
            logger.warning("PP-OCR Service Error (HTTP %s): %s", resp.status_code, resp.text)
            if resp.status_code == 422:
                logger.warning("PP-OCR Validation Error Details: %s", resp.json())
            raise HTTPException(status_code=resp.status_code, detail=f"Upstream PP-OCR error: {resp.text}")

        return parse_ppocr_response(resp.json())


def validate_proxy_input_size(raw_input: RawOCRInput) -> int:
    base64_data = normalize_raw_input_to_base64(raw_input)
    if MAX_REQUEST_BYTES > 0 and len(base64_data) > int(MAX_REQUEST_BYTES * 4 / 3) + 1024:
        max_mb = MAX_REQUEST_BYTES / 1024 / 1024
        raise HTTPException(status_code=413, detail=f"OCR input is too large. Max upload size is {max_mb:.0f} MB.")
    return len(base64_data)


@app.post("/api/paddleocr-vl-1.6")
async def proxy_paddleocr_vl(request: Request):
    """Proxy request to PaddleOCR-VL Pipeline Service."""
    try:
        ocr_request, raw_image = await parse_ocr_input(request)
        base64_size = validate_proxy_input_size(raw_image)
        logger.info("Received PaddleOCR-VL request. Base64 input size: %s bytes", base64_size)
        return await run_ocr_request(ocr_request, raw_image)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PaddleOCR-VL Proxy Error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pp-ocrv6")
async def proxy_ppocrv6(request: Request):
    """Proxy request to PP-OCRv6 OCR Pipeline Service."""
    try:
        ocr_request, raw_image = await parse_ocr_input(request)
        base64_size = validate_proxy_input_size(raw_image)
        logger.info("Received PP-OCRv6 request. Base64 input size: %s bytes", base64_size)
        return await run_ppocrv6_request(ocr_request, raw_image)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PP-OCRv6 Proxy Error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server. Target Pipeline: %s", PADDLE_SERVICE_URL)
    uvicorn.run(app, host=PANDOCR_HOST, port=PANDOCR_PORT)
