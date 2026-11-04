"""FastAPI service layer (industrial upgrade).

What changed versus the tutorial version:

* unified error contract (``{"error": {code, message, retryable, request_id}}``);
* authentication uses a constant-time comparison; ``/health`` and ``/ready``
  stay unauthenticated for probes;
* per-identity sliding-window rate limiting with ``Retry-After``;
* ``Idempotency-Key`` on ``POST /tasks`` maps a client retry to the same task;
* request/task/thread ids bound into structured logs and every trace event;
* checkpointed graph composition, ``/resume`` for interrupted tasks and
  ``/approve`` that survives a restart;
* graceful shutdown waits for in-flight tasks and closes SQLite handles.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agents.agent_graph import build_agent_graph
from app.core.budget import ExecutionBudget
from app.core.config import Settings, get_settings
from app.core.context import scope
from app.core.errors import ErrorCode, NotFoundError, TaskPilotError, ValidationError
from app.core.logging_config import configure_logging
from app.core.metrics import METRICS
from app.core.observability import log_event
from app.core.ratelimit import SlidingWindowLimiter
from app.core.timeouts import shutdown_pool
from app.schemas.tasks import ApprovalRequest, MemoryWriteRequest, TaskRequest, TaskResponse
from app.services.checkpoint import CheckpointManager, pending_nodes, thread_config
from app.services.idempotency import IdempotencyStore
from app.services.llm_service import AdaptiveLLMService
from app.services.memory import MemoryStore
from app.services.repository import TaskRepository
from app.tools.registry import build_default_registry

logger = logging.getLogger(__name__)
VERSION = "3.1.0-platform"

_PUBLIC_PATHS = {"/health", "/ready", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


class InflightTracker:
    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._count = 0
        self._shutting_down = False

    def enter(self) -> bool:
        with self._lock:
            if self._shutting_down:
                return False
            self._count += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._count = max(0, self._count - 1)
            if self._count == 0:
                self._lock.notify_all()

    def begin_shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True

    def wait(self, timeout: float) -> bool:
        deadline = time.time() + max(0.0, timeout)
        with self._lock:
            while self._count > 0:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._lock.wait(remaining)
            return True

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


class AppContext:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_directories()
        self.repository = TaskRepository(settings.taskpilot_db_path)
        self.idempotency = IdempotencyStore(settings.taskpilot_db_path, ttl_hours=settings.idempotency_ttl_hours)
        self.memory = MemoryStore(
            settings.taskpilot_db_path,
            max_items=settings.memory_max_items,
            ttl_days=settings.memory_ttl_days,
        )
        self.registry, self.breakers = build_default_registry(settings, self.idempotency)
        self.llm = AdaptiveLLMService(settings=settings)
        self.checkpoints = CheckpointManager(settings.checkpoint_db_path)
        self.graph = build_agent_graph(
            repository=self.repository,
            llm=self.llm,
            settings=settings,
            registry=self.registry,
            memory=self.memory,
            checkpointer=self.checkpoints.saver,
        )
        self.limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)
        self.inflight = InflightTracker()
        self.recovered: list[dict[str, Any]] = []

    def close(self) -> None:
        self.inflight.begin_shutdown()
        self.inflight.wait(self.settings.shutdown_grace_seconds)
        self.checkpoints.close()
        shutdown_pool(wait=False)


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    configure_logging(cfg.log_level, cfg.log_json)
    app = FastAPI(
        title=cfg.app_name,
        version=VERSION,
        description="TaskPilot Agent Platform — bounded ReAct, classified retry, circuit breaking, idempotent tools, checkpointed recovery",
    )
    holder: dict[str, AppContext | None] = {"ctx": None}
    init_lock = threading.Lock()

    def get_ctx() -> AppContext:
        with init_lock:
            if holder["ctx"] is None:
                holder["ctx"] = AppContext(cfg)
            return holder["ctx"]  # type: ignore[return-value]

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ctx = get_ctx()
        app.state.ctx = ctx
        if cfg.recover_on_startup:
            ctx.recovered = ctx.repository.mark_stale_running(cfg.stale_task_seconds)
            for state in ctx.recovered:
                log_event(logger, "task_marked_interrupted", task_id=state.get("task_id"))
        log_event(logger, "app_ready", version=VERSION, mode="mock" if cfg.mock_llm else "llm")
        try:
            yield
        finally:
            ctx.close()
            log_event(logger, "app_stopped")

    app.router.lifespan_context = lifespan
    app.state.get_ctx = get_ctx

    # ------------------------------------------------------------------
    # middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def request_pipeline(request: Request, call_next: Callable[[Request], Any]):
        ctx = get_ctx()
        if not ctx.inflight.enter():
            return JSONResponse(status_code=503, content={"error": {"code": "SHUTTING_DOWN", "message": "服务正在关闭"}})
        started = time.perf_counter()
        request_id = (request.headers.get("X-Request-ID") or uuid.uuid4().hex)[:64]
        path = request.url.path
        try:
            with scope(request_id=request_id):
                if path not in _PUBLIC_PATHS:
                    auth_error = _check_auth(request, cfg)
                    if auth_error is not None:
                        return auth_error
                    allowed, retry_after = ctx.limiter.allow(_identity(request, cfg))
                    if not allowed:
                        METRICS.inc("http_requests_total", path=path, method=request.method, status=429)
                        return JSONResponse(
                            status_code=429,
                            headers={"Retry-After": f"{retry_after:.1f}", "X-Request-ID": request_id},
                            content={"error": {"code": "RATE_LIMITED", "message": "请求过于频繁", "retryable": True, "retry_after_seconds": round(retry_after, 3), "request_id": request_id}},
                        )
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                elapsed = (time.perf_counter() - started) * 1000
                METRICS.inc("http_requests_total", path=path, method=request.method, status=response.status_code)
                METRICS.observe("http_request_latency_ms", elapsed, path=path)
                return response
        finally:
            ctx.inflight.leave()

    # ------------------------------------------------------------------
    # error contract
    # ------------------------------------------------------------------
    @app.exception_handler(TaskPilotError)
    async def taskpilot_error_handler(request: Request, exc: TaskPilotError):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:64]
        log_event(logger, "api_error", level=logging.WARNING, code=exc.code.value, message=exc.message, path=request.url.path)
        headers = {"X-Request-ID": request_id}
        if exc.retry_after is not None:
            headers["Retry-After"] = f"{exc.retry_after:.1f}"
        return JSONResponse(status_code=exc.status_code, headers=headers, content={"error": exc.to_dict(request_id)})

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:64]
        errors = [{"loc": list(err.get("loc", [])), "msg": err.get("msg", ""), "type": err.get("type", "")} for err in exc.errors()[:5]]
        return JSONResponse(
            status_code=422,
            headers={"X-Request-ID": request_id},
            content={"error": {"code": ErrorCode.VALIDATION_ERROR.value, "message": "请求参数校验失败", "retryable": False, "request_id": request_id, "details": {"errors": errors}}},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:64]
        code = {400: ErrorCode.VALIDATION_ERROR, 401: ErrorCode.UNAUTHORIZED, 403: ErrorCode.FORBIDDEN, 404: ErrorCode.NOT_FOUND, 409: ErrorCode.CONFLICT, 413: ErrorCode.VALIDATION_ERROR, 429: ErrorCode.RATE_LIMITED}.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        return JSONResponse(
            status_code=exc.status_code,
            headers={"X-Request-ID": request_id, **dict(getattr(exc, "headers", None) or {})},
            content={"error": {"code": code.value, "message": str(exc.detail), "retryable": exc.status_code >= 500, "request_id": request_id}},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:64]
        logger.exception("unhandled_request_error request_id=%s path=%s", request_id, request.url.path)
        return JSONResponse(
            status_code=500,
            headers={"X-Request-ID": request_id},
            content={"error": {"code": ErrorCode.INTERNAL_ERROR.value, "message": "Internal server error", "retryable": False, "request_id": request_id}},
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _check_auth(request: Request, config: Settings) -> JSONResponse | None:
        if not config.api_access_token:
            return None
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied, config.api_access_token):
            request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:64]
            return JSONResponse(
                status_code=401,
                headers={"X-Request-ID": request_id},
                content={"error": {"code": ErrorCode.UNAUTHORIZED.value, "message": "API key missing or invalid", "retryable": False, "request_id": request_id}},
            )
        return None

    @staticmethod
    def _identity(request: Request, config: Settings) -> str:
        if config.api_access_token:
            return "token:" + (request.headers.get("X-API-Key", "")[:8])
        return "ip:" + (request.client.host if request.client else "unknown")

    def response_for(state: dict[str, Any], ctx: AppContext) -> TaskResponse:
        return TaskResponse(
            task_id=state["task_id"],
            status=state.get("status", "running"),
            task_type=state.get("task_type"),
            goal=state.get("goal", ""),
            plan=state.get("plan"),
            answer=state.get("answer"),
            trace=ctx.repository.get_trace(state["task_id"]) if state.get("task_id") else [],
            metrics={
                **(state.get("metrics") or {}),
                "llm": ctx.llm.metrics(),
                "idempotency": ctx.idempotency.stats(),
                "circuits": ctx.breakers.snapshot(),
            },
            message={
                "pending_approval": "等待人工确认，尚未执行外部操作。",
                "completed": "任务已完成。",
                "partial": "任务完成但存在局限或降级信息，请核验结果。",
                "interrupted": "任务因服务重启中断，可调用 /tasks/{task_id}/resume 继续。",
                "cancelled": "任务已被用户拒绝执行。",
                "failed": "任务执行失败，请检查 trace 与服务日志。",
            }.get(state.get("status", "running"), "任务处理中。"),
            error_code=state.get("error_code"),
            stop_reason=state.get("stop_reason"),
            observations=state.get("observations") or [],
            budget=state.get("budget") or {},
        )

    def execute_task(state: dict[str, Any], ctx: AppContext) -> dict[str, Any]:
        config = thread_config(state["task_id"])
        try:
            result = ctx.graph.invoke(state, config)
            final = {**state, **result}
            ctx.repository.save_state(final)
            return final
        except TaskPilotError as exc:
            failed = {**state, "status": "failed", "error_code": exc.code.value, "error": f"{exc.code.value}: {exc.message}"}
            ctx.repository.save_state(failed)
            ctx.repository.update_status(state["task_id"], "failed", error_code=exc.code.value)
            ctx.repository.append_trace(state["task_id"], "api_error", {"code": exc.code.value, "message": exc.message})
            raise
        except Exception as exc:
            logger.exception("task_failed task_id=%s", state.get("task_id"))
            failed = {**state, "status": "failed", "error_code": ErrorCode.INTERNAL_ERROR.value, "error": f"{type(exc).__name__}: {exc}"}
            ctx.repository.save_state(failed)
            ctx.repository.update_status(state["task_id"], "failed", error_code=ErrorCode.INTERNAL_ERROR.value)
            ctx.repository.append_trace(state["task_id"], "api_error", {"code": ErrorCode.INTERNAL_ERROR.value, "type": type(exc).__name__})
            raise TaskPilotError(
                f"任务执行失败（{type(exc).__name__}）。请检查 /tasks/{state['task_id']}/trace。",
                code=ErrorCode.INTERNAL_ERROR,
                details={"task_id": state["task_id"]},
            ) from exc

    # ------------------------------------------------------------------
    # endpoints
    # ------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, Any]:
        ctx = get_ctx()
        return {"status": "ok", "app": cfg.app_name, "version": VERSION, "mode": "mock" if cfg.mock_llm else "llm", "inflight": ctx.inflight.count}

    @app.get("/ready")
    def ready() -> JSONResponse:
        ctx = get_ctx()
        checks: dict[str, Any] = {}
        ok = True
        try:
            ctx.repository.get_row("__ready_probe__")
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {type(exc).__name__}"
            ok = False
        try:
            ctx.graph.get_state(thread_config("__ready_probe__"))
            checks["checkpointer"] = "ok"
        except Exception as exc:
            checks["checkpointer"] = f"error: {type(exc).__name__}"
            ok = False
        checks["tools"] = ctx.registry.names()
        checks["tool_count"] = len(checks["tools"])
        if not checks["tools"]:
            ok = False
        body = {"status": "ready" if ok else "not_ready", "checks": checks, "recovered_tasks": len(ctx.recovered)}
        return JSONResponse(status_code=200 if ok else 503, content=body)

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        ctx = get_ctx()
        return {
            "service": {"version": VERSION, "mode": "mock" if cfg.mock_llm else "llm", "inflight": ctx.inflight.count},
            "http": METRICS.snapshot(),
            "llm": ctx.llm.metrics(),
            "tools": {"registered": ctx.registry.names()},
            "circuits": ctx.breakers.snapshot(),
            "idempotency": ctx.idempotency.stats(),
            "memory": ctx.memory.stats(),
            "tasks": ctx.repository.count_by_status(),
        }

    @app.post("/files/upload")
    async def upload_csv(file: UploadFile = File(...)) -> dict[str, Any]:
        filename = Path(file.filename or "").name
        if not filename or not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="演示版只允许上传 CSV 文件。")
        size = 0
        chunks: list[bytes] = []
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > 10 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="CSV 文件不得超过 10 MiB。")
            chunks.append(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="上传文件为空。")
        safe_name = f"{uuid.uuid4().hex[:10]}_{filename}"
        destination = cfg.upload_dir / safe_name
        destination.write_bytes(b"".join(chunks))
        return {"filename": safe_name, "bytes": size, "message": "文件已保存到受控上传目录。"}

    @app.post("/tasks", response_model=TaskResponse)
    def create_task(request: TaskRequest, http_request: Request) -> TaskResponse:
        ctx = get_ctx()
        request_key = (http_request.headers.get("Idempotency-Key") or "").strip()[:120]
        if request_key:
            existing_id = ctx.repository.find_by_request_key(request_key, user_id=request.user_id)
            if existing_id:
                existing = ctx.repository.get_state(existing_id)
                if existing:
                    METRICS.inc("task_request_replays_total")
                    return response_for(existing, ctx)

        task_id = uuid.uuid4().hex
        budget = ExecutionBudget.create(cfg.budget_limits())
        initial_state: dict[str, Any] = {
            "task_id": task_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "goal": request.goal,
            "task_type": request.task_type,
            "input_file": request.input_file,
            "approved": request.approved,
            "status": "running",
            "plan": None,
            "context": {},
            "tool_result": {},
            "observations": [],
            "react_step": 0,
            "next_action": {},
            "budget": budget.to_dict(),
            "trace": [],
            "metrics": {},
            "degraded_reasons": [],
            "request_id": http_request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16],
        }
        ctx.repository.save_state(initial_state)
        if request_key and not ctx.repository.register_request_key(request_key, task_id, user_id=request.user_id):
            existing_id = ctx.repository.find_by_request_key(request_key, user_id=request.user_id)
            existing = ctx.repository.get_state(existing_id) if existing_id else None
            if existing:
                return response_for(existing, ctx)

        with scope(task_id=task_id, thread_id=task_id):
            final = execute_task(initial_state, ctx)
        return response_for(final, ctx)

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str, user_id: str = "local-demo") -> TaskResponse:
        ctx = get_ctx()
        state = ctx.repository.get_state(task_id)
        if not state or state.get("user_id") != user_id:
            raise NotFoundError("任务不存在。")
        return response_for(state, ctx)

    @app.get("/tasks/{task_id}/trace")
    def get_task_trace(task_id: str, user_id: str = "local-demo") -> dict[str, Any]:
        ctx = get_ctx()
        state = ctx.repository.get_state(task_id)
        if not state or state.get("user_id") != user_id:
            raise NotFoundError("任务不存在。")
        return {
            "task_id": task_id,
            "status": state.get("status"),
            "stop_reason": state.get("stop_reason"),
            "budget": state.get("budget"),
            "observations": state.get("observations", []),
            "trace": ctx.repository.get_trace(task_id),
        }

    @app.post("/tasks/{task_id}/approve", response_model=TaskResponse)
    def approve_task(task_id: str, request: ApprovalRequest) -> TaskResponse:
        ctx = get_ctx()
        state = ctx.repository.get_state(task_id)
        if not state or state.get("user_id") != request.user_id:
            raise NotFoundError("任务不存在。")
        if state.get("status") != "pending_approval":
            raise TaskPilotError(
                f"当前任务状态为 {state.get('status')}，不需要审批或已无法审批。",
                code=ErrorCode.CONFLICT,
            )

        if not request.approved:
            answer = {
                "title": "操作已拒绝",
                "summary": "用户拒绝了本次操作。系统没有执行外部副作用。",
                "reason": request.reason or "用户选择拒绝。",
                "external_side_effect_performed": False,
                "next_steps": ["如需继续，请创建一个新的任务并明确安全范围。"],
            }
            moved = ctx.repository.transition_status(
                task_id,
                "pending_approval",
                "cancelled",
                patch={"approved": False, "answer": answer, "approval_reason": request.reason, "stop_reason": "approval_rejected"},
            )
            if not moved:
                raise TaskPilotError("该审批已被其他请求处理，请刷新任务状态。", code=ErrorCode.CONFLICT)
            ctx.repository.append_trace(task_id, "approval_decision", {"decision": "rejected", "reason": request.reason})
            return response_for(ctx.repository.get_state(task_id) or state, ctx)

        moved = ctx.repository.transition_status(
            task_id,
            "pending_approval",
            "running",
            patch={"approved": True, "approval_reason": request.reason or "用户已明确批准。", "answer": None, "stop_reason": None},
        )
        if not moved:
            raise TaskPilotError("该审批已被其他请求处理，请刷新任务状态。", code=ErrorCode.CONFLICT)
        state = ctx.repository.get_state(task_id) or state
        ctx.repository.append_trace(task_id, "approval_decision", {"decision": "approved", "reason": request.reason})
        with scope(task_id=task_id, thread_id=task_id):
            final = execute_task(state, ctx)
        return response_for(final, ctx)

    @app.post("/tasks/{task_id}/resume", response_model=TaskResponse)
    def resume_task(task_id: str, user_id: str = "local-demo") -> TaskResponse:
        ctx = get_ctx()
        state = ctx.repository.get_state(task_id)
        if not state or state.get("user_id") != user_id:
            raise NotFoundError("任务不存在。")
        previous_status = state.get("status")
        if previous_status not in {"interrupted", "failed"}:
            raise TaskPilotError(
                f"任务状态为 {previous_status}，无需恢复；运行中的任务需先由 reaper 标记为 interrupted。",
                code=ErrorCode.CONFLICT,
            )
        if not ctx.repository.claim_for_resume(task_id, {"interrupted", "failed"}):
            raise TaskPilotError(
                "该任务已被另一个恢复请求领取，请刷新任务状态。",
                code=ErrorCode.CONFLICT,
            )
        state = ctx.repository.get_state(task_id) or state
        ctx.repository.append_trace(task_id, "resume_requested", {"from_status": previous_status})
        config = thread_config(task_id)
        with scope(task_id=task_id, thread_id=task_id):
            try:
                if pending_nodes(ctx.graph, config):
                    result = ctx.graph.invoke(None, config)
                else:
                    result = ctx.graph.invoke(state, config)
                final = {**state, **result}
                ctx.repository.save_state(final)
                METRICS.inc("task_recoveries_total", outcome="success")
                return response_for(final, ctx)
            except Exception as exc:
                METRICS.inc("task_recoveries_total", outcome="failure")
                ctx.repository.append_trace(task_id, "resume_failed", {"error_type": type(exc).__name__})
                raise TaskPilotError(
                    f"恢复失败：{type(exc).__name__}",
                    code=ErrorCode.INTERNAL_ERROR,
                    details={"task_id": task_id},
                ) from exc

    @app.get("/tasks")
    def list_tasks(limit: int = 30, user_id: str = "local-demo") -> dict[str, Any]:
        ctx = get_ctx()
        return {"items": ctx.repository.list_tasks(limit=max(1, min(limit, 100)), user_id=user_id)}

    @app.post("/memory")
    def write_memory(request: MemoryWriteRequest) -> dict[str, Any]:
        ctx = get_ctx()
        memory_id = ctx.memory.remember(
            user_id=request.user_id,
            session_id=request.session_id,
            content=request.content,
            kind=request.kind,
            tags=request.tags,
            importance=request.importance,
        )
        if memory_id is None:
            raise ValidationError("记忆内容无效或已被去重。")
        return {"id": memory_id, "kind": request.kind, "stored": True}

    @app.get("/memory")
    def read_memory(user_id: str = "local-demo", query: str = "", limit: int = 5) -> dict[str, Any]:
        ctx = get_ctx()
        items = ctx.memory.recall(user_id=user_id, query=query, limit=max(1, min(limit, 20)), max_chars=2000)
        return {"items": items}

    @app.delete("/memory")
    def delete_memory(user_id: str = "local-demo", kind: str | None = None) -> dict[str, Any]:
        if kind is not None and kind not in {"short_term", "long_term"}:
            raise ValidationError("kind 只能是 short_term 或 long_term")
        ctx = get_ctx()
        return {"deleted": ctx.memory.forget(user_id=user_id, kind=kind)}

    return app


app = create_app()