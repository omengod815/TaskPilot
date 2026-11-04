"""Controlled demo tools.

Design rules kept from the original project:

* no ``eval`` — arithmetic uses an AST whitelist;
* file access is confined to ``data/public`` and the upload directory;
* the "external action" tool never sends email / pays / deletes anything; it
  writes a local plan file.  It is still treated as a *side-effect* tool so the
  idempotency and approval machinery is exercised end to end.

New in the industrial version:

* every return value carries ``trust="untrusted"`` metadata where it is
  derived from external content (web pages, CSV cells, user text), because
  tool output is data, never instructions;
* the demo action is idempotent by construction: the file name is derived
  from the idempotency key, and ``lookup_demo_action`` lets a recovering
  worker discover the already-applied effect instead of replaying it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.core.errors import ToolError, ToolUnavailableError

settings = get_settings()


def _safe_data_path(filename: str) -> Path:
    """Resolve a filename only inside approved demo folders; block traversal."""

    if not filename or Path(filename).name != filename or ".." in filename:
        raise ToolError("文件名不合法：只允许传入文件名，不允许路径或 '..'。")
    candidates = [Path("data/public") / filename, settings.upload_dir / filename]
    for candidate in candidates:
        resolved = candidate.resolve()
        allowed_roots = [Path("data/public").resolve(), settings.upload_dir.resolve()]
        if any(resolved.is_relative_to(root) for root in allowed_roots) and resolved.is_file():
            return resolved
    raise ToolError(f"找不到文件 {filename}。请先上传文件，或使用 data/public 中的示例文件。")


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}
_ALLOWED_UNARY = {ast.UAdd: lambda x: x, ast.USub: lambda x: -x}


def safe_calculate(expression: str) -> dict[str, Any]:
    if len(expression) > 120:
        raise ToolError("表达式过长，限制为 120 个字符。")
    try:
        root = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise ToolError("无法解析数学表达式。") from exc

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ToolError("为避免资源消耗，幂运算指数绝对值不能超过 8。")
            result = _ALLOWED_BINOPS[type(node.op)](left, right)
            if not math.isfinite(result):
                raise ToolError("计算结果不是有限数值。")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](evaluate(node.operand))
        raise ToolError("只允许数字与 + - * / ** % 和括号，不允许变量、函数或代码执行。")

    result = evaluate(root)
    return {"expression": expression, "result": result, "trust": "untrusted"}


def analyze_csv(filename: str) -> dict[str, Any]:
    path = _safe_data_path(filename)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ToolError(f"CSV 读取失败：{exc}") from exc
    if df.empty:
        raise ToolError("CSV 文件没有数据行。")
    if len(df) > 200_000:
        raise ToolError("演示版限制单次分析不超过 200000 行。")

    numeric = df.select_dtypes(include="number")
    numeric_summary: dict[str, dict[str, float]] = {}
    for col in numeric.columns[:20]:
        series = numeric[col].dropna()
        if len(series):
            numeric_summary[str(col)] = {
                "mean": round(float(series.mean()), 6),
                "min": round(float(series.min()), 6),
                "max": round(float(series.max()), 6),
                "std": round(float(series.std(ddof=0)), 6),
            }

    metrics: dict[str, float | int | str] = {}
    actual_col = next((c for c in df.columns if str(c).lower() in {"actual", "y_true", "真实值", "实际值"}), None)
    pred_col = next((c for c in df.columns if str(c).lower() in {"predicted", "prediction", "y_pred", "预测值"}), None)
    if actual_col and pred_col:
        pairs = df[[actual_col, pred_col]].dropna()
        if len(pairs):
            actual = pairs[actual_col].astype(float)
            pred = pairs[pred_col].astype(float)
            errors = actual - pred
            mae = float(errors.abs().mean())
            rmse = float(math.sqrt((errors ** 2).mean()))
            denominator = float(((actual - actual.mean()) ** 2).sum())
            r2 = 1.0 - float((errors ** 2).sum()) / denominator if denominator else 0.0
            metrics = {"MAE": round(mae, 6), "RMSE": round(rmse, 6), "R2": round(r2, 6), "metric_rows": int(len(pairs))}

    return {
        "filename": path.name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "numeric_summary": numeric_summary,
        "prediction_metrics": metrics,
        "preview": df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records"),
        "trust": "untrusted",
    }


def _tokens(text: str) -> set[str]:
    latin = set(re.findall(r"[a-zA-Z0-9_+-]{2,}", text.lower()))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    cjk = set()
    for run in cjk_runs:
        cjk.add(run)
        cjk.update(run[i:i + 2] for i in range(len(run) - 1))
    return latin | cjk


def search_local_knowledge(query: str, top_k: int = 4) -> dict[str, Any]:
    root = Path("data/knowledge_base")
    query_tokens = _tokens(query)
    results = []
    for path in root.glob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for index, paragraph in enumerate(paragraphs):
            tokens = _tokens(paragraph)
            overlap = len(query_tokens & tokens)
            if overlap:
                results.append({
                    "title": f"{path.stem} · 段落 {index + 1}",
                    "source": str(path),
                    "excerpt": paragraph[:900],
                    "score": round(overlap / max(1, len(query_tokens)), 4),
                    "trust": "untrusted",
                })
    results.sort(key=lambda item: item["score"], reverse=True)
    return {"query": query, "results": results[:max(1, min(top_k, 10))], "mode": "local_keyword_search", "trust": "untrusted"}


def search_web(query: str, top_k: int = 5) -> dict[str, Any]:
    """Online search when configured; raises on failure so callers can degrade."""

    if not settings.tavily_api_key:
        local = search_local_knowledge(query, top_k=top_k)
        local["mode"] = "local_fallback_no_tavily_key"
        local["degraded"] = True
        local["note"] = "未配置 TAVILY_API_KEY，已回退到本地知识库检索；这不是实时互联网搜索。"
        return local
    import httpx

    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": settings.tavily_api_key, "query": query, "max_results": max(1, min(top_k, 10)), "search_depth": "basic"},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # network / HTTP / JSON
        raise ToolUnavailableError(f"在线检索不可用：{type(exc).__name__}") from exc
    return {
        "query": query,
        "mode": "tavily_web_search",
        "results": [
            {"title": item.get("title", "Untitled"), "source": item.get("url", ""), "url": item.get("url", ""), "excerpt": item.get("content", "")[:900], "score": item.get("score", 0), "trust": "untrusted"}
            for item in data.get("results", [])
        ],
        "trust": "untrusted",
    }


def _action_filename(goal: str, idempotency_key: str | None) -> str:
    if idempotency_key:
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:16]
        return f"action_plan_{digest}.json"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", goal[:50]).strip("_") or "action"
    return f"action_plan_{safe_name}.txt"


def lookup_demo_action(idempotency_key: str, output_dir: Path | None = None) -> dict[str, Any] | None:
    """Reconciliation: did a previous attempt already apply the effect?"""

    out_dir = output_dir or settings.output_dir
    path = out_dir / _action_filename("", idempotency_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    payload["reconciled"] = True
    return payload


def execute_demo_action(goal: str, *, idempotency_key: str | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    """Demonstration-only action; never sends/publishes/deletes externally.

    The effect is a local JSON file whose name is derived from the
    idempotency key, written atomically (temp file + ``os.replace``).  A replay
    with the same key returns the existing effect instead of creating a second
    one; that is what makes the recovery test meaningful.
    """

    out_dir = output_dir or settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _action_filename(goal, idempotency_key)
    payload = {
        "prepared": True,
        "goal": goal,
        "file": str(path),
        "external_side_effect_performed": False,
        "idempotency_key": idempotency_key,
        "applied_at": time.time(),
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["replayed"] = True
        return payload
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    return payload