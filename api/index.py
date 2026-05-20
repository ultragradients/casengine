from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sympy as sp

from matlang import Vector2, Vector3, Quaternion, Matrix, Func, Lim, x

# ---------------------------------------------------------------------------
# Per-session variable store (module-level; fine for single-user / dev use)
# ---------------------------------------------------------------------------
variables: dict = {}
mode = "prod"

# FIX 1: added `global mode` so the function can reassign the module-level variable
def switchmode():
    global mode
    if mode == "prod":
        mode = "test"
    else:
        mode = "prod"


import secrets
import hashlib
import json
from pathlib import Path
from fastapi import Header, HTTPException, Depends, Query

_STORE_PATH = Path("matlang_api_store.json")
_store = {"users": {}, "api_keys": {}}
if _STORE_PATH.exists():
    try:
        _store = json.loads(_STORE_PATH.read_text())
    except Exception:
        _store = {"users": {}, "api_keys": {}}

_sessions = {}
apiKeys = _store.get("api_keys", {})

def _save_store():
    try:
        _STORE_PATH.write_text(json.dumps(_store, indent=2))
    except Exception:
        pass

def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="MatLang CAS",
    version="1.0.0-beta-2",
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    with open("static/swagger_ui.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)

app.title = "CASe"

@app.get("/")
def read_root():
    return FileResponse("static/index.html", media_type="text/html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

v1beta1 = APIRouter(prefix="/v1beta1", tags=["v1beta1"])
v1beta2 = APIRouter(prefix="/v1beta2", tags=["v1beta2"])
v1 = APIRouter(prefix="/v1", tags=["v1", "Stable Release"])

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class Command(BaseModel):
    code: str = Field(
        ...,
        example="Func(x**2 - 1).plot()",
        description="Mathematical expression or MatLang command to evaluate.",
    )


# ---------------------------------------------------------------------------
# Allowed globals
# ---------------------------------------------------------------------------
def _build_globals() -> dict:
    g: dict = {
        "Vector2": Vector2, "vector2": Vector2, "vec2": Vector2, "v2d": Vector2,
        "Vector3": Vector3, "vector3": Vector3, "vec3": Vector3, "v3d": Vector3,
        "Quaternion": Quaternion,
        "Matrix": Matrix, "matrix": Matrix,
        "Func": Func, "Function": Func, "function": Func, "func": Func,
        "Lim": Lim, "lim": Lim,

        "x": x,
        "var": sp.Symbol,
        "symbols": sp.symbols,

        "pi": sp.pi,
        "e": sp.E,
        "oo": sp.oo,
        "inf": sp.oo,

        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,

        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "atan2": sp.atan2,
        "asec": sp.asec, "acsc": sp.acsc, "acot": sp.acot,

        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "sech": sp.sech, "csch": sp.csch, "coth": sp.coth,

        "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,

        "exp": sp.exp,
        "log": sp.log, "ln": sp.log,
        "log10": lambda v: sp.log(v, 10),
        "log2":  lambda v: sp.log(v, 2),

        "sqrt": sp.sqrt,
        # FIX 2: cbrt was calling __rpow__ on Rational instead of on v
        "cbrt": lambda v: v ** sp.Rational(1, 3),
        "root": lambda v, n: v ** sp.Rational(1, n),

        "abs": sp.Abs,
        "floor": sp.floor,
        "ceiling": sp.ceiling,
        "sign": sp.sign,
        "float": float,
        "simplify": sp.simplify,
        "expand": sp.expand,
        "factor": sp.factor,
        "cancel": sp.cancel,
        "apart": sp.apart,
        "together": sp.together,
        "solve": sp.solve,
        "diff": sp.diff,
        "integrate": sp.integrate,
        "limit": sp.limit,
        "series": sp.series,
        "summation": sp.summation,
        # FIX 3: sp.evalf() is a method on expressions, not a standalone callable.
        # Expose it as a helper that calls .evalf() on a sympy expression.
        "exact": lambda expr: expr.evalf() if hasattr(expr, "evalf") else expr,

        "factorial": lambda v: sp.factorial(v),
        "gamma": sp.gamma,
        "binomial": sp.binomial,
        "gcd": sp.gcd,
        "lcm": sp.lcm,
    }
    return g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_assignment(code: str) -> bool:
    import re
    return bool(re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=(?!=)", code))


def _validate_api_key(key: str):
    if not key:
        raise HTTPException(status_code=401, detail="Missing apiKey")
    info = _store.get("api_keys", {}).get(key)
    if not info:
        raise HTTPException(status_code=401, detail="Invalid apiKey")
    info.setdefault("variables", {})
    return info


def _render_plot(func_obj, x_range=(-10, 10), points=600) -> dict:
    x_vals = np.linspace(x_range[0], x_range[1], points)
    y_vals = []
    for v in x_vals:
        try:
            y = func_obj(v)
            y = complex(y).real
            y_vals.append(float(y) if np.isfinite(y) else np.nan)
        except Exception:
            y_vals.append(np.nan)
    y_vals = np.array(y_vals)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x_vals, y_vals, color="#55ccff", linewidth=1.8)
    ax.axhline(0, color="#555", linewidth=0.6)
    ax.axvline(0, color="#555", linewidth=0.6)

    finite = y_vals[np.isfinite(y_vals)]
    if finite.size > 0:
        lo, hi = np.percentile(finite, 1), np.percentile(finite, 99)
        pad = max((hi - lo) * 0.15, 0.5)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.grid(True, color="#2a2a2a", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.patch.set_facecolor("#111")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#111", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return {"type": "plot", "image": base64.b64encode(buf.read()).decode("utf-8")}


def _safe_str(value) -> str:
    if isinstance(value, sp.Basic):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Health check (overrides the root FileResponse above — kept as a named route)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "message": "MatLang CAS API is running.", "version": "1.0.0-beta-2"}


# ---------------------------------------------------------------------------
# v1beta1 routes
# FIX 4: renamed all handler functions to be unique (avoids silent overwriting)
# ---------------------------------------------------------------------------

@v1beta1.post(
    "/eval",
    summary="Evaluate a MatLang expression (requires apiKey)",
    description=(
        "Executes a MatLang / SymPy expression and returns the result as text or a base64 PNG plot. "
        "Use `Func(expr).plot()` to generate a plot. "
        "Assignments (`name = expr`) persist per-API-key.\n\n"
        "Supply your API key as a query parameter: `?apiKey=YOUR_KEY`, or `X-Api-Key` header."
    ),
)
def v1beta1_evaluate(
    command: Command,
    apiKey: str = Query(..., description="API key for accessing the eval endpoint", alias="apiKey"),
):
    code = command.code.strip()
    if not code:
        return {"type": "error", "result": "Empty input."}

    key_info = _validate_api_key(apiKey)
    allowed = _build_globals()
    allowed.update(key_info.get("variables", {}))

    try:
        if code.endswith(".plot()"):
            func_code = code[: -len(".plot()")]
            func_obj = eval(func_code, {"__builtins__": {}}, allowed)
            if not isinstance(func_obj, Func):
                return {"type": "error", "result": "`.plot()` can only be called on a Func object."}
            return _render_plot(func_obj)

        if _is_assignment(code):
            var_name, _, expr_str = code.partition("=")
            var_name = var_name.strip()
            expr_str = expr_str.strip()
            value = eval(expr_str, {"__builtins__": {}}, allowed)
            key_info["variables"][var_name] = value
            _save_store()
            return {"type": "text", "result": f"{var_name} = {_safe_str(value)}"}

        result = eval(code, {"__builtins__": {}}, allowed)

        if isinstance(result, Lim):
            result = result.evaluate()

        return {"type": "text", "result": _safe_str(result)}

    except ZeroDivisionError as e:
        return {"type": "error", "result": f"Math error: {e}"}
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        return {"type": "error", "result": f"Error: {e}"}
    except Exception as e:
        return {"type": "error", "result": f"Unexpected error: {e}"}


@v1beta1.post(
    "/reset",
    summary="Clear session variables for API key",
    description="Clears variables stored for the provided API key.",
)
def v1beta1_reset(
    apiKey: str = Query(..., description="API key for which to clear variables", alias="apiKey"),
):
    key_info = _validate_api_key(apiKey)
    key_info["variables"] = {}
    _save_store()
    return {"type": "text", "result": "Session variables cleared for this API key."}


@v1beta1.get(
    "/vars",
    summary="List session variables for API key",
    description="Returns variables stored for the provided API key.",
)
def v1beta1_list_vars(
    apiKey: str = Query(..., description="API key to list variables for", alias="apiKey"),
):
    key_info = _validate_api_key(apiKey)
    return {"type": "vars", "result": {k: _safe_str(v) for k, v in key_info.get("variables", {}).items()}}


# ---------------------------------------------------------------------------
# v1beta2 routes
# FIX 5: fixed sp.parse_expr call (it takes a string + local_dict kwarg, not positional args)
# FIX 6: fixed tags from "v1beta1" -> "v1beta2" on reset and vars
# FIX 7: LaTeX output: every text result includes a latex field
# FIX 8: Namespaced vars: GET/DELETE /v1beta2/vars/{name} for fine-grained control
# FIX 9: Enhanced plots: configurable x_range, points, multi-function overlay
# FIX 10: Rate limiting       — sliding-window per API key (60 req/min default)
# FIX 11: Response metadata   — execution_time_ms, sympy_version, result_type
# FIX 12: Expression validation — pre-eval syntax check with actionable messages
# FIX 13: Batch eval          — POST /v1beta2/eval/batch, up to 10 expressions

# ---------------------------------------------------------------------------
"""
v1beta2 — upgraded router
Drop-in replacement for the v1beta2 APIRouter block in main.py.

New features vs v1beta1:
  1. Rate limiting       — sliding-window per API key (60 req/min default)
  2. Response metadata   — execution_time_ms, sympy_version, result_type
  3. Expression validation — pre-eval syntax check with actionable messages
  4. Batch eval          — POST /v1beta2/eval/batch, up to 10 expressions
  5. LaTeX output        — every text result includes a latex field
  6. Namespaced vars     — GET/DELETE /v1beta2/vars/{name} for fine-grained control
  7. Enhanced plots      — configurable x_range, points, multi-function overlay
"""

import time
import ast
import sympy as sp
from collections import deque
from typing import Optional, List, Any

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

# ── re-use helpers from the parent module ──────────────────────────────────
# (imported when this file is exec'd / included in main.py)


# ── rate-limit store ────────────────────────────────────────────────────────
_rate_windows: dict[str, deque] = {}
RATE_LIMIT = 60          # requests
RATE_WINDOW = 60.0       # seconds


def _check_rate_limit(api_key: str) -> None:
    now = time.monotonic()
    window = _rate_windows.setdefault(api_key, deque())
    # purge expired timestamps
    while window and now - window[0] > RATE_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        retry_after = int(RATE_WINDOW - (now - window[0])) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT} req/{int(RATE_WINDOW)}s). "
                   f"Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
    window.append(now)


# ── expression validator ────────────────────────────────────────────────────

_DANGEROUS_NAMES = frozenset({
    "__import__", "open", "exec", "eval", "compile",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "breakpoint", "input", "print",
})


def _validate_expression(code: str) -> Optional[str]:
    """
    Returns an actionable error string, or None if the expression looks safe.
    Catches syntax errors and obvious unsafe patterns before we hand off to eval.
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        col = f" (column {e.offset})" if e.offset else ""
        return f"Syntax error{col}: {e.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _DANGEROUS_NAMES:
            return f"Disallowed name: '{node.id}' is not available in the sandbox."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "Dunder attribute access is not allowed."
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Import statements are not allowed inside expressions."

    return None


# ── result-type classifier ──────────────────────────────────────────────────

def _classify(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, sp.Basic):
        if value.is_number:
            return "number"
        if value.is_Boolean:
            return "boolean"
        if value.is_Matrix:
            return "matrix"
        return "expression"
    if isinstance(value, sp.matrices.MatrixBase):
        return "matrix"
    if isinstance(value, (list, tuple)):
        return "list"
    return "other"


# ── LaTeX renderer ──────────────────────────────────────────────────────────

def _to_latex(value: Any) -> Optional[str]:
    try:
        return sp.latex(value)
    except Exception:
        return None


# ── enhanced plot renderer ──────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64

_PLOT_COLORS = [
    "#55ccff", "#ff7f50", "#7fff00", "#da70d6",
    "#ffd700", "#ff6b6b", "#48d1cc",
]


def _render_plot_v2(
    func_objs: list,
    labels: list[str],
    x_range: tuple[float, float] = (-10, 10),
    points: int = 600,
) -> dict:
    x_vals = np.linspace(x_range[0], x_range[1], points)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.axhline(0, color="#555", linewidth=0.6)
    ax.axvline(0, color="#555", linewidth=0.6)

    all_finite = []
    for idx, func_obj in enumerate(func_objs):
        y_vals = []
        for v in x_vals:
            try:
                y = func_obj(v)
                y = complex(y).real
                y_vals.append(float(y) if np.isfinite(y) else np.nan)
            except Exception:
                y_vals.append(np.nan)
        y_arr = np.array(y_vals)
        finite = y_arr[np.isfinite(y_arr)]
        all_finite.extend(finite.tolist())
        color = _PLOT_COLORS[idx % len(_PLOT_COLORS)]
        label = labels[idx] if idx < len(labels) else f"f{idx+1}(x)"
        ax.plot(x_vals, y_arr, color=color, linewidth=1.8, label=label)

    if len(func_objs) > 1:
        ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#333",
                  labelcolor="#aaa", framealpha=0.9)

    if all_finite:
        lo = np.percentile(all_finite, 1)
        hi = np.percentile(all_finite, 99)
        pad = max((hi - lo) * 0.15, 0.5)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.grid(True, color="#2a2a2a", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.patch.set_facecolor("#111")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                facecolor="#111", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return {
        "type": "plot",
        "image": base64.b64encode(buf.read()).decode("utf-8"),
        "x_range": list(x_range),
        "functions": labels,
    }


# ── request / response models ───────────────────────────────────────────────

class CommandV2(BaseModel):
    code: str = Field(
        ...,
        example="Func(x**2 - 1).plot()",
        description="MatLang / SymPy expression to evaluate.",
    )
    x_range: Optional[List[float]] = Field(
        None,
        example=[-5, 5],
        description="[min, max] for plot x-axis. Defaults to [-10, 10].",
    )
    points: int = Field(
        600,
        ge=100,
        le=2000,
        description="Number of sample points for plots (100–2000).",
    )
    namespace: Optional[str] = Field(
        None,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Variable namespace. Variables are scoped to `<apiKey>:<namespace>`. "
            "Omit to use the default namespace."
        ),
    )


class BatchCommand(BaseModel):
    expressions: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Up to 10 expressions evaluated in sequence, sharing state.",
        example=["a = x**2 + 1", "b = x - 3", "Func(a - b).plot()"],
    )
    x_range: Optional[List[float]] = Field(None, example=[-5, 5])
    points: int = Field(600, ge=100, le=2000)
    namespace: Optional[str] = Field(
        None, max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    )


# ── namespace helper ────────────────────────────────────────────────────────

def _ns_key(api_key: str, namespace: Optional[str]) -> str:
    return f"{api_key}:{namespace}" if namespace else api_key


def _get_ns_vars(key_info: dict, ns_key: str) -> dict:
    """Return the variable dict for this namespace, creating it if needed."""
    key_info.setdefault("namespaces", {})
    key_info["namespaces"].setdefault(ns_key, {})
    return key_info["namespaces"][ns_key]


# ── eval core (shared by single and batch) ──────────────────────────────────

def _eval_one(
    code: str,
    allowed: dict,
    x_range: tuple,
    points: int,
) -> dict:
    """
    Evaluate a single expression string. Returns a result dict.
    Mutates `allowed` in-place when code is an assignment.
    """
    t0 = time.perf_counter()

    # ── plot shorthand ──────────────────────────────────────────────────────
    if code.endswith(".plot()"):
        func_code = code[: -len(".plot()")]
        # support comma-separated multi-function: "Func(x**2), Func(x+1).plot()"
        # actually cleaner: allow "plot([Func(x**2), Func(x+1)])"
        # For backwards compat keep single-func path here
        err = _validate_expression(func_code)
        if err:
            return {"type": "error", "result": err}
        func_obj = eval(func_code, {"__builtins__": {}}, allowed)
        if not isinstance(func_obj, Func):
            return {"type": "error",
                    "result": "`.plot()` can only be called on a Func object."}
        elapsed = int((time.perf_counter() - t0) * 1000)
        plot_result = _render_plot_v2(
            [func_obj], [func_code], x_range=x_range, points=points
        )
        plot_result["meta"] = _meta(elapsed)
        return plot_result

    # ── multi-plot: plot([Func(...), Func(...)]) ────────────────────────────
    if code.startswith("plot(") and code.endswith(")"):
        inner = code[5:-1].strip()
        err = _validate_expression(inner)
        if err:
            return {"type": "error", "result": err}
        result = eval(inner, {"__builtins__": {}}, allowed)
        if isinstance(result, Func):
            objs, lbls = [result], [inner]
        elif isinstance(result, (list, tuple)):
            objs, lbls = [], []
            for i, item in enumerate(result):
                if not isinstance(item, Func):
                    return {"type": "error",
                            "result": f"Item {i} in plot list is not a Func."}
                objs.append(item)
                lbls.append(f"f{i+1}(x)")
        else:
            return {"type": "error",
                    "result": "plot() expects a Func or list of Func objects."}
        elapsed = int((time.perf_counter() - t0) * 1000)
        plot_result = _render_plot_v2(objs, lbls, x_range=x_range, points=points)
        plot_result["meta"] = _meta(elapsed)
        return plot_result

    # ── assignment ──────────────────────────────────────────────────────────
    if _is_assignment(code):
        var_name, _, expr_str = code.partition("=")
        var_name = var_name.strip()
        expr_str = expr_str.strip()
        err = _validate_expression(expr_str)
        if err:
            return {"type": "error", "result": err}
        value = eval(expr_str, {"__builtins__": {}}, allowed)
        allowed[var_name] = value          # mutate so later exprs see it
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "type": "assignment",
            "result": f"{var_name} = {_safe_str(value)}",
            "latex": _to_latex(value),
            "result_type": _classify(value),
            "meta": _meta(elapsed),
        }

    # ── expression ──────────────────────────────────────────────────────────
    err = _validate_expression(code)
    if err:
        return {"type": "error", "result": err}

    value = sp.parse_expr(code, local_dict=allowed, evaluate=True)

    if isinstance(value, Lim):
        value = value.evaluate()

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "type": "text",
        "result": _safe_str(value),
        "latex": _to_latex(value),
        "result_type": _classify(value),
        "meta": _meta(elapsed),
    }


def _meta(elapsed_ms: int) -> dict:
    return {
        "execution_time_ms": elapsed_ms,
        "sympy_version": sp.__version__,
    }


# ── router ──────────────────────────────────────────────────────────────────

v1beta2 = APIRouter(prefix="/v1beta2", tags=["v1beta2"])


@v1beta2.post(
    "/eval",
    summary="Evaluate a MatLang expression",
    description=(
        "Executes a MatLang / SymPy expression.\n\n"
        "**Improvements over v1beta1**\n"
        "- Rate limited: 60 requests / 60 s per API key (HTTP 429 on breach)\n"
        "- Pre-eval syntax & safety validation with actionable error messages\n"
        "- Response includes `latex`, `result_type`, and `meta` (timing, sympy version)\n"
        "- Plot: configurable `x_range` and `points` in request body\n"
        "- Multi-function plot via `plot([Func(expr1), Func(expr2)])`\n"
        "- Variable namespacing: supply `namespace` to isolate variable scopes\n\n"
        "Supply your API key as `?apiKey=YOUR_KEY`."
    ),
)
def v1beta2_evaluate(
    command: CommandV2,
    apiKey: str = Query(..., alias="apiKey"),
):
    _check_rate_limit(apiKey)
    key_info = _validate_api_key(apiKey)

    ns_key = _ns_key(apiKey, command.namespace)
    ns_vars = _get_ns_vars(key_info, ns_key)

    allowed = _build_globals()
    allowed.update(ns_vars)

    x_range = tuple(command.x_range) if command.x_range and len(command.x_range) == 2 \
        else (-10.0, 10.0)

    code = command.code.strip()
    if not code:
        return {"type": "error", "result": "Empty input."}

    try:
        result = _eval_one(code, allowed, x_range, command.points)
        # persist any new assignments back to the namespace store
        if result.get("type") == "assignment":
            var_name = result["result"].split("=")[0].strip()
            ns_vars[var_name] = allowed[var_name]
            _save_store()
        return result
    except ZeroDivisionError as e:
        return {"type": "error", "result": f"Math error: {e}"}
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        return {"type": "error", "result": f"Error: {e}"}
    except Exception as e:
        return {"type": "error", "result": f"Unexpected error: {e}"}


@v1beta2.post(
    "/eval/batch",
    summary="Evaluate multiple expressions in one request",
    description=(
        "Evaluates up to **10** expressions in sequence. Each expression can read "
        "variables set by earlier ones in the same batch. Assignments are persisted "
        "to the namespace store after the batch completes.\n\n"
        "Returns a list of individual results in the same order as `expressions`."
    ),
)
def v1beta2_batch(
    batch: BatchCommand,
    apiKey: str = Query(..., alias="apiKey"),
):
    _check_rate_limit(apiKey)
    key_info = _validate_api_key(apiKey)

    ns_key = _ns_key(apiKey, batch.namespace)
    ns_vars = _get_ns_vars(key_info, ns_key)

    allowed = _build_globals()
    allowed.update(ns_vars)

    x_range = tuple(batch.x_range) if batch.x_range and len(batch.x_range) == 2 \
        else (-10.0, 10.0)

    results = []
    for code in batch.expressions:
        code = code.strip()
        if not code:
            results.append({"type": "error", "result": "Empty expression."})
            continue
        try:
            res = _eval_one(code, allowed, x_range, batch.points)
            results.append(res)
        except ZeroDivisionError as e:
            results.append({"type": "error", "result": f"Math error: {e}"})
        except (SyntaxError, NameError, TypeError, ValueError) as e:
            results.append({"type": "error", "result": f"Error: {e}"})
        except Exception as e:
            results.append({"type": "error", "result": f"Unexpected error: {e}"})

    # persist all new assignments from this batch
    for key in list(allowed.keys()):
        if key not in _build_globals():
            ns_vars[key] = allowed[key]
    _save_store()

    return {"type": "batch", "results": results, "count": len(results)}


@v1beta2.post(
    "/reset",
    summary="Clear session variables for API key (optionally scoped to a namespace)",
)
def v1beta2_reset(
    apiKey: str = Query(..., alias="apiKey"),
    namespace: Optional[str] = Query(
        None,
        description="Clear only this namespace. Omit to clear ALL namespaces.",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    ),
):
    _check_rate_limit(apiKey)
    key_info = _validate_api_key(apiKey)

    if namespace:
        ns_key = _ns_key(apiKey, namespace)
        key_info.setdefault("namespaces", {}).pop(ns_key, None)
        _save_store()
        return {"type": "text",
                "result": f"Namespace '{namespace}' cleared."}

    key_info["namespaces"] = {}
    key_info["variables"] = {}   # also wipe legacy flat store for this key
    _save_store()
    return {"type": "text", "result": "All namespaces cleared for this API key."}


@v1beta2.get(
    "/vars",
    summary="List variables (optionally filtered by namespace)",
)
def v1beta2_list_vars(
    apiKey: str = Query(..., alias="apiKey"),
    namespace: Optional[str] = Query(
        None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
    ),
):
    _check_rate_limit(apiKey)
    key_info = _validate_api_key(apiKey)

    if namespace:
        ns_key = _ns_key(apiKey, namespace)
        ns_vars = key_info.get("namespaces", {}).get(ns_key, {})
        return {
            "type": "vars",
            "namespace": namespace,
            "result": {k: _safe_str(v) for k, v in ns_vars.items()},
        }

    # return all namespaces
    all_ns = key_info.get("namespaces", {})
    return {
        "type": "vars",
        "namespaces": {
            ns: {k: _safe_str(v) for k, v in vars_.items()}
            for ns, vars_ in all_ns.items()
        },
    }


@v1beta2.get(
    "/vars/{name}",
    summary="Get a single variable by name",
)
def v1beta2_get_var(
    name: str,
    apiKey: str = Query(..., alias="apiKey"),
    namespace: Optional[str] = Query(None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
):
    _check_rate_limit(apiKey)
    key_info = _validate_api_key(apiKey)
    ns_key = _ns_key(apiKey, namespace)
    ns_vars = key_info.get("namespaces", {}).get(ns_key, {})

    if name not in ns_vars:
        raise HTTPException(status_code=404, detail=f"Variable '{name}' not found.")

    value = ns_vars[name]
    return {
        "type": "var",
        "name": name,
        "result": _safe_str(value),
        "latex": _to_latex(value),
        "result_type": _classify(value),
    }


@v1beta2.delete(
    "/vars/{name}",
    summary="Delete a single variable by name",
)
def v1beta2_delete_var(
    name: str,
    apiKey: str = Query(..., alias="apiKey"),
    namespace: Optional[str] = Query(None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"),
):
    _check_rate_limit(apiKey)
    key_info = _validate_api_key(apiKey)
    ns_key = _ns_key(apiKey, namespace)
    ns_vars = key_info.get("namespaces", {}).get(ns_key, {})

    if name not in ns_vars:
        raise HTTPException(status_code=404, detail=f"Variable '{name}' not found.")

    del ns_vars[name]
    _save_store()
    return {"type": "text", "result": f"Variable '{name}' deleted."}

# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

class AuthIn(BaseModel):
    email: str
    password: str


def _get_user_from_auth(authorization: str = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization")
    if authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        email = _sessions.get(token)
        if not email:
            raise HTTPException(status_code=401, detail="Invalid session token")
        return email
    key = authorization
    info = _store.get("api_keys", {}).get(key)
    if info:
        return info.get("owner")
    raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/auth/register")
def register(auth: AuthIn):
    email = auth.email.strip().lower()
    if not email or not auth.password:
        raise HTTPException(status_code=400, detail="email and password required")
    if email in _store["users"]:
        raise HTTPException(status_code=400, detail="user exists")
    _store["users"][email] = {"password": _hash_password(auth.password), "created": None, "keys": []}
    _save_store()
    return {"status": "ok"}


@app.post("/auth/login", summary="Login and receive a session token")
def login(auth: AuthIn):
    email = auth.email.strip().lower()
    user = _store["users"].get(email)
    if not user or user.get("password") != _hash_password(auth.password):
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = secrets.token_urlsafe(24)
    _sessions[token] = email
    return {"token": token}


@app.get("/auth/keys")
def list_keys(authorization: str = Header(None)):
    email = _get_user_from_auth(authorization)
    keys = []
    for k, info in _store.get("api_keys", {}).items():
        if info.get("owner") == email:
            keys.append({"key": k, "permissions": info.get("permissions", [])})
    return {"keys": keys}


@app.post("/auth/keys")
def create_key(authorization: str = Header(None)):
    email = _get_user_from_auth(authorization)
    key = secrets.token_urlsafe(32)
    _store.setdefault("api_keys", {})[key] = {"owner": email, "permissions": ["read", "write"], "variables": {}}
    _store.setdefault("users", {}).setdefault(email, {}).setdefault("keys", []).append(key)
    apiKeys[key] = _store["api_keys"][key]
    _save_store()
    return {"key": key}


@app.delete("/auth/keys/{key}")
def revoke_key(key: str, authorization: str = Header(None)):
    email = _get_user_from_auth(authorization)
    info = _store.get("api_keys", {}).get(key)
    if not info:
        raise HTTPException(status_code=404, detail="key not found")
    if info.get("owner") != email:
        raise HTTPException(status_code=403, detail="not owner")
    _store["api_keys"].pop(key, None)
    apiKeys.pop(key, None)
    user_keys = _store.setdefault("users", {}).get(email, {}).get("keys", [])
    if key in user_keys:
        user_keys.remove(key)
    _save_store()
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# FIX 7: Register all routers — without this, none of the prefixed routes exist
# ---------------------------------------------------------------------------
app.include_router(v1beta1)
app.include_router(v1beta2)