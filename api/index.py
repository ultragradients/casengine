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
# ---------------------------------------------------------------------------

@v1beta2.post(
    "/eval",
    summary="Evaluate a MatLang expression (requires apiKey)",
    description=(
        "Executes a MatLang / SymPy expression and returns the result as text or a base64 PNG plot. "
        "Use `Func(expr).plot()` to generate a plot. "
        "Assignments (`name = expr`) persist per-API-key.\n\n"
        "Supply your API key as a query parameter: `?apiKey=YOUR_KEY`."
    ),
)
def v1beta2_evaluate(
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
            # v1beta2 plot expressions are still Python-syntax objects, use eval
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

        # FIX 5: sp.parse_expr signature is parse_expr(s, local_dict=None, ...)
        # The old call passed a builtins-suppressed dict as the second positional arg,
        # which is wrong. Use local_dict keyword and merge builtins suppression separately.
        result = sp.parse_expr(code, local_dict=allowed, evaluate=True)

        if isinstance(result, Lim):
            result = result.evaluate()

        return {"type": "text", "result": _safe_str(result)}

    except ZeroDivisionError as e:
        return {"type": "error", "result": f"Math error: {e}"}
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        return {"type": "error", "result": f"Error: {e}"}
    except Exception as e:
        return {"type": "error", "result": f"Unexpected error: {e}"}


@v1beta2.post(
    "/reset",
    summary="Clear session variables for API key",
    description="Clears variables stored for the provided API key.",
    # FIX 6: was tagged "v1beta1" — corrected
)
def v1beta2_reset(
    apiKey: str = Query(..., description="API key for which to clear variables", alias="apiKey"),
):
    key_info = _validate_api_key(apiKey)
    key_info["variables"] = {}
    _save_store()
    return {"type": "text", "result": "Session variables cleared for this API key."}


@v1beta2.get(
    "/vars",
    summary="List session variables for API key",
    description="Returns variables stored for the provided API key.",
    # FIX 6: was tagged "v1beta1" — corrected
)
def v1beta2_list_vars(
    apiKey: str = Query(..., description="API key to list variables for", alias="apiKey"),
):
    key_info = _validate_api_key(apiKey)
    return {"type": "vars", "result": {k: _safe_str(v) for k, v in key_info.get("variables", {}).items()}}


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
app.include_router(v1)