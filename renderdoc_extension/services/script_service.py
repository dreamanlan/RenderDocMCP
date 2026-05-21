"""
Script execution service for RenderDoc.
Allows arbitrary Python code execution against the ReplayController.
"""

import io
import json
import traceback
import contextlib

import renderdoc as rd

try:
    import qrenderdoc as qrd
except Exception:
    qrd = None


# Max bytes for stdout / stderr / result payloads.
_MAX_OUTPUT = 64 * 1024


def _truncate(text, limit=_MAX_OUTPUT):
    """Truncate text to limit bytes, preserving head and tail."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    b = text.encode("utf-8", errors="replace")
    if len(b) <= limit:
        return text
    half = (limit - 64) // 2
    head = b[:half].decode("utf-8", errors="replace")
    tail = b[-half:].decode("utf-8", errors="replace")
    return head + "\n...[truncated %d bytes]...\n" % (len(b) - 2 * half) + tail


def _make_json_safe(value, _depth=0):
    """Convert value to JSON-serializable form. Falls back to str() for unknowns."""
    if _depth > 8:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            try:
                key = k if isinstance(k, str) else str(k)
            except Exception:
                key = "<key>"
            out[key] = _make_json_safe(v, _depth + 1)
        return out
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return "<bytes len=%d>" % len(value)
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


class ScriptService:
    """Execute arbitrary Python code in RenderDoc's replay thread."""

    def __init__(self, ctx, invoke_fn):
        self.ctx = ctx
        self._invoke = invoke_fn

    def execute_python(self, code, max_output=_MAX_OUTPUT):
        """Execute user-provided Python code on the replay thread.

        The code runs with the following names available:
          controller   - ReplayController bound to current event.
          pyrenderdoc  - CaptureContext (qrenderdoc).
          rd           - renderdoc module.
          qrd          - qrenderdoc module (may be None).
          result       - assign here for the return value (default None).

        Args:
            code: Python source string (will be compiled with exec).
            max_output: Max bytes for stdout/stderr/result string (default 64KB).

        Returns dict:
            success:   bool
            result:    JSON-safe value (truncated if needed)
            stdout:    captured stdout (truncated)
            stderr:    captured stderr (truncated)
            error:     traceback string if exec failed
            truncated: dict { result: bool, stdout: bool, stderr: bool }
        """
        if not self.ctx.IsCaptureLoaded():
            raise ValueError("No capture loaded")
        if not isinstance(code, str) or not code:
            raise ValueError("code (non-empty string) is required")

        try:
            compiled = compile(code, "<execute_python>", "exec")
        except SyntaxError as e:
            return {
                "success": False,
                "result": None,
                "stdout": "",
                "stderr": "",
                "error": "SyntaxError: %s (line %s, col %s)" % (e.msg, e.lineno, e.offset),
                "truncated": {"result": False, "stdout": False, "stderr": False},
            }

        limit = int(max_output) if max_output else _MAX_OUTPUT
        if limit <= 0:
            limit = _MAX_OUTPUT

        outcome = {
            "success": False,
            "result": None,
            "stdout": "",
            "stderr": "",
            "error": None,
            "truncated": {"result": False, "stdout": False, "stderr": False},
        }

        def callback(controller):
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            ns = {
                "__name__": "__execute_python__",
                "__builtins__": __builtins__,
                "controller": controller,
                "pyrenderdoc": self.ctx,
                "rd": rd,
                "qrd": qrd,
                "result": None,
            }

            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(compiled, ns)
                outcome["success"] = True
                user_result = ns.get("result", None)
                safe = _make_json_safe(user_result)
                # If result is a string, apply size limit; otherwise serialize to check size.
                if isinstance(safe, str):
                    truncated = _truncate(safe, limit)
                    outcome["truncated"]["result"] = (truncated != safe)
                    outcome["result"] = truncated
                else:
                    try:
                        serialized = json.dumps(safe)
                        if len(serialized.encode("utf-8")) > limit:
                            # Replace with truncated string form.
                            outcome["result"] = _truncate(serialized, limit)
                            outcome["truncated"]["result"] = True
                        else:
                            outcome["result"] = safe
                    except Exception:
                        outcome["result"] = _truncate(str(safe), limit)
                        outcome["truncated"]["result"] = True
            except Exception:
                outcome["success"] = False
                outcome["error"] = traceback.format_exc()
            finally:
                so = stdout_buf.getvalue()
                se = stderr_buf.getvalue()
                tr_so = _truncate(so, limit)
                tr_se = _truncate(se, limit)
                outcome["stdout"] = tr_so
                outcome["stderr"] = tr_se
                outcome["truncated"]["stdout"] = (tr_so != so)
                outcome["truncated"]["stderr"] = (tr_se != se)

        self._invoke(callback)
        return outcome
