import builtins
import re
import sys
import types


class UltimateMagicMock(int):
    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self

    def __getitem__(self, item):
        return self

    def __iter__(self):
        return iter([])


class PerfectMockModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return UltimateMagicMock(0)


def install_opengl_guard():
    if getattr(builtins, "_tactile_infiller_opengl_guard_installed", False):
        return

    mock_obj = PerfectMockModule("OpenGL.GL")
    orig_import = builtins.__import__

    def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("OpenGL") or name in ["EGL", "OSMesa"]:
            if globals is not None and "__file__" in globals:
                try:
                    with open(globals["__file__"], "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    tokens = re.findall(
                        r"\b([gG][lL][A-Za-z0-9_]+|[eE][gG][lL][A-Za-z0-9_]+|OSMesa[A-Za-z0-9_]+)\b",
                        content,
                    )
                    for token in tokens:
                        globals.setdefault(token, UltimateMagicMock(0))
                except Exception:
                    pass
            return mock_obj
        return orig_import(name, globals, locals, fromlist, level)

    builtins.__import__ = custom_import
    builtins._tactile_infiller_opengl_guard_installed = True
    sys.modules["EGL"] = mock_obj
    sys.modules["OSMesa"] = mock_obj
    sys.modules["OpenGL"] = mock_obj
    sys.modules["OpenGL.GL"] = mock_obj
    sys.modules["OpenGL.GL.shaders"] = mock_obj
