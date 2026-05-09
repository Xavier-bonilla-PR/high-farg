"""Slipnet weight learning system for Copycat."""

import collections as _collections
import inspect as _inspect
import sys as _sys
import types as _types

# ── inspect.getargspec shim (removed in Python 3.11) ──────────────────────
# copycat/codeletMethods.py asserts on a 4-field namedtuple; getfullargspec
# returns 7 fields, so we wrap it to keep the original ArgSpec shape.
if not hasattr(_inspect, "getargspec"):
    _ArgSpec = _collections.namedtuple(
        "ArgSpec", ["args", "varargs", "keywords", "defaults"]
    )

    def _getargspec(f):
        s = _inspect.getfullargspec(f)
        return _ArgSpec(s.args, s.varargs, s.varkw, s.defaults)

    _inspect.getargspec = _getargspec

# ── Headless / missing-library mocks ──────────────────────────────────────
# copycat's __init__.py transitively imports tkinter (via gui), matplotlib,
# and numpy.  None of these are needed by the learner.  We inject stub
# modules into sys.modules *before* any copycat code is imported so that
# those imports succeed in headless / minimal environments.

def _install_stubs():
    def _stub(name):
        m = _types.ModuleType(name)
        _sys.modules[name] = m
        return m

    # tkinter / GUI
    class _GUI:
        def __init__(self, *a, **kw): pass

    for _n in ("copycat.gui", "copycat.gui.gui"):
        if _n not in _sys.modules:
            _m = _stub(_n)
            _m.GUI = _GUI

    # matplotlib
    for _n in ("matplotlib", "matplotlib.pyplot"):
        if _n not in _sys.modules:
            _m = _stub(_n)
            # plot.py calls plt.rcdefaults() at module scope
            _m.rcdefaults = lambda: None
            _m.subplots = lambda *a, **kw: (None, None)
            _m.show = lambda: None
            _m.savefig = lambda *a, **kw: None

    # numpy
    if "numpy" not in _sys.modules:
        _stub("numpy")

    # copycat.plot — mock the whole submodule so plot_answers is importable
    if "copycat.plot" not in _sys.modules:
        _m = _stub("copycat.plot")
        _m.plot_answers = lambda *a, **kw: None

_install_stubs()
