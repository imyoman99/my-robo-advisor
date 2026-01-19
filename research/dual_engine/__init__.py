"""Dual-Engine backtest package.

Static + Dynamic engines with a master allocator.

Entry points:
- `python -m research.dual_engine.cli --config dual_engine`
"""

from .runner import DualEngineResult, run_dual_engine_backtest
