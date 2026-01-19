"""Backward-compatible wrapper.

이 파일은 과거 단일 파일 프로토타입이었고, 현재는 모듈화된 패키지로 이전되었습니다.

권장 실행:
`python -m research.dual_engine.cli --config dual_engine`
"""

from __future__ import annotations

from research.dual_engine.runner import run_dual_engine_backtest


def run() -> None:
    result = run_dual_engine_backtest(config_name="dual_engine")
    print("[DualEngine]", result.performance)
    print("[Benchmark]", result.benchmark_performance)


if __name__ == "__main__":
    run()
