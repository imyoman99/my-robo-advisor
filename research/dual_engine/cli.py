from __future__ import annotations

import argparse

from .runner import run_dual_engine_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-Engine Backtest")
    parser.add_argument(
        "--config", type=str, default="dual_engine", help="config/{name}.yaml"
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--print-equity-head", type=int, default=0)
    args = parser.parse_args()

    result = run_dual_engine_backtest(
        config_name=args.config, start=args.start, end=args.end
    )
    print("[DualEngine]", result.performance)
    print("[Benchmark]", result.benchmark_performance)

    if args.print_equity_head and not result.equity.empty:
        print(result.equity.head(args.print_equity_head))


if __name__ == "__main__":
    main()
