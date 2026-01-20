from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

if __package__ is None or __package__ == "":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from research.backtester.runner import run_dual_engine_backtest
    from research.backtester.config import load_yaml_config, resolve_logic_config_name
    from research.backtester.metrics import performance_summary
    from research.backtester.utils import month_day_in_season
else:
    from .runner import run_dual_engine_backtest
    from .config import load_yaml_config, resolve_logic_config_name
    from .metrics import performance_summary
    from .utils import month_day_in_season


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtester")
    parser.add_argument(
        "--config", type=str, default="backtester", help="config/{name}.yaml"
    )
    parser.add_argument(
        "--logic",
        type=str,
        default=None,
        help="logic config name (e.g. logic_fast_follower_01 or config.logic_fast_follower_01)",
    )
    parser.add_argument(
        "--json-run",
        type=str,
        default=None,
        help="results run folder name to load config_backtester.json & config_logic.json",
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--print-equity-head", type=int, default=0)
    args = parser.parse_args()

    result = run_dual_engine_backtest(
        config_name=args.config,
        logic_name=args.logic,
        json_run=args.json_run,
        start=args.start,
        end=args.end,
    )
    print("[Backtester]", result.performance)
    print("[Benchmark]", result.benchmark_performance)

    _save_results(result, args.config, args.logic)

    if args.print_equity_head and not result.equity.empty:
        print(result.equity.head(args.print_equity_head))


def _resolve_output_root(cfg: dict) -> Path:
    results_cfg = cfg.get("RESULTS", {}) or {}
    root = results_cfg.get("root") or "results"
    root_path = Path(root)
    if not root_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        root_path = (project_root / root_path).resolve()
    return root_path


def _drawdown_series(equity: pd.Series | None) -> pd.Series:
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _mdd_from_equity(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    dd = _drawdown_series(equity)
    return float(dd.min()) if not dd.empty else 0.0


def _perf_dict(equity: pd.Series, returns: pd.Series, trading_days: int) -> dict:
    if equity is None or equity.empty:
        return {"CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "TotalReturn": 0.0}
    return performance_summary(equity, returns, trading_days)


def _format_metrics(metrics: dict) -> dict:
    return {
        "CAGR": float(metrics.get("CAGR", 0.0)),
        "MDD": float(metrics.get("MDD", 0.0)),
        "Sharpe": float(metrics.get("Sharpe", 0.0)),
        "TotalReturn": float(metrics.get("TotalReturn", 0.0)),
    }


def _season_mask(index: pd.DatetimeIndex, start_md: str, end_md: str) -> pd.Series:
    return pd.Series(
        [month_day_in_season(pd.Timestamp(d), start_md, end_md) for d in index],
        index=index,
    )


def _save_results(result, config_name: str, logic_name: str | None) -> None:
    output_root = _resolve_output_root(result.config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{config_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        user_cfg = load_yaml_config(config_name)
    except Exception:
        user_cfg = {}

    logic_cfg = {}
    logic_config_name = resolve_logic_config_name(user_cfg, logic_name)
    if logic_config_name:
        try:
            logic_cfg = load_yaml_config(logic_config_name)
        except Exception:
            logic_cfg = {}

    (run_dir / "config_backtester.json").write_text(
        json.dumps(user_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "config_logic.json").write_text(
        json.dumps(logic_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "config_merged.json").write_text(
        json.dumps(result.config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selection_log = list(getattr(result, "dynamic_selection_log", []) or [])
    if selection_log:
        (run_dir / "dynamic_selection.json").write_text(
            json.dumps(selection_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        rows: list[dict[str, Any]] = []
        for entry in selection_log:
            date = entry.get("date")
            season = entry.get("season")
            selected = set(entry.get("selected", []) or [])
            corr_count = entry.get("corr_count", {}) or {}
            momentum = entry.get("momentum", {}) or {}
            corr_rank = entry.get("corr_rank", {}) or {}
            momentum_rank = entry.get("momentum_rank", {}) or {}
            total_score = entry.get("total_score", {}) or {}
            tickers = sorted(set(corr_count) | set(momentum) | selected)
            for t in tickers:
                rows.append(
                    {
                        "date": date,
                        "season": season,
                        "ticker": t,
                        "selected": t in selected,
                        "corr_count": corr_count.get(t),
                        "momentum": momentum.get(t),
                        "corr_rank": corr_rank.get(t),
                        "momentum_rank": momentum_rank.get(t),
                        "total_score": total_score.get(t),
                    }
                )

        if rows:
            pd.DataFrame(rows).to_csv(run_dir / "dynamic_selection.csv", index=False)

    chart_data = None
    if result.equity is not None and not result.equity.empty:
        chart_data = result.equity.to_frame("equity")
        if result.static_equity is not None and not result.static_equity.empty:
            chart_data = chart_data.join(
                result.static_equity.rename("static_equity"), how="outer"
            )
        if result.dynamic_equity is not None and not result.dynamic_equity.empty:
            chart_data = chart_data.join(
                result.dynamic_equity.rename("dynamic_equity"), how="outer"
            )
        if result.benchmark_equity is not None and not result.benchmark_equity.empty:
            chart_data = chart_data.join(
                result.benchmark_equity.rename("benchmark_equity"), how="outer"
            )
        chart_data["drawdown"] = _drawdown_series(chart_data["equity"])
        if "static_equity" in chart_data.columns:
            chart_data["static_drawdown"] = _drawdown_series(
                chart_data["static_equity"]
            )
        if "dynamic_equity" in chart_data.columns:
            chart_data["dynamic_drawdown"] = _drawdown_series(
                chart_data["dynamic_equity"]
            )
        if "benchmark_equity" in chart_data.columns:
            chart_data["benchmark_drawdown"] = _drawdown_series(
                chart_data["benchmark_equity"]
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    if result.equity is not None and not result.equity.empty:
        ax.plot(result.equity.index, result.equity.values, label="Backtester")
    if result.benchmark_equity is not None and not result.benchmark_equity.empty:
        ax.plot(
            result.benchmark_equity.index,
            result.benchmark_equity.values,
            label="Benchmark",
        )
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    perf = result.performance or {}
    bench = result.benchmark_performance or {}
    trading_days = int(result.config.get("TRADING_DAYS", 252))
    static_perf = _perf_dict(result.static_equity, result.static_returns, trading_days)
    dynamic_perf = _perf_dict(
        result.dynamic_equity, result.dynamic_returns, trading_days
    )
    static_perf = _format_metrics(static_perf)
    dynamic_perf = _format_metrics(dynamic_perf)
    perf = _format_metrics(perf)
    bench = _format_metrics(bench)
    summary = (
        "Backtester\n"
        f"CAGR: {perf.get('CAGR', float('nan')):.4f}\n"
        f"MDD: {perf.get('MDD', float('nan')):.4f}\n"
        f"Sharpe: {perf.get('Sharpe', float('nan')):.4f}\n"
        f"TotalReturn: {perf.get('TotalReturn', float('nan')):.4f}\n"
        "\nBenchmark\n"
        f"CAGR: {bench.get('CAGR', float('nan')):.4f}\n"
        f"MDD: {bench.get('MDD', float('nan')):.4f}\n"
        f"Sharpe: {bench.get('Sharpe', float('nan')):.4f}\n"
        f"TotalReturn: {bench.get('TotalReturn', float('nan')):.4f}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(run_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    if chart_data is not None and "drawdown" in chart_data.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(chart_data.index, chart_data["drawdown"], label="Backtester")
        if "benchmark_drawdown" in chart_data.columns:
            ax.plot(
                chart_data.index, chart_data["benchmark_drawdown"], label="Benchmark"
            )
        mdd_text = (
            f"MDD (Backtester): {perf.get('MDD', 0.0):.4f}\n"
            f"MDD (Bench): {bench.get('MDD', 0.0):.4f}"
        )
        ax.text(
            0.02,
            0.98,
            mdd_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        ax.set_title("Drawdown")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "drawdown.png", dpi=150)
        plt.close(fig)

    seasons = result.config.get("DYNAMIC", {}).get("SEASONS", []) or []
    prices = result.prices
    season_sections: list[str] = []
    for season in seasons:
        name = str(season.get("name", "Season"))
        start_md = str(season.get("start_md", "01-01"))
        end_md = str(season.get("end_md", "12-31"))
        tickers = [str(t) for t in (season.get("tickers", []) or [])]
        available = [t for t in tickers if t in prices.columns]

        season_sections.append(f"### {name}")
        season_sections.append("")
        season_sections.append(f"- Range: {start_md} ~ {end_md}")
        season_sections.append("")
        season_sections.append("**Tickers**")
        season_sections.append("")
        season_sections.append(", ".join(tickers) if tickers else "(none)")
        season_sections.append("")

        if not available:
            season_sections.append("- No available tickers in price data.")
            season_sections.append("")
            continue

        sub = prices[available].dropna(how="all")
        if sub.empty:
            season_sections.append("- No price data available.")
            season_sections.append("")
            continue

        mask = _season_mask(sub.index, start_md, end_md)
        sub = sub.loc[mask]
        if sub.empty:
            season_sections.append("- No data in season window.")
            season_sections.append("")
            continue

        norm = sub / sub.iloc[0]
        basket = norm.mean(axis=1)
        basket_returns = basket.pct_change(fill_method=None).fillna(0.0)
        basket_perf = _format_metrics(
            performance_summary(basket, basket_returns, trading_days)
        )

        season_sections.append("**Season Basket (equal-weight)**")
        season_sections.append("")
        season_sections.append("| Metric | Value |")
        season_sections.append("|---|---:|")
        season_sections.append(f"| CAGR | {basket_perf['CAGR']:.6f} |")
        season_sections.append(f"| MDD | {basket_perf['MDD']:.6f} |")
        season_sections.append(f"| Sharpe | {basket_perf['Sharpe']:.6f} |")
        season_sections.append(f"| TotalReturn | {basket_perf['TotalReturn']:.6f} |")
        season_sections.append("")

        season_sections.append("**Per-Ticker Return/MDD**")
        season_sections.append("")
        season_sections.append("| Ticker | TotalReturn | MDD |")
        season_sections.append("|---|---:|---:|")
        for t in available:
            s = sub[t].dropna()
            if s.empty:
                continue
            total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
            mdd = _mdd_from_equity(s)
            season_sections.append(f"| {t} | {total_ret:.6f} | {mdd:.6f} |")
        season_sections.append("")

    selection_lines: list[str] = []
    if selection_log:
        selection_lines = [
            "## Dynamic Selection",
            "",
            "- Selection log: dynamic_selection.json",
            "- Flat scores: dynamic_selection.csv",
            "",
        ]

    md_lines = [
        f"# Backtester ({config_name})",
        "",
        f"- Run ID: {run_id}",
        "",
        "## Summary",
        "",
        "| Metric | Backtester | Benchmark |",
        "|---|---:|---:|",
        f"| CAGR | {perf.get('CAGR', float('nan')):.6f} | {bench.get('CAGR', float('nan')):.6f} |",
        f"| MDD | {perf.get('MDD', float('nan')):.6f} | {bench.get('MDD', float('nan')):.6f} |",
        f"| Sharpe | {perf.get('Sharpe', float('nan')):.6f} | {bench.get('Sharpe', float('nan')):.6f} |",
        f"| TotalReturn | {perf.get('TotalReturn', float('nan')):.6f} | {bench.get('TotalReturn', float('nan')):.6f} |",
        "",
        "## Strategy Breakdown",
        "",
        "| Metric | Static | Dynamic |",
        "|---|---:|---:|",
        f"| CAGR | {static_perf['CAGR']:.6f} | {dynamic_perf['CAGR']:.6f} |",
        f"| MDD | {static_perf['MDD']:.6f} | {dynamic_perf['MDD']:.6f} |",
        f"| Sharpe | {static_perf['Sharpe']:.6f} | {dynamic_perf['Sharpe']:.6f} |",
        f"| TotalReturn | {static_perf['TotalReturn']:.6f} | {dynamic_perf['TotalReturn']:.6f} |",
        "",
        *selection_lines,
        "## Charts",
        "",
        "![Equity Curve](equity_curve.png)",
        "",
        "![Drawdown](drawdown.png)",
        "",
        "## Seasonal Performance",
        "",
        *season_sections,
        "",
        "## Data",
        "",
        "- (no csv output)",
    ]
    (run_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[Saved] {run_dir}")


if __name__ == "__main__":
    main()
