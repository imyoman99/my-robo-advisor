import argparse

import pandas as pd

from infra.config import load_yaml_config
from infra.data_manager.downloader import load_ohlcv_bulk
from research.backtest.report import save_equity_curve, save_equity_plot, save_metrics
from research.backtest.engine import backtest, backtest_threshold
from research.backtest.portfolio import combined_weight_func
from research.backtest.metrics import summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--source", type=str, default="fdr")
    args = parser.parse_args()

    backtest_config = load_yaml_config("backtest")
    data_config = load_yaml_config("data")
    universe_config = load_yaml_config("universe")
    periods_config = load_yaml_config("periods")

    universe_key = backtest_config.get("universe_key")
    if universe_key:
        universe = universe_config.get(universe_key, [])
    else:
        universe = backtest_config.get("universe", [])
    if not universe:
        raise ValueError("backtest.yaml에 universe 또는 universe_key가 비어 있습니다.")

    default_periods = periods_config.get("default", {})
    start = (
        args.start
        or backtest_config.get("periods", {}).get("start")
        or default_periods.get("start")
    )
    end = (
        args.end
        or backtest_config.get("periods", {}).get("end")
        or default_periods.get("end")
    )

    data = load_ohlcv_bulk(
        universe,
        start=start,
        end=end,
        source=data_config.get("source", args.source),
        use_cache=data_config.get("use_cache", True),
        cache_dir=data_config.get("cache_dir", "data/raw"),
        preprocess=data_config.get("preprocess", True),
    )
    prices = pd.concat({k: v["close"] for k, v in data.items()}, axis=1).dropna()

    weight_func = combined_weight_func(backtest_config)

    rebalance_cfg = backtest_config.get("rebalance", {})
    costs_cfg = backtest_config.get("costs", {})

    if rebalance_cfg.get("mode", "time") == "threshold":
        result = backtest_threshold(
            prices,
            weight_func=weight_func,
            rebalance_threshold=rebalance_cfg.get("threshold", 0.05),
            initial_equity=backtest_config.get("initial_equity", 1.0),
            fee_bps=costs_cfg.get("fee_bps", 0.0),
            slippage_bps=costs_cfg.get("slippage_bps", 0.0),
        )
    else:
        result = backtest(
            prices,
            weight_func=weight_func,
            rebalance_freq=rebalance_cfg.get("freq", "M"),
            initial_equity=backtest_config.get("initial_equity", 1.0),
            fee_bps=costs_cfg.get("fee_bps", 0.0),
            slippage_bps=costs_cfg.get("slippage_bps", 0.0),
        )

    metrics = summary(result.equity, result.returns)
    print(metrics)
    save_metrics(metrics)
    save_equity_curve(result.equity)
    try:
        save_equity_plot(result.equity)
    except ImportError:
        print("matplotlib 미설치로 차트 저장은 건너뜁니다.")


if __name__ == "__main__":
    main()
