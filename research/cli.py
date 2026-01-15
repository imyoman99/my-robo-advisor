import argparse

import pandas as pd

from infra.config import load_yaml_config
from infra.data_manager.loader import load_ohlcv_bulk
from infra.data_manager.saver import save_equity_curve
from research.backtest.engine import backtest
from research.backtest.metrics import summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--source", type=str, default="fdr")
    args = parser.parse_args()

    config = load_yaml_config("research")
    universe = config.get("universe", [])
    if not universe:
        raise ValueError("research.yaml에 universe가 비어 있습니다.")

    data = load_ohlcv_bulk(universe, start=args.start, end=args.end, source=args.source)
    prices = pd.concat({k: v["close"] for k, v in data.items()}, axis=1).dropna()

    result = backtest(
        prices,
        rebalance_freq=config.get("rebalance_freq", "M"),
        fee_bps=config.get("fee_bps", 0.0),
        slippage_bps=config.get("slippage_bps", 0.0),
    )

    metrics = summary(result.equity, result.returns)
    print(metrics)
    save_equity_curve(result.equity)


if __name__ == "__main__":
    main()
