import argparse

from infra.config import load_yaml_config
from research.backtest.metrics import summary
from research.backtest.pipeline import run_cio_backtest
from research.backtest.report import save_equity_curve, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    backtest_config = load_yaml_config("backtest")
    data_config = load_yaml_config("data")
    universe_config = load_yaml_config("universe")

    start = args.start or backtest_config.get("periods", {}).get("start")
    end = args.end or backtest_config.get("periods", {}).get("end")

    result = run_cio_backtest(
        backtest_cfg=backtest_config,
        data_cfg=data_config,
        universe_cfg=universe_config,
        start=start,
        end=end,
    )

    metrics = summary(result.equity, result.returns)
    print(metrics)
    save_metrics(metrics)
    save_equity_curve(result.equity)


if __name__ == "__main__":
    main()
