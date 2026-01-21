from __future__ import annotations

from typing import Callable, Dict, Optional

import pandas as pd

from infra.config import load_yaml_config
from infra.logger import get_logger
from live.broker.kis import KISBroker
from live.data.market_data import MarketDataProvider
from live.execution.order_manager import OrderManager
from live.monitoring.slack import SlackNotifier
from live.state import LiveStateStore
from live.strategy.dynamic import LiveDynamicStrategy
from live.strategy.static import LiveStaticStrategy
from core.types import DynamicState


WeightFunc = Callable[[Dict[str, float]], Dict[str, float]]


class LiveEngine:
    def __init__(
        self, weight_func: Optional[WeightFunc] = None, mock: bool = True
    ) -> None:
        self.config = load_yaml_config("live")
        strategy_config_name = str(self.config.get("strategy_config", "backtester"))
        self.strategy_config = load_yaml_config(strategy_config_name)
        self.logger = get_logger("live.engine")
        self.broker = KISBroker(
            mock=mock,
            state_path=self.config.get("mock_state_file"),
            initial_cash=float(self.config.get("initial_cash", 10_000_000)),
        )
        self.order_manager = OrderManager(self.broker)
        self.notifier = SlackNotifier(self.config.get("slack_webhook"))
        cache_dir = self.config.get("cache_dir")
        self.market = MarketDataProvider(
            source=str(self.config.get("data_source", "kis")),
            use_cache=bool(self.config.get("use_cache", False)),
            cache_dir=str(cache_dir) if cache_dir else None,
            preprocess=True,
            kis_client=getattr(self.broker, "client", None),
        )
        self.state_store = LiveStateStore(
            str(self.config.get("state_file", "live/state.json"))
        )
        self.strategy = LiveDynamicStrategy(self.strategy_config)
        self.static_strategy = LiveStaticStrategy(self.strategy_config)
        self.weight_func = weight_func or self._default_weight_func

    def _default_weight_func(self, prices: Dict[str, float]) -> Dict[str, float]:
        symbols = list(prices.keys())
        if not symbols:
            return {}
        w = 1.0 / len(symbols)
        return {s: w for s in symbols}

    def _get_universe(self) -> list[str]:
        static_tickers = self.static_strategy.tickers()
        universe_source = str(self.config.get("dynamic_universe_source") or "kis_krx")
        limit = self.config.get("universe_limit")
        dynamic_universe = self.market.list_universe(
            source=universe_source,
            limit=int(limit) if limit is not None else None,
        )
        merged = list(dict.fromkeys(dynamic_universe + static_tickers))
        return merged

    def _lookback_days(self) -> int:
        cfg = self.strategy_config.get("DYNAMIC", {}) or {}
        sel_cfg = cfg.get("SELECTION", {}) or {}
        mom_window = int(cfg.get("MOMENTUM_WINDOW", 60))
        corr_window = int(cfg.get("CORRELATION_WINDOW", 60))
        lookback_years = int(sel_cfg.get("lookback_years", 1))
        base = max(mom_window, corr_window) + 5
        return int(max(base, lookback_years * 365))

    def _target_positions(self, prices: Dict[str, float]) -> Dict[str, int]:
        target_weights = self.weight_func(prices)
        total_cash = self.config.get("initial_cash", 10_000_000)
        target_positions = {}
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            qty = int((total_cash * weight) / price)
            target_positions[symbol] = qty
        return target_positions

    def _target_positions_from_weights(
        self,
        weights: Dict[str, float],
        prices: Dict[str, float],
        *,
        cash: float,
        holdings: Dict[str, int],
    ) -> Dict[str, int]:
        total_equity = float(cash)
        for symbol, qty in holdings.items():
            px = prices.get(symbol, 0.0)
            if px is None or px <= 0:
                continue
            total_equity += float(qty) * float(px)

        targets: Dict[str, int] = {}
        for symbol, weight in weights.items():
            px = prices.get(symbol, 0.0)
            if px is None or px <= 0:
                continue
            target_value = total_equity * float(weight)
            targets[symbol] = int(target_value / float(px))
        return targets

    def run_once(self) -> None:
        universe = self._get_universe()
        if not universe:
            self.logger.warning("Universe is empty")
            return

        lookback_days = int(self.config.get("lookback_days", self._lookback_days()))
        close_df = self.market.get_field_frame(
            universe, "close", lookback_days=lookback_days
        )
        volume_df = self.market.get_field_frame(
            universe, "volume", lookback_days=lookback_days
        )
        if close_df.empty:
            self.logger.warning("No price data from API")
            return

        as_of = close_df.index[-1]
        close_prices = close_df.loc[as_of]
        latest_prices = {k: float(v) for k, v in close_prices.items() if pd.notna(v)}

        state = self.state_store.load()
        holdings = self.broker.get_positions()
        try:
            cash = self.broker.get_cash()
        except Exception:
            cash = float(state.cash or self.config.get("initial_cash", 10_000_000))

        dynamic_targets, ranked, selected = self.strategy.compute_targets(
            close_df,
            universe,
            as_of,
            volumes=volume_df if not volume_df.empty else None,
        )
        static_targets = self.static_strategy.target_weights()

        static_ratio = float(self.strategy_config.get("STATIC", {}).get("RATIO", 0.0))
        dynamic_ratio = float(self.strategy_config.get("DYNAMIC", {}).get("RATIO", 1.0))
        total_ratio = static_ratio + dynamic_ratio
        if total_ratio > 0:
            static_ratio /= total_ratio
            dynamic_ratio /= total_ratio

        targets: Dict[str, float] = {}
        for k, v in static_targets.items():
            targets[k] = targets.get(k, 0.0) + (float(v) * static_ratio)
        for k, v in dynamic_targets.items():
            targets[k] = targets.get(k, 0.0) + (float(v) * dynamic_ratio)

        dyn_state = DynamicState(
            cash=float(cash),
            holdings={k: float(v) for k, v in holdings.items()},
            high_water=dict(state.high_water),
        )
        next_state = self.strategy.apply_stop_loss(
            dyn_state,
            prices=close_prices,
            ranked=ranked,
            allowed_symbols=selected,
        )

        stop_orders = []
        for symbol, qty in holdings.items():
            if symbol not in next_state.holdings and qty > 0:
                stop_orders.append(self.broker.place_order(symbol, int(qty), "sell"))

        if stop_orders:
            self.notifier.notify(f"손절 주문: {stop_orders}")

        holdings = self.broker.get_positions()
        try:
            cash = self.broker.get_cash()
        except Exception:
            cash = float(next_state.cash)

        dyn_rebalance = self.strategy.rebalance_dates(close_df.index)
        static_rebalance = self.static_strategy.rebalance_dates(close_df.index)
        rebalance_dates = dyn_rebalance | static_rebalance
        as_of_key = as_of.strftime("%Y-%m-%d")
        is_rebalance_day = (
            as_of in rebalance_dates and state.last_rebalance_date != as_of_key
        )

        orders = []
        if is_rebalance_day and targets:
            target_positions = self._target_positions_from_weights(
                targets,
                latest_prices,
                cash=float(cash),
                holdings=holdings,
            )
            orders = self.order_manager.rebalance_to_target_positions(target_positions)
            if orders:
                self.notifier.notify(f"리밸런싱 주문: {orders}")

        state.last_rebalance_date = (
            as_of_key if is_rebalance_day else state.last_rebalance_date
        )
        state.high_water = dict(next_state.high_water)
        state.last_target_weights = {k: float(v) for k, v in targets.items()}
        state.cash = float(cash)
        self.state_store.save(state)

        if orders or stop_orders:
            self.logger.info("run_once completed with orders")
        else:
            self.logger.info("run_once completed")
