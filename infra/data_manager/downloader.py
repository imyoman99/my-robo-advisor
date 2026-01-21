from __future__ import annotations

import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from infra.config import load_yaml_config
except ModuleNotFoundError:  # 직접 실행 시 fallback
    import sys as _sys

    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in _sys.path:
        _sys.path.append(_root)
    from infra.config import load_yaml_config
try:
    from .cache import read_cache, write_cache
    from .preprocess import preprocess_ohlcv
except ImportError:  # 직접 실행 시 fallback
    from infra.data_manager.cache import read_cache, write_cache
    from infra.data_manager.preprocess import preprocess_ohlcv

logger = logging.getLogger(__name__)


def _parse_dt(value: object) -> pd.Timestamp | None:
    try:
        if isinstance(value, pd.Timestamp):
            return value
        ts = pd.to_datetime(str(value), errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts)
    except Exception:
        return None


def _get_parquet_date_range(
    path: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """parquet에서 날짜 범위를 빠르게 추출합니다.

    - raw 포맷(Date/date 컬럼)
    - processed 포맷(인덱스가 date일 수도 있음)
    """

    if not path or not os.path.exists(path):
        return None, None

    def _extract(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if df is None or df.empty:
            return None, None
        for col in ("date", "Date"):
            if col in df.columns:
                return _parse_dt(df[col].min()), _parse_dt(df[col].max())
        # 마지막 폴백: 인덱스가 날짜인 경우
        if isinstance(df.index, pd.DatetimeIndex):
            return pd.Timestamp(df.index.min()), pd.Timestamp(df.index.max())
        try:
            idx = pd.to_datetime(df.index, errors="coerce")
            idx = idx[~pd.isna(idx)]
            if len(idx) == 0:
                return None, None
            return pd.Timestamp(idx.min()), pd.Timestamp(idx.max())
        except Exception:
            return None, None

    for col in ("Date", "date"):
        try:
            df = pd.read_parquet(path, columns=[col])
            s, e = _extract(df)
            if s is not None or e is not None:
                return s, e
        except Exception:
            pass

    try:
        df = pd.read_parquet(path)
    except Exception:
        return None, None

    return _extract(df)


def load_data_config(name: str = "data") -> Dict[str, Any]:
    return load_yaml_config(name)


def get_krx_universe_codes(
    config: Dict[str, Any], *, refresh_master: bool = False
) -> list[str]:
    """KRX 종목코드를 파일 저장 없이 메모리로만 반환."""
    cfg = dict(config)
    lists_cfg = dict(cfg.get("lists", {}) or {})
    lists_cfg["refresh_master"] = bool(refresh_master)
    cfg["lists"] = lists_cfg
    sources_cfg = dict(cfg.get("sources", {}) or {})
    sources_cfg["use_marcap"] = False
    sources_cfg["use_fdr_listing"] = True
    cfg["sources"] = sources_cfg
    downloader = UnifiedDownloader(cfg)
    downloader.refresh_master = bool(refresh_master)
    downloader.use_marcap = False
    downloader.use_fdr = True
    codes, _ = downloader._get_krx_list()
    return list(dict.fromkeys([str(c).zfill(6) for c in codes if str(c).strip()]))


def _latest_krx_list_csv(base_dir: str) -> str | None:
    try:
        base = Path(base_dir)
        candidates = list(base.rglob("krx_codes_*.csv"))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])
    except Exception:
        return None


def _read_csv_codes(
    csv_path: str,
    *,
    code_cols: Iterable[str],
    shares_cols: Iterable[str],
    market_filter: Iterable[str] | None = None,
    pad_code: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    if not csv_path or not os.path.exists(csv_path):
        return [], {}

    df = None
    for enc in ("utf-8", "cp949"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        return [], {}

    code_col = next((c for c in code_cols if c in df.columns), None)
    if code_col is None:
        return [], {}

    # 마켓 필터(옵션): Market/시장/Exchange 컬럼이 있으면 해당 값만 사용
    if market_filter:
        market_col = next(
            (c for c in ("Market", "시장", "Exchange") if c in df.columns), None
        )
        if market_col is not None:
            allowed = {str(x).upper() for x in market_filter}
            df = df[df[market_col].astype(str).str.upper().isin(allowed)]

    df[code_col] = df[code_col].astype(str).str.strip()
    if pad_code:
        df[code_col] = df[code_col].str.zfill(6)
    codes = df[code_col].tolist()

    shares_col = next((c for c in shares_cols if c in df.columns), None)
    shares_map: Dict[str, float] = {}
    if shares_col is not None:
        shares_series = df[shares_col]
        if shares_series.dtype == object:
            shares_series = shares_series.astype(str).str.replace(",", "")
        shares_series = pd.to_numeric(shares_series, errors="coerce").fillna(0)
        shares_map = {
            str(code).zfill(6): float(val)
            for code, val in zip(df[code_col], shares_series)
        }

    return codes, shares_map


def _normalize_fdr_df(
    df: pd.DataFrame, code: str, shares: float | None = None
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.reset_index()
    df["Amount"] = df["Close"] * df["Volume"]
    if shares is None or pd.isna(shares):
        shares = 0
    df["Marcap"] = df["Close"] * shares
    df["Code"] = str(code).zfill(6)

    cols = [
        "Date",
        "Code",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Amount",
        "Marcap",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]


def get_ohlcv(
    code: str,
    start: str | None = None,
    end: str | None = None,
    *,
    preprocess: bool = True,
    use_marcap: bool = True,
    use_fdr: bool = True,
    marcap_dir: str | None = None,
) -> pd.DataFrame:
    """종목 코드와 기간을 받아 OHLCV DataFrame을 반환합니다.

    - 기본: Marcap 기반(가능 시) + FDR로 최신 구간 옵션 보완
    - 반환 포맷: preprocess=True이면 date index + 소문자 컬럼(open/high/low/close/volume...)
    """

    # end 기본값은 호출 시점 기준 (문자열)
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    downloader = StockDownloader(
        base_dir="data/raw",
        start_date=start or "2005-01-03",
        end_date=end,
        preprocess=preprocess,
        overwrite=False,
        max_workers=1,
        marcap_dir=marcap_dir,
        use_marcap=use_marcap,
        use_fdr=use_fdr,
        preload_marcap=False,
    )
    return downloader.fetch_ohlcv(code, shares_map={})


def _load_csv(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    candidates = [
        Path("data/processed") / f"{symbol}.csv",
        Path("data/raw") / f"{symbol}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"CSV not found for {symbol}")

    df = pd.read_csv(path)
    if start or end:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        if start:
            df = df.loc[df.index >= pd.to_datetime(start)]
        if end:
            df = df.loc[df.index <= pd.to_datetime(end)]
    return df


def _load_fdr(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FinanceDataReader가 필요합니다. `pip install finance-datareader` 실행 후 재시도하세요."
        ) from exc
    return fdr.DataReader(symbol, start, end)


def load_ohlcv_bulk(
    universe: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: str = "fdr",
    use_cache: bool = True,
    cache_dir: str = "data/raw",
    preprocess: bool = True,
) -> Dict[str, pd.DataFrame]:
    """유니버스 OHLCV 로드(캐시/전처리 포함).

    - source: "fdr" | "csv"
    - 반환: {symbol: DataFrame}
    """

    data: Dict[str, pd.DataFrame] = {}
    for symbol in universe:
        cache_key = f"{source}:{symbol}:{start}:{end}"  # start/end까지 포함해 안전 캐시
        if use_cache:
            cached = read_cache(cache_key, cache_dir=cache_dir)
            if cached is not None:
                data[symbol] = cached
                continue

        if source == "csv":
            df = _load_csv(symbol, start, end)
        else:
            df = _load_fdr(symbol, start, end)

        if preprocess:
            df = preprocess_ohlcv(df)

        data[symbol] = df
        if use_cache:
            write_cache(cache_key, df, cache_dir=cache_dir)

    return data


class StockDownloader:
    def __init__(
        self,
        base_dir: str = "data/raw",
        start_date: str = "2005-01-03",
        end_date: str = "2026-01-15",
        *,
        preprocess: bool = True,
        overwrite: bool = False,
        max_workers: int = 8,
        marcap_dir: str | None = None,
        use_marcap: bool = True,
        use_fdr: bool = False,
        preload_marcap: bool = True,
    ):
        """KRX 전체 종목을 (Marcap + FDR)로 내려받아 파일로 저장.

        저장 파일은 종목별 Parquet(기본)이며, 저장 전에 preprocess_ohlcv로 컬럼/날짜/결측을 정리할 수 있습니다.
        - 기본: Marcap 기반
        - FDR은 옵션(use_fdr=True)일 때만 최신 구간을 이어붙입니다.
        - preload_marcap=True이면 run()에서 Marcap 데이터를 1회 로딩해 스레드에 공유합니다.
        """

        self.base_dir = base_dir
        self.start_date = start_date
        self.end_date = end_date
        self.preprocess = preprocess
        self.overwrite = overwrite

        self.max_workers = int(max_workers)
        self.use_marcap = bool(use_marcap)
        self.use_fdr = bool(use_fdr)
        self.preload_marcap = bool(preload_marcap)

        default_marcap_dir = os.path.join(os.path.dirname(self.base_dir), "marcap_repo")
        self.marcap_dir = marcap_dir or default_marcap_dir  # marcap 저장소 경로

        self.start_ts = pd.to_datetime(self.start_date)
        self.end_ts = pd.to_datetime(self.end_date)

        self._fdr = None

        os.makedirs(self.base_dir, exist_ok=True)
        if self.use_marcap:
            self._prepare_marcap()

    def _marcap_parquet_path(self, year: int) -> Path:
        return Path(self.marcap_dir) / "data" / f"marcap-{int(year)}.parquet"

    def get_master_from_marcap(self, *, year: int | None = None) -> pd.DataFrame:
        """marcap parquet에서 Code-Name 매핑(마스터)을 생성합니다."""

        if not self.use_marcap:
            return pd.DataFrame()

        target_year = (
            int(year) if year is not None else int(min(self.end_ts.year, 2025))
        )
        p = self._marcap_parquet_path(target_year)
        if not p.exists():
            data_dir = Path(self.marcap_dir) / "data"
            candidates = sorted(data_dir.glob("marcap-*.parquet"))
            if not candidates:
                return pd.DataFrame()
            p = candidates[-1]

        cols = ["Code", "Name", "Market", "Stocks", "Date"]
        df = pd.read_parquet(p, columns=cols)
        if df.empty or "Code" not in df.columns:
            return pd.DataFrame()

        df["Code"] = df["Code"].astype(str).str.strip().str.zfill(6)
        df = df[df["Code"].str.match(r"^\d{6}$", na=False)]
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date")

        # 마지막 값 기준으로 이름/시장/주식수 확보
        keep_cols = [c for c in ("Code", "Name", "Market", "Stocks") if c in df.columns]
        master = (
            df.groupby("Code", as_index=False)[keep_cols].last()
            if keep_cols
            else df[["Code"]].drop_duplicates()
        )
        if "Stocks" in master.columns:
            master["Stocks"] = pd.to_numeric(master["Stocks"], errors="coerce").fillna(
                0
            )
        return master

    def get_codes_and_shares_from_marcap(
        self, *, year: int | None = None
    ) -> tuple[list[str], dict[str, float]]:
        """marcap_repo의 parquet에서 종목코드/상장주식수(Stocks)를 추출합니다."""

        if not self.use_marcap:
            return [], {}

        master = self.get_master_from_marcap(year=year)
        if master.empty:
            return [], {}

        codes = master["Code"].astype(str).str.zfill(6).tolist()
        shares_map: dict[str, float] = {}
        if "Stocks" in master.columns:
            shares_map = {
                str(code).zfill(6): float(stocks)
                for code, stocks in zip(master["Code"], master["Stocks"])
            }
        return codes, shares_map

    def _prepare_marcap(self):
        """Marcap 라이브러리가 없으면 Git Clone 후 경로 추가"""
        # 이미 경로가 있으면 clone 생략
        if not os.path.exists(self.marcap_dir):
            git = shutil.which("git")
            if not git:
                raise RuntimeError(
                    "marcap_repo가 없고 git도 없습니다. git 설치 후 재시도하거나 --use-marcap false로 실행하세요."
                )
            logger.info("Marcap 저장소 클론 중...")
            # shell=False로 안전 실행
            import subprocess

            try:
                subprocess.run(
                    [
                        git,
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/FinanceData/marcap.git",
                        self.marcap_dir,
                    ],
                    check=True,
                )
            except Exception as e:
                raise RuntimeError(f"Marcap clone 실패: {e}") from e

        # 경로 주입
        if self.marcap_dir not in sys.path:
            sys.path.append(self.marcap_dir)

    def _get_marcap_module(self):
        import importlib

        try:
            return importlib.import_module("marcap")
        except Exception as exc:
            raise ImportError(
                "marcap 모듈을 불러올 수 없습니다. (clone 실패/경로 주입 실패/의존성 문제 가능)"
            ) from exc

    def _get_fdr_module(self):
        if self._fdr is not None:
            return self._fdr
        try:
            import FinanceDataReader as fdr
        except ImportError as exc:
            raise ImportError(
                "FinanceDataReader가 필요합니다. `pip install finance-datareader` 실행 후 재시도하세요."
            ) from exc
        self._fdr = fdr
        return fdr

    def _marcap_for_code(
        self,
        code: str,
        *,
        start_ts: pd.Timestamp | None = None,
        end_ts: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """코드 단위로 Marcap 데이터를 가져옵니다.

        1) marcap.marcap_data가 코드 필터 인자를 지원하면 코드 단위로 로드(메모리 안전)
        2) 지원하지 않으면 연도별로 청크 로드 후 code만 필터링(폴백)
        """

        if not self.use_marcap:
            return pd.DataFrame()

        code6 = str(code).zfill(6)

        eff_start = start_ts or self.start_ts
        eff_end = end_ts or self.end_ts

        start_year = int(pd.Timestamp(eff_start).year)
        end_year = int(min(pd.Timestamp(eff_end).year, 2025))
        chunks: list[pd.DataFrame] = []

        for y in range(start_year, end_year + 1):
            p = self._marcap_parquet_path(y)
            if not p.exists():
                continue
            try:
                df_y = pd.read_parquet(
                    p,
                    columns=[
                        "Date",
                        "Code",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "Amount",
                        "Marcap",
                    ],
                )
            except Exception:
                continue
            if df_y.empty or "Code" not in df_y.columns:
                continue
            df_y["Code"] = df_y["Code"].astype(str).str.strip().str.zfill(6)
            df_y = df_y[df_y["Code"] == code6]
            if df_y.empty:
                continue
            chunks.append(df_y)

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df[(df["Date"] >= eff_start) & (df["Date"] <= eff_end)]
        return df

    def _preload_marcap_groups(
        self, target_codes: list[str]
    ) -> dict[str, pd.DataFrame]:
        """Marcap 데이터를 1회 로딩해 코드별 DataFrame dict로 구성.

        메모리 사용량이 커질 수 있으므로 preload_marcap=True일 때만 사용.
        """

        if not self.use_marcap:
            return {}

        codes = {str(c).zfill(6) for c in target_codes}

        start_year = int(self.start_ts.year)
        end_year = int(min(self.end_ts.year, 2025))

        groups: dict[str, list[pd.DataFrame]] = {}
        for y in range(start_year, end_year + 1):
            p = self._marcap_parquet_path(y)
            if not p.exists():
                continue
            try:
                df_y = pd.read_parquet(
                    p,
                    columns=[
                        "Date",
                        "Code",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "Amount",
                        "Marcap",
                    ],
                )
            except Exception:
                continue
            if df_y.empty or "Code" not in df_y.columns:
                continue

            df_y["Code"] = df_y["Code"].astype(str).str.strip().str.zfill(6)
            df_y = df_y[df_y["Code"].isin(codes)]
            if df_y.empty:
                continue

            for code, df_code in df_y.groupby("Code"):
                groups.setdefault(str(code), []).append(df_code)

        merged: dict[str, pd.DataFrame] = {}
        for code, parts in groups.items():
            merged[code] = pd.concat(parts, ignore_index=True)

        return merged

    def _get_target_list(self):
        """KRX 종목코드/상장주식수 확보 (기본: marcap, 옵션: FDR)."""

        if self.use_marcap:
            master = self.get_master_from_marcap()
            if not master.empty and "Code" in master.columns:
                codes = master["Code"].astype(str).str.zfill(6).tolist()
                shares_map: dict[str, float] = {}
                if "Stocks" in master.columns:
                    shares_map = {
                        str(code).zfill(6): float(stocks)
                        for code, stocks in zip(master["Code"], master["Stocks"])
                    }

                # 코드-이름 매핑 저장(StockDownloader 단독 실행 시에도 남기기)
                try:
                    list_path = os.path.join(
                        self.base_dir,
                        f"krx_master_{datetime.now().strftime('%Y%m%d')}.csv",
                    )
                    cols = [
                        c
                        for c in ("Code", "Name", "Market", "Stocks")
                        if c in master.columns
                    ]
                    master[cols].to_csv(list_path, index=False, encoding="utf-8-sig")
                except Exception:
                    pass

                logger.info(f"KRX 종목코드 확보 완료 (marcap, {len(codes)}개)")
                return codes, shares_map

        if self.use_fdr:
            fdr = self._get_fdr_module()
            stocks = fdr.StockListing("KRX")
            stocks["Code"] = stocks["Code"].astype(str).str.zfill(6)
            stock_col = next(
                (c for c in stocks.columns if "Stocks" in c or "상장주식수" in c), None
            )
            shares_map = (
                stocks.set_index("Code")[stock_col].to_dict() if stock_col else {}
            )
            try:
                list_path = os.path.join(
                    self.base_dir, f"krx_master_{datetime.now().strftime('%Y%m%d')}.csv"
                )
                cols = [
                    c
                    for c in ("Code", "Name", "Market", "Stocks")
                    if c in stocks.columns
                ]
                if cols:
                    stocks[cols].to_csv(list_path, index=False, encoding="utf-8-sig")
            except Exception:
                pass
            logger.info(f"KRX 종목코드 확보 완료 (FDR, {len(stocks)}개)")
            return stocks["Code"].tolist(), shares_map

        return [], {}

    def process_stock(
        self, code, shares_map, marcap_groups: dict[str, pd.DataFrame] | None = None
    ):
        """개별 종목 처리 로직 (Marcap + FDR 병합)"""
        save_path = os.path.join(self.base_dir, f"{code}.parquet")

        if (not self.overwrite) and os.path.exists(save_path):
            return "Skip"

        try:
            df = self.fetch_ohlcv(
                code, shares_map=shares_map, marcap_groups=marcap_groups
            )
            if df.empty:
                return "Empty"

            # 저장 포맷: date 컬럼 포함
            if isinstance(df.index, pd.DatetimeIndex):
                df_to_save = df.reset_index()
            else:
                df_to_save = df
            df_to_save.to_parquet(save_path, index=False)
            return "Success"
        except Exception as e:
            return f"Error: {str(e)}"

    def fetch_ohlcv(
        self,
        code: str,
        *,
        shares_map: Dict[str, float] | None = None,
        marcap_groups: dict[str, pd.DataFrame] | None = None,
        start_ts: pd.Timestamp | None = None,
        end_ts: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """단일 종목의 (Marcap + FDR) OHLCV를 DataFrame으로 반환."""

        shares_map = shares_map or {}
        eff_start = start_ts or self.start_ts
        eff_end = end_ts or self.end_ts

        # (A) Marcap 데이터(가능하면 코드 단위로 로드)
        if marcap_groups is not None:
            df_m = marcap_groups.get(str(code).zfill(6), pd.DataFrame())
        else:
            df_m = self._marcap_for_code(code, start_ts=eff_start, end_ts=eff_end)

        # (B) FDR 데이터: Marcap 마지막 날짜 + 1일부터 end_date까지 자동 연결
        df_f = pd.DataFrame()
        if self.use_fdr:
            fdr = self._get_fdr_module()

            fdr_start = eff_start
            if not df_m.empty:
                if "Date" in df_m.columns:
                    last_m = pd.to_datetime(df_m["Date"], errors="coerce").max()
                else:
                    last_m = pd.to_datetime(df_m.index, errors="coerce").max()
                if pd.notna(last_m):
                    fdr_start = pd.Timestamp(last_m) + pd.Timedelta(days=1)

            try:
                df_f = fdr.DataReader(code, fdr_start, eff_end)
            except Exception as exc:
                # FDR은 네트워크/소스 이슈가 잦으므로, marcap 데이터만이라도 저장되게 폴백
                logger.debug(f"FDR 보강 실패({code}): {exc}")
                df_f = pd.DataFrame()

        if not df_f.empty:
            df_f = df_f.reset_index()
            df_f["Amount"] = df_f["Close"] * df_f["Volume"]
            shares = shares_map.get(str(code).zfill(6), shares_map.get(code, 0))
            df_f["Marcap"] = df_f["Close"] * (shares if pd.notna(shares) else 0)
            df_f["Code"] = str(code).zfill(6)

            cols_needed = [
                "Date",
                "Code",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Amount",
                "Marcap",
            ]
            for c in cols_needed:
                if c not in df_f.columns:
                    df_f[c] = 0
            df_f = df_f[cols_needed]

        # (C) 병합
        cols = [
            "Date",
            "Code",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Amount",
            "Marcap",
        ]
        if not df_m.empty:
            for c in cols:
                if c not in df_m.columns:
                    df_m[c] = pd.NA

            if not df_f.empty:
                last_m_date = pd.to_datetime(df_m["Date"], errors="coerce").max()
                if pd.notna(last_m_date):
                    df_f = df_f[
                        pd.to_datetime(df_f["Date"], errors="coerce") > last_m_date
                    ]
                df_final = pd.concat([df_m[cols], df_f], ignore_index=True)
            else:
                df_final = df_m[cols]
        else:
            df_final = df_f

        if df_final.empty:
            return pd.DataFrame()

        df_final = df_final.sort_values("Date").reset_index(drop=True)

        # (D) 전처리: 표준 컬럼/인덱스/결측 처리
        if self.preprocess:
            df_final = preprocess_ohlcv(df_final)
        return df_final

    def run(self):
        """전체 프로세스 실행"""
        # 1. 타겟 리스트 확보
        target_codes, shares_map = self._get_target_list()

        logger.info(f"🔥 총 {len(target_codes)}개 종목 변환 시작...")

        # 3. 루프 실행
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover
            tqdm = None  # type: ignore

        def _tqdm(iterable, **kwargs):
            if tqdm is None:
                return iterable
            return tqdm(iterable, **kwargs)

        results = []
        workers = max(1, int(self.max_workers))

        marcap_groups: dict[str, pd.DataFrame] | None = None
        if self.use_marcap and self.preload_marcap:
            logger.info("Marcap 데이터 사전 로딩 시작(메모리 사용량 증가 가능)")
            marcap_groups = self._preload_marcap_groups(target_codes)
            logger.info("Marcap 데이터 사전 로딩 완료")

        # 네트워크 요청(FDR) 병렬화: 너무 크게 잡으면 차단/레이트리밋 위험
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(self.process_stock, code, shares_map, marcap_groups): code
                for code in target_codes
            }
            for fut in _tqdm(as_completed(futures), total=len(futures)):
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append(f"Error: {str(e)}")

        logger.info("작업 완료!")
        logger.info(f"성공: {results.count('Success')}, 스킵: {results.count('Skip')}")
        logger.info(
            f"데이터없음: {results.count('Empty')}, 에러: {sum(1 for r in results if r.startswith('Error'))}"
        )


class UnifiedDownloader:
    def __init__(self, config: Dict[str, Any]) -> None:
        cfg = config
        self.data_cfg = cfg.get("data", cfg)
        self.paths_cfg = cfg.get("paths", {})
        self.date_cfg = cfg.get("date", {})
        self.download_cfg = cfg.get("download", {})
        self.market_cfg = cfg.get("markets", {})
        self.lists_cfg = cfg.get("lists", {})
        self.sources_cfg = cfg.get("sources", {})

        self.start_date = self.date_cfg.get("start", "2005-01-03")
        self.end_date = self.date_cfg.get("end") or datetime.now().strftime("%Y-%m-%d")

        self.output_mode = str(self.data_cfg.get("output_mode", "raw")).lower()
        self.preprocess = bool(self.data_cfg.get("preprocess", True))

        # list_only: 다운로드/저장 없이(로컬 parquet 기준) 마스터 CSV만 갱신
        self.refresh_master = bool(self.lists_cfg.get("refresh_master", True))

        self.use_marcap = bool(self.sources_cfg.get("use_marcap", True))
        self.use_fdr = bool(self.sources_cfg.get("use_fdr", True))
        self.preload_marcap = bool(self.sources_cfg.get("preload_marcap", True))

        self._fdr = None

    def _is_kr_kind(self, kind: str) -> bool:
        return kind in ("kr_stocks", "kr_etf")

    def _filter_kr_codes(self, codes: Iterable[str]) -> list[str]:
        items = [str(c).strip() for c in codes if str(c).strip()]
        s = pd.Series([c.zfill(6) for c in items])
        valid = s.str.match(r"^\d{6}$", na=False)
        return list(dict.fromkeys(s[valid].tolist()))

    def _filter_df_by_codes(
        self,
        df: pd.DataFrame,
        code_col: str,
        codes: Iterable[str],
        *,
        pad_kr: bool = True,
    ) -> pd.DataFrame:
        if df is None or df.empty or code_col not in df.columns:
            return pd.DataFrame()
        base = df.copy()
        s = base[code_col].astype(str).str.strip()
        if pad_kr:
            s = s.str.zfill(6)
            allow = set(self._filter_kr_codes(codes))
        else:
            allow = set(str(c).strip() for c in codes if str(c).strip())
        base[code_col] = s
        return base[base[code_col].isin(allow)]

    def _filter_df_by_valid_kr(self, df: pd.DataFrame, code_col: str) -> pd.DataFrame:
        if df is None or df.empty or code_col not in df.columns:
            return pd.DataFrame()
        s = df[code_col].astype(str).str.strip().str.zfill(6)
        allow = set(self._filter_kr_codes(s.tolist()))
        base = df.copy()
        base[code_col] = s
        return base[base[code_col].isin(allow)]

    def _filter_shares_map(
        self, shares_map: Dict[str, float], allow_codes: Iterable[str]
    ) -> Dict[str, float]:
        allow = set(allow_codes)
        return {
            str(k).zfill(6): float(v)
            for k, v in (shares_map or {}).items()
            if str(k).zfill(6) in allow
        }

    def _collect_local_codes(self, *, kind: str) -> list[str]:
        """로컬에 이미 저장된 parquet 파일명 기준으로 코드 리스트 생성."""

        candidates: list[str] = []
        for processed in (False, True):
            folder = self._resolve_dir(kind, processed=processed)
            try:
                p = Path(folder)
                if not p.exists():
                    continue
                for f in p.glob("*.parquet"):
                    candidates.append(f.stem)
            except Exception:
                continue

        if not candidates:
            return []

        codes = [str(c).strip() for c in candidates if str(c).strip()]
        if self._is_kr_kind(kind):
            codes = self._filter_kr_codes(codes)
        # 순서 유지 unique
        return list(dict.fromkeys(codes))

    def _refresh_kr_etf_master_from_local(self) -> None:
        if not self.refresh_master:
            return
        local_codes = self._collect_local_codes(kind="kr_etf")
        if not local_codes:
            logger.warning("로컬 kr_etf parquet이 없어 마스터를 갱신하지 않았습니다.")
            return

        if not self.use_fdr:
            logger.warning(
                "sources.use_fdr=false라 KR ETF 마스터(이름/카테고리) 갱신을 건너뜁니다."
            )
            return

        fdr = self._get_fdr()
        listing = fdr.StockListing("ETF/KR")
        if listing is None or getattr(listing, "empty", True):
            logger.warning("FDR ETF/KR listing이 비어 마스터를 갱신하지 않았습니다.")
            return

        df = self._filter_df_by_codes(listing, "Symbol", local_codes, pad_kr=True)
        self._save_etf_master(df, us=False)
        logger.info(f"KR ETF 마스터 갱신 완료(로컬 기준): {len(df)}개")

    def _refresh_krx_master_from_local(self) -> None:
        if not self.refresh_master:
            return
        local_codes = self._collect_local_codes(kind="kr_stocks")
        if not local_codes:
            logger.warning(
                "로컬 kr_stocks parquet이 없어 마스터를 갱신하지 않았습니다."
            )
            return

        allow = set(local_codes)

        # 1) marcap master 우선 (Code/Name/Market/Stocks)
        if self.use_marcap:
            try:
                downloader = StockDownloader(
                    base_dir=self._resolve_dir("kr_stocks", processed=False),
                    start_date=self.start_date,
                    end_date=self.end_date,
                    preprocess=False,
                    overwrite=False,
                    max_workers=1,
                    use_marcap=True,
                    use_fdr=False,
                    preload_marcap=False,
                )
                master = downloader.get_master_from_marcap()
                if master is not None and not master.empty and "Code" in master.columns:
                    master = self._filter_df_by_codes(
                        master, "Code", local_codes, pad_kr=True
                    )
                    self._save_krx_master(master)
                    logger.info(
                        f"KR 주식 마스터 갱신 완료(로컬 기준, marcap): {len(master)}개"
                    )
                    return
            except Exception as exc:
                logger.warning(f"marcap 기반 마스터 갱신 실패: {exc}")

        # 2) FDR listing 폴백 (Code/Name 정도)
        if self.use_fdr:
            try:
                fdr = self._get_fdr()
                stocks = fdr.StockListing(self.market_cfg.get("kr", "KRX"))
                if stocks is None or getattr(stocks, "empty", True):
                    return

                df = stocks.copy()
                if "Code" not in df.columns:
                    return
                df = self._filter_df_by_codes(df, "Code", local_codes, pad_kr=True)
                # save schema normalize
                out = pd.DataFrame({"Code": df["Code"]})
                name_col = (
                    "Name"
                    if "Name" in df.columns
                    else ("회사명" if "회사명" in df.columns else None)
                )
                if name_col:
                    out["Name"] = df[name_col]
                self._save_krx_master(out)
                logger.info(f"KR 주식 마스터 갱신 완료(로컬 기준, FDR): {len(out)}개")
            except Exception as exc:
                logger.warning(f"FDR 기반 마스터 갱신 실패: {exc}")

    def _get_fdr(self):
        if self._fdr is not None:
            return self._fdr
        try:
            import FinanceDataReader as fdr
        except ImportError as exc:
            raise ImportError(
                "FinanceDataReader가 필요합니다. `pip install finance-datareader` 실행 후 재시도하세요."
            ) from exc
        self._fdr = fdr
        return fdr

    def _resolve_dir(self, kind: str, processed: bool) -> str:
        group = "processed" if processed else "raw"
        path_map = self.paths_cfg.get(group) or {}
        if kind in path_map:
            path = str(path_map[kind])
            if not os.path.isabs(path):
                base_dir = self.data_cfg.get("base_dir", "data")
                return os.path.join(base_dir, path)
            return path
        base_dir = self.data_cfg.get("base_dir", "data")
        folder = "processed" if processed else "raw"
        return os.path.join(base_dir, folder, kind)

    def _save_df(self, df: pd.DataFrame, code: str, *, kind: str) -> None:
        if df.empty:
            return

        # list_only 모드에서는 parquet 저장을 하지 않음
        if self.output_mode == "list_only":
            return

        incremental = bool(self.data_cfg.get("incremental", True))

        # raw 저장
        if self.output_mode in ("raw", "both"):
            raw_dir = self._resolve_dir(kind, processed=False)
            os.makedirs(raw_dir, exist_ok=True)
            raw_path = os.path.join(raw_dir, f"{str(code).zfill(6)}.parquet")
            df_raw = df.copy()
            if isinstance(df_raw.index, pd.DatetimeIndex):
                df_raw = df_raw.reset_index()

            if incremental and os.path.exists(raw_path):
                try:
                    exist = pd.read_parquet(raw_path)
                    # 날짜 컬럼 통일
                    if "date" in exist.columns and "Date" not in exist.columns:
                        exist = exist.rename(columns={"date": "Date"})
                    if "date" in df_raw.columns and "Date" not in df_raw.columns:
                        df_raw = df_raw.rename(columns={"date": "Date"})
                    merged = pd.concat([exist, df_raw], ignore_index=True)
                    if "Date" in merged.columns:
                        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
                        merged = merged.dropna(subset=["Date"]).sort_values("Date")
                        merged = merged.drop_duplicates(subset=["Date"], keep="last")
                    df_raw = merged
                except Exception:
                    pass
            df_raw.to_parquet(raw_path, index=False)

        # processed 저장
        if self.output_mode in ("processed", "both"):
            processed_dir = self._resolve_dir(kind, processed=True)
            os.makedirs(processed_dir, exist_ok=True)
            processed_path = os.path.join(
                processed_dir, f"{str(code).zfill(6)}.parquet"
            )
            df_proc = preprocess_ohlcv(df) if self.preprocess else df.copy()
            if isinstance(df_proc.index, pd.DatetimeIndex):
                df_proc = df_proc.reset_index()

            if incremental and os.path.exists(processed_path):
                try:
                    exist = pd.read_parquet(processed_path)
                    merged = pd.concat([exist, df_proc], ignore_index=True)
                    if "date" in merged.columns:
                        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
                        merged = merged.dropna(subset=["date"]).sort_values("date")
                        merged = merged.drop_duplicates(subset=["date"], keep="last")
                    df_proc = merged
                except Exception:
                    pass
            df_proc.to_parquet(processed_path, index=False)

    def _save_krx_master(self, master: pd.DataFrame) -> None:
        if master is None or master.empty:
            return

        cols = [c for c in ("Code", "Name", "Market", "Stocks") if c in master.columns]
        if not cols:
            return

        df = master.copy()
        df["Code"] = df["Code"].astype(str).str.zfill(6)
        self._write_master_csv(df[cols], prefix="krx_master")

    def _write_master_csv(self, payload: pd.DataFrame, *, prefix: str) -> None:
        if payload is None or payload.empty:
            return

        base_dir = self.data_cfg.get("base_dir", "data")
        out_dir = Path(base_dir) / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)

        latest = out_dir / f"{prefix}_latest.csv"
        dated = out_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d')}.csv"

        for path in (latest, dated):
            try:
                payload.to_csv(path, index=False, encoding="utf-8-sig")
            except PermissionError as exc:
                alt = (
                    out_dir
                    / f"{path.stem}_{datetime.now().strftime('%H%M%S')}{path.suffix}"
                )
                logger.warning(
                    f"master CSV 저장 실패(잠금 가능): {path} ({exc}). 대체 파일로 저장합니다: {alt}"
                )
                payload.to_csv(alt, index=False, encoding="utf-8-sig")
            except Exception as exc:
                logger.warning(f"master CSV 저장 실패: {path} ({exc})")

    def _save_etf_master(self, master: pd.DataFrame, *, us: bool) -> None:
        if master is None or master.empty:
            return
        df = master.copy()
        if "Symbol" in df.columns:
            if not us:
                df = self._filter_df_by_valid_kr(df, "Symbol")
            else:
                df["Symbol"] = df["Symbol"].astype(str).str.strip()

        # 가능한 컬럼만 저장 (FDR 버전별 차이 대응)
        cols = [c for c in ("Symbol", "Name", "Category", "MarCap") if c in df.columns]
        if not cols:
            cols = list(df.columns)
        prefix = f"{'us_etf' if us else 'kr_etf'}_master"
        self._write_master_csv(df[cols], prefix=prefix)

    def _plan_incremental(
        self, *, codes: List[str], kind: str
    ) -> dict[str, str | None]:
        """코드별로 어느 시점부터 다시 받을지(start_date override)를 계산합니다.

        - None: 전체 범위 필요(파일 없음)
        - "SKIP": 이미 범위 충족
        - "YYYY-MM-DD": 해당 날짜부터 증분 수집
        """

        incremental = bool(self.data_cfg.get("incremental", True))
        skip_if_covered = bool(self.data_cfg.get("skip_if_covered", True))

        start_ts = pd.to_datetime(self.start_date)
        end_ts = pd.to_datetime(self.end_date)

        # 어떤 파일을 기준으로 스킵 판단할지: raw 우선
        raw_dir = self._resolve_dir(kind, processed=False)
        processed_dir = self._resolve_dir(kind, processed=True)

        plan: dict[str, str | None] = {}
        for code in codes:
            code6 = str(code).zfill(6)
            raw_path = os.path.join(raw_dir, f"{code6}.parquet")
            proc_path = os.path.join(processed_dir, f"{code6}.parquet")

            target_path = raw_path if os.path.exists(raw_path) else proc_path
            if not os.path.exists(target_path):
                plan[code6] = None
                continue

            s, e = _get_parquet_date_range(target_path)
            if s is None or e is None:
                plan[code6] = None
                continue

            if skip_if_covered and s <= start_ts and e >= end_ts:
                plan[code6] = "SKIP"
                continue

            if incremental and e < end_ts:
                inc_start = (pd.Timestamp(e) + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
                plan[code6] = inc_start
                continue

            plan[code6] = None

        return plan

    def _download_codes(
        self,
        codes: List[str],
        *,
        kind: str,
        fetch_fn,
        shares_map: Dict[str, float] | None = None,
        max_workers: int = 8,
    ) -> None:
        if not codes:
            return
        shares_map = shares_map or {}

        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover
            tqdm = None  # type: ignore

        def _tqdm(iterable, **kwargs):
            if tqdm is None:
                return iterable
            return tqdm(iterable, **kwargs)

        workers = max(1, int(max_workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch_fn, code, shares_map): code for code in codes}
            for fut in _tqdm(as_completed(futures), total=len(futures)):
                code = futures[fut]
                try:
                    df = fut.result()
                    self._save_df(df, code, kind=kind)
                except Exception as e:
                    logger.warning(f"{kind} {code} 실패: {e}")

    def _get_krx_list(self) -> Tuple[List[str], Dict[str, float]]:
        use_list = bool(self.lists_cfg.get("use_list_stocks", False))
        if use_list:
            csv_path = self.lists_cfg.get("stocks_csv")
            codes, shares = _read_csv_codes(
                csv_path,
                code_cols=("종목코드", "Code", "Ticker", "Symbol"),
                shares_cols=("상장주식수", "Stocks", "상장좌수", "Shares"),
                market_filter=("KRX", "KOSPI", "KOSDAQ"),
                pad_code=True,
            )
            if codes:
                filtered_codes = self._filter_kr_codes(codes)
                shares = self._filter_shares_map(shares, filtered_codes)
                return filtered_codes, shares

        # 기본 경로: marcap(로컬 parquet)로 종목코드 확보 (네트워크 불필요)
        if self.use_marcap:
            try:
                downloader = StockDownloader(
                    base_dir=self._resolve_dir("kr_stocks", processed=False),
                    start_date=self.start_date,
                    end_date=self.end_date,
                    preprocess=False,
                    overwrite=False,
                    max_workers=1,
                    use_marcap=True,
                    use_fdr=False,
                    preload_marcap=False,
                )
                master = downloader.get_master_from_marcap()
                if self.refresh_master:
                    self._save_krx_master(master)
                codes, shares_map = downloader.get_codes_and_shares_from_marcap()
                if codes:
                    codes = self._filter_kr_codes(codes)
                    shares_map = self._filter_shares_map(shares_map, codes)
                    logger.info(f"KRX 종목코드 확보 완료 (marcap, {len(codes)}개)")
                    return codes, shares_map
            except Exception as exc:
                logger.warning(f"marcap에서 종목코드 확보 실패: {exc}")

        # 옵션: FDR listing (원하면 config에 sources.use_fdr_listing: true)
        allow_fdr_listing = bool(self.sources_cfg.get("use_fdr_listing", False))
        if not allow_fdr_listing:
            # 마지막 폴백: 저장된 CSV가 있으면 사용
            fallback_csv = _latest_krx_list_csv(self.data_cfg.get("base_dir", "data"))
            if fallback_csv:
                codes, shares = _read_csv_codes(
                    fallback_csv,
                    code_cols=("종목코드", "Code", "Ticker", "Symbol"),
                    shares_cols=("상장주식수", "Stocks", "상장좌수", "Shares"),
                    market_filter=("KRX", "KOSPI", "KOSDAQ"),
                    pad_code=True,
                )
                if codes:
                    filtered_codes = self._filter_kr_codes(codes)
                    shares = self._filter_shares_map(shares, filtered_codes)
                    logger.info(f"KRX 종목코드 확보 완료 (저장된 CSV, {len(codes)}개)")
                    return filtered_codes, shares

            logger.error(
                "KRX 종목코드 확보 실패: marcap/CSV 모두 사용할 수 없습니다. "
                "(필요 시 sources.use_fdr_listing=true로 FDR listing도 허용하세요.)"
            )
            return [], {}

        fdr = self._get_fdr()
        stocks = fdr.StockListing(self.market_cfg.get("kr", "KRX"))
        stocks = self._filter_df_by_valid_kr(stocks, "Code")
        stock_col = next(
            (c for c in stocks.columns if "Stocks" in c or "상장주식수" in c), None
        )
        shares_map = stocks.set_index("Code")[stock_col].to_dict() if stock_col else {}

        # 목록 자동 저장 (다음 실행 폴백용)
        if self.refresh_master:
            try:
                base_dir = self.data_cfg.get("base_dir", "data")
                list_dir = Path(base_dir) / "raw"
                list_dir.mkdir(parents=True, exist_ok=True)
                list_path = (
                    list_dir / f"krx_codes_{datetime.now().strftime('%Y%m%d')}.csv"
                )
                cols = ["Code"] + (["Name"] if "Name" in stocks.columns else [])
                stocks[cols].to_csv(list_path, index=False, encoding="utf-8-sig")
            except Exception:
                pass

        return stocks["Code"].tolist(), shares_map

    def _get_us_list(self) -> List[str]:
        use_list = bool(self.lists_cfg.get("use_list_stocks", False))
        if use_list:
            csv_path = self.lists_cfg.get("stocks_csv")
            codes, _ = _read_csv_codes(
                csv_path,
                code_cols=("Ticker", "Symbol", "Code"),
                shares_cols=("Shares", "Stocks", "상장주식수"),
                market_filter=("NASDAQ", "NYSE", "AMEX", "US"),
                pad_code=False,
            )
            if codes:
                return codes

        fdr = self._get_fdr()
        markets = self.market_cfg.get("us", ["NASDAQ", "NYSE", "AMEX"])
        tickers: List[str] = []
        for m in markets:
            try:
                listing = fdr.StockListing(m)
            except Exception:
                continue
            col = "Symbol" if "Symbol" in listing.columns else "Code"
            tickers.extend(listing[col].astype(str).tolist())
        return list(dict.fromkeys(tickers))

    def _get_etf_list(self, *, us: bool) -> Tuple[List[str], Dict[str, float]]:
        use_list = bool(self.lists_cfg.get("use_list_etf", True))
        if use_list:
            csv_path = self.lists_cfg.get("etf_csv")
            codes, shares = _read_csv_codes(
                csv_path,
                code_cols=("종목코드", "Code", "Ticker", "Symbol"),
                shares_cols=("상장좌수", "상장주식수", "Stocks", "Shares"),
                market_filter=(
                    ("NASDAQ", "NYSE", "AMEX", "US")
                    if us
                    else ("KRX", "KOSPI", "KOSDAQ")
                ),
                pad_code=(not us),
            )
            # CSV에 이상한 코드(알파뉴메릭)가 섞여 있을 수 있어 KR은 6자리 숫자만 통과
            if not us and codes:
                filtered_codes = self._filter_kr_codes(codes)
                shares = self._filter_shares_map(shares, filtered_codes)
                return filtered_codes, shares
            return codes, shares

        # 리스트를 쓰지 않는 경우: FDR의 ETF listing을 사용
        if not self.use_fdr:
            logger.error(
                "ETF 목록 확보 실패: lists.use_list_etf=false 이고 sources.use_fdr=false 입니다."
            )
            return [], {}

        market = "ETF/US" if us else "ETF/KR"
        try:
            fdr = self._get_fdr()
            listing = fdr.StockListing(market)
            if listing is None or getattr(listing, "empty", True):
                logger.error(
                    f"ETF 목록 확보 실패: FDR StockListing('{market}') 결과가 비어있습니다."
                )
                return [], {}

            code_col = (
                "Symbol"
                if "Symbol" in listing.columns
                else ("Code" if "Code" in listing.columns else None)
            )
            if code_col is None:
                logger.error(
                    f"ETF 목록 확보 실패: FDR listing 컬럼에 Symbol/Code가 없습니다. columns={list(listing.columns)}"
                )
                return [], {}

            df = listing.copy()
            df[code_col] = df[code_col].astype(str).str.strip()

            if not us:
                # KR ETF는 KRX 6자리 숫자 코드만 사용 (알파뉴메릭 심볼은 제외)
                df = self._filter_df_by_valid_kr(df, code_col)

            codes = df[code_col].astype(str).tolist()
            codes = list(dict.fromkeys([c for c in codes if c]))

            if self.refresh_master:
                self._save_etf_master(df, us=us)
            logger.info(f"ETF 목록 확보 완료 (FDR {market}, {len(codes)}개)")
            return codes, {}
        except Exception as exc:
            logger.error(f"ETF 목록 확보 실패 (FDR {market}): {exc}")
            return [], {}

    def _fetch_kr(self, code: str, shares_map: Dict[str, float]) -> pd.DataFrame:
        downloader = StockDownloader(
            base_dir=self._resolve_dir("kr_stocks", processed=False),
            start_date=self.start_date,
            end_date=self.end_date,
            preprocess=False,
            overwrite=False,
            max_workers=1,
            use_marcap=self.use_marcap,
            use_fdr=self.use_fdr,
            preload_marcap=self.preload_marcap,
        )
        return downloader.fetch_ohlcv(code, shares_map=shares_map)

    def _fetch_us(self, code: str, _: Dict[str, float]) -> pd.DataFrame:
        fdr = self._get_fdr()
        df = fdr.DataReader(code, self.start_date, self.end_date)
        return _normalize_fdr_df(df, code, shares=None)

    def _fetch_kr_etf(self, code: str, shares_map: Dict[str, float]) -> pd.DataFrame:
        fdr = self._get_fdr()
        df = fdr.DataReader(code, self.start_date, self.end_date)
        shares = shares_map.get(str(code).zfill(6), shares_map.get(code, 0))
        return _normalize_fdr_df(df, code, shares=shares)

    def _fetch_us_etf(self, code: str, _: Dict[str, float]) -> pd.DataFrame:
        fdr = self._get_fdr()
        df = fdr.DataReader(code, self.start_date, self.end_date)
        return _normalize_fdr_df(df, code, shares=None)

    def run(self, *, max_workers: int = 8) -> None:
        # output_mode=list_only: 로컬 parquet에 존재하는 코드만 기준으로 마스터 CSV 덮어쓰기
        if self.output_mode == "list_only":
            self._refresh_krx_master_from_local()
            self._refresh_kr_etf_master_from_local()
            logger.info("output_mode=list_only: 다운로드는 수행하지 않았습니다.")
            return

        if self.download_cfg.get("kr_stocks", False):
            codes, shares = self._get_krx_list()
            if codes:
                plan = self._plan_incremental(
                    codes=[str(c).zfill(6) for c in codes], kind="kr_stocks"
                )
                codes_to_run = [c for c in codes if plan.get(str(c).zfill(6)) != "SKIP"]

                kr_downloader = StockDownloader(
                    base_dir=self._resolve_dir("kr_stocks", processed=False),
                    start_date=self.start_date,
                    end_date=self.end_date,
                    preprocess=False,
                    overwrite=False,
                    max_workers=1,
                    use_marcap=self.use_marcap,
                    use_fdr=self.use_fdr,
                    preload_marcap=self.preload_marcap,
                )

                marcap_groups: dict[str, pd.DataFrame] | None = None
                if self.use_marcap and self.preload_marcap:
                    logger.info("Marcap 데이터 사전 로딩 시작(메모리 사용량 증가 가능)")
                    marcap_groups = kr_downloader._preload_marcap_groups(
                        [str(c).zfill(6) for c in codes_to_run]
                    )
                    logger.info("Marcap 데이터 사전 로딩 완료")

                def _fetch_kr_shared(
                    code: str, shares_map: Dict[str, float]
                ) -> pd.DataFrame:
                    code6 = str(code).zfill(6)
                    start_override = plan.get(code6)
                    if start_override and start_override != "SKIP":
                        return kr_downloader.fetch_ohlcv(
                            code6,
                            shares_map=shares_map,
                            marcap_groups=marcap_groups,
                            start_ts=pd.to_datetime(start_override),
                            end_ts=pd.to_datetime(self.end_date),
                        )
                    return kr_downloader.fetch_ohlcv(
                        code,
                        shares_map=shares_map,
                        marcap_groups=marcap_groups,
                    )

                self._download_codes(
                    codes_to_run,
                    kind="kr_stocks",
                    fetch_fn=_fetch_kr_shared,
                    shares_map=shares,
                    max_workers=max_workers,
                )

        if self.download_cfg.get("us_stocks", False):
            codes = self._get_us_list()
            self._download_codes(
                codes,
                kind="us_stocks",
                fetch_fn=self._fetch_us,
                shares_map={},
                max_workers=max_workers,
            )

        if self.download_cfg.get("kr_etf", False):
            codes, shares = self._get_etf_list(us=False)
            if not codes:
                logger.warning(
                    "kr_etf 다운로드가 활성화됐지만 ETF 목록이 비어있습니다. "
                    "(lists.use_list_etf=false면 FDR ETF/KR listing이 필요합니다.)"
                )
            self._download_codes(
                codes,
                kind="kr_etf",
                fetch_fn=self._fetch_kr_etf,
                shares_map=shares,
                max_workers=max_workers,
            )

        if self.download_cfg.get("us_etf", False):
            codes, shares = self._get_etf_list(us=True)
            if not codes:
                logger.warning(
                    "us_etf 다운로드가 활성화됐지만 ETF 목록이 비어있습니다. "
                    "(lists.use_list_etf=false면 FDR ETF/US listing이 필요합니다.)"
                )
            self._download_codes(
                codes,
                kind="us_etf",
                fetch_fn=self._fetch_us_etf,
                shares_map=shares,
                max_workers=max_workers,
            )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Unified data downloader")
    parser.add_argument(
        "--config",
        type=str,
        default="data",
        help="config/{name}.yaml",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_data_config(args.config)
    ud = UnifiedDownloader(cfg)
    logger.info(
        f"base_dir={ud.data_cfg.get('base_dir')}, output_mode={ud.data_cfg.get('output_mode')}"
    )
    logger.info(f"download={ud.download_cfg}")
    ud.run(max_workers=int(args.max_workers))
