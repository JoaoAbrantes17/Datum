import logging
import time
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger("datum")
logger.setLevel(logging.INFO)


class Datum:
    """
    Datum is the sole canonical gateway for historical market data.

    Responsibilities
    ----------------
    - Download raw OHLCV + corporate actions (if available)
    - Clean, validate, and normalize all data
    - Explicitly adjust prices (where applicable) using:

        adjustment_factor_t = AdjClose_t / Close_t

      Applied to:
        Open, High, Low, Close

    - Enforce strict schema, typing, determinism, and auditability
    """

    BINANCE_BASE_URL = "https://data-api.binance.vision"  # public market-data base

    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str,
        freq: Literal["1d", "1wk", "1mo"],
        price_field: Literal["adj_close", "close"] = "adj_close",
        dropna_policy: Literal["strict", "forward", "interpolate"] = "strict",
        calendar: Literal["exchange", "business"] = "exchange",
        timezone: str = "UTC",
        retry_attempts: int = 1,
        asset_class: Literal["Equities", "Crypto"] = "Equities",
    ):
        self.tickers = sorted(set(tickers))
        self.start = start
        self.end = end
        self.freq = freq
        self.price_field = price_field
        self.dropna_policy = dropna_policy
        self.calendar = calendar
        self.timezone = timezone
        self.retry_attempts = retry_attempts
        self.asset_class = asset_class

        self._raw: pd.DataFrame | None = None
        self._prices: pd.DataFrame | None = None

        self._metadata: Dict[str, Any] = {
            "coverage": {},
            "missing_data_actions": [],
            "corporate_actions": {},
            "adjustment_checks": {},
            "source": asset_class,
        }

        self._load()

    # ------------------------------------------------------------------
    # DATA ACQUISITION
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.asset_class == "Equities":
            self._raw = self._download_yahoo()
        elif self.asset_class == "Crypto":
            self._raw = self._download_binance()
        else:
            raise ValueError("asset_class must be 'Equities' or 'Crypto'")

        if self._raw is None or self._raw.empty:
            raise RuntimeError("Failed to download market data")

        self._process()

    def _download_yahoo(self) -> pd.DataFrame:
        last_exc: Exception | None = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                data = yf.download(
                    tickers=self.tickers,
                    start=self.start,
                    end=self.end,
                    interval=self.freq,
                    auto_adjust=False,
                    group_by="ticker",
                    actions=True,
                    threads=False,
                )
                if data is None or data.empty:
                    raise ValueError("Empty download from Yahoo Finance")

                # Normalize to MultiIndex columns: (ticker, field)
                if not isinstance(data.columns, pd.MultiIndex):
                    # single-ticker case
                    if len(self.tickers) != 1:
                        raise RuntimeError("Unexpected Yahoo column format for multiple tickers")
                    t = self.tickers[0]
                    data.columns = pd.MultiIndex.from_product([[t], list(data.columns)])

                return data

            except Exception as exc:
                last_exc = exc
                logger.warning({"event": "download_failure", "source": "yahoo", "attempt": attempt, "error": str(exc)})
                time.sleep(1)

        raise RuntimeError("Failed to download market data from Yahoo Finance") from last_exc

    def _download_binance(self) -> pd.DataFrame:
        last_exc: Exception | None = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                frames: list[pd.DataFrame] = []
                for symbol in self.tickers:
                    df = self._binance_fetch_klines(symbol=symbol, interval=self._binance_interval(self.freq))
                    # Map into the same “raw” schema Yahoo provides
                    raw = pd.DataFrame(
                        {
                            "Open": df["open"],
                            "High": df["high"],
                            "Low": df["low"],
                            "Close": df["close"],
                            "Adj Close": df["close"],         # no corporate-action adj for crypto spot
                            "Volume": df["volume"],
                            "Dividends": 0.0,
                            "Stock Splits": 0.0,
                        },
                        index=df.index,
                    )
                    raw.columns = pd.MultiIndex.from_product([[symbol], list(raw.columns)])
                    frames.append(raw)

                if not frames:
                    raise RuntimeError("No crypto symbols returned data from Binance")

                return pd.concat(frames, axis=1).sort_index(axis=1)

            except Exception as exc:
                last_exc = exc
                logger.warning({"event": "download_failure", "source": "binance", "attempt": attempt, "error": str(exc)})
                time.sleep(1)

        raise RuntimeError("Failed to download market data from Binance") from last_exc

    def _binance_interval(self, freq: str) -> str:
        # Datum freq stays the same; only Binance needs mapping
        mapping = {"1d": "1d", "1wk": "1w", "1mo": "1M"}
        if freq not in mapping:
            raise ValueError(f"Unsupported freq for Binance: {freq}")
        return mapping[freq]

    def _binance_fetch_klines(self, symbol: str, interval: str) -> pd.DataFrame:
        start_ts = pd.Timestamp(self.start)
        end_ts = pd.Timestamp(self.end)

        # Make timestamps explicit UTC
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")

        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")

        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        url = f"{self.BINANCE_BASE_URL}/api/v3/klines"
        out: list[list[Any]] = []

        while start_ms < end_ms:
            r = requests.get(
                url,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=30,
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break

            out.extend(batch)
            start_ms = int(batch[-1][0]) + 1  # advance 1ms to avoid duplicates
            time.sleep(0.05)

        if not out:
            raise ValueError(f"Empty download from Binance for {symbol}")

        # Only keep the fields we actually need (avoid dtype issues from other columns)
        df = pd.DataFrame(out, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","n_trades",
            "taker_buy_base_volume","taker_buy_quote_volume","ignore",
        ])[["open_time", "open", "high", "low", "close", "volume"]]

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Check only the numeric fields we care about
        if df[["open", "high", "low", "close", "volume"]].isna().any().any():
            raise ValueError(f"NaNs after numeric coercion for {symbol} (Binance)")

        return df


    # ------------------------------------------------------------------
    # PROCESSING PIPELINE
    # ------------------------------------------------------------------
    def _process(self) -> None:
        frames: list[pd.DataFrame] = []

        for ticker in self.tickers:
            # _raw must be MultiIndex (ticker, field)
            if not isinstance(self._raw.columns, pd.MultiIndex) or ticker not in self._raw.columns.get_level_values(0):
                logger.warning({"event": "missing_ticker", "ticker": ticker})
                continue

            df = self._raw[ticker].copy()

            # index hygiene
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize(self.timezone)
            df.index = df.index.tz_convert("UTC")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]

            required = [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]
            for col in required:
                if col not in df.columns:
                    df[col] = np.nan

            df = df[required]
            df = df.apply(pd.to_numeric, errors="coerce")

            if any(df.dtypes == "object"):
                raise TypeError("Object dtype detected after coercion")

            # volume sanity checks
            if (df["Volume"] < 0).any():
                raise ValueError(f"Negative volume detected for {ticker}")

            zero_vol = int((df["Volume"] == 0).sum())
            if zero_vol > 0:
                logger.warning({"event": "zero_volume_days", "ticker": ticker, "count": zero_vol})

            # explicit adjustment (crypto will have Adj Close == Close)
            with np.errstate(divide="ignore", invalid="ignore"):
                adj_factor = df["Adj Close"] / df["Close"]

            adjusted = pd.DataFrame(
                {
                    "open_adj": df["Open"] * adj_factor,
                    "high_adj": df["High"] * adj_factor,
                    "low_adj": df["Low"] * adj_factor,
                    "close_adj": df["Close"] * adj_factor,
                    "volume": df["Volume"],
                    "dividends": df["Dividends"],
                    "splits": df["Stock Splits"],
                },
                index=df.index,
            )

            # missing data policy
            if adjusted.isna().any().any():
                if self.dropna_policy == "strict":
                    raise ValueError(f"Missing data detected for {ticker}")

                if self.dropna_policy == "forward":
                    adjusted = adjusted.ffill()
                    self._metadata["missing_data_actions"].append({"ticker": ticker, "action": "forward_fill"})

                if self.dropna_policy == "interpolate":
                    adjusted = adjusted.interpolate(method="time")
                    self._metadata["missing_data_actions"].append({"ticker": ticker, "action": "interpolate"})

            # calendar alignment
            if self.calendar == "business":
                idx = pd.date_range(adjusted.index.min(), adjusted.index.max(), freq="B", tz="UTC")
                adjusted = adjusted.reindex(idx)

            adjusted.columns = pd.MultiIndex.from_product([[ticker], adjusted.columns])
            frames.append(adjusted)

            self._metadata["coverage"][ticker] = {"start": str(adjusted.index.min()), "end": str(adjusted.index.max())}

            # crypto has no dividends/splits in this interface
            self._metadata["corporate_actions"][ticker] = {
                "dividends": float(df["Dividends"].sum(skipna=True)),
                "splits": int((df["Stock Splits"].fillna(0) != 0).sum()),
            }

            self._metadata["adjustment_checks"][ticker] = {
                "adj_close_match": bool(
                    np.allclose(adjusted[(ticker, "close_adj")], df["Adj Close"], equal_nan=True)
                )
            }

        if not frames:
            raise RuntimeError("No valid ticker data available")

        self._prices = pd.concat(frames, axis=1).sort_index(axis=1)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def get_prices(self) -> pd.DataFrame:
        if self._prices is None:
            raise RuntimeError("Prices not available")
        return self._prices.copy()

    def get_metadata(self) -> dict:
        return dict(self._metadata)
