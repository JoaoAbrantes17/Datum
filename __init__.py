import logging
import time
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("datum")
logger.setLevel(logging.INFO)


class Datum:
    """
    Datum is the sole canonical gateway for historical market data.

    Responsibilities
    ----------------
    - Download raw OHLCV + corporate actions from Yahoo Finance
    - Clean, validate, and normalize all data
    - Explicitly adjust prices using:

        adjustment_factor_t = AdjClose_t / Close_t

      Applied to:
        Open, High, Low, Close

    - Enforce strict schema, typing, determinism, and auditability
    """

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
        retry_attempts: int = 3,
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

        self._raw: pd.DataFrame | None = None
        self._prices: pd.DataFrame | None = None

        self._metadata: Dict[str, Any] = {
            "coverage": {},
            "missing_data_actions": [],
            "corporate_actions": {},
            "adjustment_checks": {},
        }

        self._load()

    # ------------------------------------------------------------------
    # DATA ACQUISITION
    # ------------------------------------------------------------------
    def _load(self) -> None:
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

                if data.empty:
                    raise ValueError("Empty download from Yahoo Finance")

                self._raw = data
                break

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    {
                        "event": "download_failure",
                        "attempt": attempt,
                        "error": str(exc),
                    }
                )
                time.sleep(1)

        if self._raw is None:
            raise RuntimeError("Failed to download market data") from last_exc

        self._process()

    # ------------------------------------------------------------------
    # PROCESSING PIPELINE
    # ------------------------------------------------------------------
    def _process(self) -> None:
        frames: list[pd.DataFrame] = []

        for ticker in self.tickers:
            if ticker not in self._raw.columns.get_level_values(0):
                logger.warning(
                    {"event": "missing_ticker", "ticker": ticker}
                )
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
                logger.warning(
                    {
                        "event": "zero_volume_days",
                        "ticker": ticker,
                        "count": zero_vol,
                    }
                )

            # explicit adjustment
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
                    self._metadata["missing_data_actions"].append(
                        {"ticker": ticker, "action": "forward_fill"}
                    )

                if self.dropna_policy == "interpolate":
                    adjusted = adjusted.interpolate(method="time")
                    self._metadata["missing_data_actions"].append(
                        {"ticker": ticker, "action": "interpolate"}
                    )

            # calendar alignment
            if self.calendar == "business":
                idx = pd.date_range(
                    adjusted.index.min(),
                    adjusted.index.max(),
                    freq="B",
                    tz="UTC",
                )
                adjusted = adjusted.reindex(idx)

            adjusted.columns = pd.MultiIndex.from_product(
                [[ticker], adjusted.columns]
            )
            frames.append(adjusted)

            self._metadata["coverage"][ticker] = {
                "start": str(adjusted.index.min()),
                "end": str(adjusted.index.max()),
            }

            self._metadata["corporate_actions"][ticker] = {
                "dividends": float(df["Dividends"].sum()),
                "splits": int((df["Stock Splits"] != 0).sum()),
            }

            self._metadata["adjustment_checks"][ticker] = {
                "adj_close_match": bool(
                    np.allclose(
                        adjusted[(ticker, "close_adj")],
                        df["Adj Close"],
                        equal_nan=True,
                    )
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
