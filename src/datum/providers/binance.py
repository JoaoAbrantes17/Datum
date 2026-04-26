import logging
import time
from typing import Any

import pandas as pd
import requests

from datum._types import FreqLiteral
from datum.exceptions import DatumFetchError
from datum.providers import BaseProvider, register

logger = logging.getLogger("datum")

_BINANCE_BASE_URL = "https://data-api.binance.vision"
_FREQ_MAP = {"1d": "1d", "1wk": "1w", "1mo": "1M"}


class BinanceProvider(BaseProvider):

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        freq: FreqLiteral,
        retry_attempts: int,
    ) -> pd.DataFrame:
        interval = _FREQ_MAP.get(freq)
        if interval is None:
            raise ValueError(f"Unsupported freq for Binance: {freq!r}. Supported: {list(_FREQ_MAP)}")

        last_exc: Exception | None = None

        for attempt in range(1, retry_attempts + 1):
            try:
                frames: list[pd.DataFrame] = []
                for symbol in tickers:
                    df = _fetch_klines(symbol, interval, start, end)
                    df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
                    frames.append(df)

                if not frames:
                    raise RuntimeError("No symbols returned data from Binance")

                return pd.concat(frames, axis=1).sort_index(axis=1)

            except Exception as exc:
                last_exc = exc
                logger.warning({"event": "download_failure", "source": "binance", "attempt": attempt, "error": str(exc)})
                time.sleep(min(2 ** attempt, 30))

        raise DatumFetchError("Failed to download market data from Binance") from last_exc


def _fetch_klines(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

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

    url = f"{_BINANCE_BASE_URL}/api/v3/klines"
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
        start_ms = int(batch[-1][0]) + 1
        time.sleep(0.05)

    if not out:
        raise ValueError(f"Empty download from Binance for {symbol}")

    df = pd.DataFrame(out, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ])[["open_time", "open", "high", "low", "close", "volume"]]

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError(f"NaNs after numeric coercion for {symbol} (Binance)")

    return df


register("binance", BinanceProvider)
