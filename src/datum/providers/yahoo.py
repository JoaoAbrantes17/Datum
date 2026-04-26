import logging
import time

import pandas as pd
import yfinance as yf

from datum._types import FreqLiteral
from datum.exceptions import DatumFetchError
from datum.providers import BaseProvider, register

logger = logging.getLogger("datum")


class YahooProvider(BaseProvider):

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        freq: FreqLiteral,
        retry_attempts: int,
    ) -> pd.DataFrame:
        last_exc: Exception | None = None

        for attempt in range(1, retry_attempts + 1):
            try:
                data = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    interval=freq,
                    auto_adjust=False,
                    actions=False,
                    group_by="ticker",
                    threads=False,
                    progress=False,
                )
                if data is None or data.empty:
                    raise ValueError("Empty download from Yahoo Finance")

                # yfinance returns flat columns for a single ticker, MultiIndex for multiple
                if not isinstance(data.columns, pd.MultiIndex):
                    if len(tickers) != 1:
                        raise RuntimeError("Unexpected Yahoo column format for multiple tickers")
                    data.columns = pd.MultiIndex.from_product([tickers, list(data.columns)])

                return self._to_canonical(data, tickers)

            except Exception as exc:
                last_exc = exc
                logger.warning({"event": "download_failure", "source": "yahoo", "attempt": attempt, "error": str(exc)})
                time.sleep(min(2 ** attempt, 30))

        raise DatumFetchError("Failed to download market data from Yahoo Finance") from last_exc

    def _to_canonical(self, data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """Rename Yahoo field names to CANONICAL_FIELDS and drop unused columns."""
        rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        frames = []
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                continue
            df = data[ticker].copy()
            df = df.rename(columns=rename)
            df = df[["open", "high", "low", "close", "volume"]]
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            frames.append(df)

        if not frames:
            raise DatumFetchError("No valid ticker data from Yahoo Finance")

        return pd.concat(frames, axis=1).sort_index(axis=1)


register("yahoo", YahooProvider)
