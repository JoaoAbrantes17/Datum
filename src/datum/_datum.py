import logging
from typing import Any, Dict

import pandas as pd

from datum._types import CalendarLiteral, FreqLiteral, PolicyLiteral, ProviderLiteral
from datum.exceptions import DatumFetchError
from datum.pipeline import process_ticker
from datum.providers import get_provider
from datum.schema import CANONICAL_FIELDS

logger = logging.getLogger("datum")
logger.setLevel(logging.INFO)


class Datum:
    """
    Canonical gateway for historical market data.

    Fetches OHLCV from the requested provider and normalises the output to a
    consistent MultiIndex (ticker, field) DataFrame with fields:
        open, high, low, close, volume

    Index: UTC DatetimeIndex, sorted, deduplicated.
    """

    def __init__(
        self,
        tickers: list[str],
        start: str,
        end: str,
        freq: FreqLiteral,
        provider: ProviderLiteral = "yahoo",
        dropna_policy: PolicyLiteral = "strict",
        calendar: CalendarLiteral = "exchange",
        timezone: str = "UTC",
        retry_attempts: int = 1,
    ):
        self.tickers = sorted(set(tickers))
        self.start = start
        self.end = end
        self.freq = freq
        self.provider = provider
        self.dropna_policy = dropna_policy
        self.calendar = calendar
        self.timezone = timezone
        self.retry_attempts = retry_attempts

        self._raw: pd.DataFrame | None = None
        self._prices: pd.DataFrame | None = None

        self._metadata: Dict[str, Any] = {
            "coverage": {},
            "missing_data_actions": [],
            "source": provider,
        }

        self._load()

    # ------------------------------------------------------------------
    # DATA ACQUISITION
    # ------------------------------------------------------------------

    def _load(self) -> None:
        provider_instance = get_provider(self.provider)
        self._raw = provider_instance.fetch(
            self.tickers, self.start, self.end, self.freq, self.retry_attempts
        )
        if self._raw is None or self._raw.empty:
            raise DatumFetchError("Failed to download market data")
        self._process()

    # ------------------------------------------------------------------
    # PROCESSING PIPELINE
    # ------------------------------------------------------------------

    def _process(self) -> None:
        frames: list[pd.DataFrame] = []

        for ticker in self.tickers:
            if (
                not isinstance(self._raw.columns, pd.MultiIndex)
                or ticker not in self._raw.columns.get_level_values(0)
            ):
                logger.warning({"event": "missing_ticker", "ticker": ticker})
                continue

            flat = self._raw[ticker].copy()
            canonical = process_ticker(
                flat,
                ticker,
                self.dropna_policy,
                self.calendar,
                self.timezone,
                self._metadata,
            )
            canonical.columns = pd.MultiIndex.from_product([[ticker], canonical.columns])
            frames.append(canonical)

        if not frames:
            raise RuntimeError("No valid ticker data available")

        prices = pd.concat(frames, axis=1)
        columns = pd.MultiIndex.from_product([self.tickers, CANONICAL_FIELDS])
        self._prices = prices.reindex(columns=columns)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def get_prices(self) -> pd.DataFrame:
        if self._prices is None:
            raise RuntimeError("Prices not available")
        return self._prices.copy()

    def get_metadata(self) -> dict:
        return dict(self._metadata)
