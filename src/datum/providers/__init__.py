from abc import ABC, abstractmethod

import pandas as pd

from datum._types import FreqLiteral


class BaseProvider(ABC):
    """
    Contract for all data providers.

    fetch() returns a MultiIndex (ticker, field) DataFrame where field ∈ CANONICAL_FIELDS
    and the index is a UTC-aware DatetimeIndex, sorted and deduplicated.
    """

    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        freq: FreqLiteral,
        retry_attempts: int,
    ) -> pd.DataFrame: ...


_REGISTRY: dict[str, type[BaseProvider]] = {}


def register(name: str, cls: type[BaseProvider]) -> None:
    _REGISTRY[name] = cls


def get_provider(name: str) -> BaseProvider:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


# Trigger provider registration at import time
from datum.providers import yahoo, binance, xbbg  # noqa: E402, F401
