import logging
import time

import pandas as pd

from datum._types import FreqLiteral
from datum.exceptions import DatumFetchError, DatumSchemaError, DatumTerminalError
from datum.providers import BaseProvider, register

logger = logging.getLogger("datum")

_FREQ_MAP = {"1d": "DAILY", "1wk": "WEEKLY", "1mo": "MONTHLY"}

_BDH_FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]

_RENAME = {
    "PX_OPEN": "open",
    "PX_HIGH": "high",
    "PX_LOW": "low",
    "PX_LAST": "close",
    "PX_VOLUME": "volume",
}

_CANONICAL = ["open", "high", "low", "close", "volume"]


class XbbgProvider(BaseProvider):
    """
    Bloomberg Terminal provider via xbbg.

    Requirements:
      - Bloomberg Terminal must be running (connects via localhost:8194)
      - pip install xbbg
      - pip install blpapi --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/

    Tickers must use Bloomberg yellow-key format, e.g. "AAPL US Equity", "SPX Index".
    """

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
        freq: FreqLiteral,
        retry_attempts: int,
    ) -> pd.DataFrame:
        try:
            from xbbg import blp  # noqa: F401 — import check only here
        except ImportError:
            raise ImportError(
                "xbbg is not installed. Run:\n"
                "  pip install xbbg\n"
                "  pip install blpapi "
                "--index-url https://blpapi.bloomberg.com/repository/releases/python/simple/"
            )

        bbg_freq = _FREQ_MAP.get(freq)
        if bbg_freq is None:
            raise ValueError(f"Unsupported freq for xbbg: {freq!r}. Supported: {list(_FREQ_MAP)}")

        _validate_tickers(tickers)

        last_exc: Exception | None = None

        for attempt in range(1, retry_attempts + 1):
            try:
                from xbbg import blp as _blp

                raw = _blp.bdh(
                    tickers=tickers,
                    flds=_BDH_FIELDS,
                    start_date=start,
                    end_date=end,
                    Per=bbg_freq,
                )

                try:
                    df = raw.to_pandas()
                except AttributeError as exc:
                    raise DatumSchemaError(
                        "xbbg return type has no .to_pandas() — check xbbg version"
                    ) from exc

                return _reshape(df, tickers)

            except (DatumTerminalError, DatumSchemaError, ValueError):
                raise  # non-retriable

            except Exception as exc:
                err = str(exc).lower()
                if "8194" in err or "connection" in err or "connect" in err:
                    raise DatumTerminalError(
                        "Bloomberg Terminal is not running or blpapi cannot connect on port 8194."
                    ) from exc
                if "not authorized" in err or "entitlement" in err:
                    raise DatumFetchError(
                        "Bloomberg entitlement error. Check terminal field permissions."
                    ) from exc

                last_exc = exc
                logger.warning({"event": "download_failure", "source": "xbbg", "attempt": attempt, "error": str(exc)})
                time.sleep(min(2 ** attempt, 30))

        raise DatumFetchError("Failed to download market data from Bloomberg") from last_exc


def _validate_tickers(tickers: list[str]) -> None:
    for t in tickers:
        if " " not in t:
            raise ValueError(
                f"Bloomberg ticker '{t}' appears to be missing its yellow-key suffix "
                "(e.g. 'AAPL US Equity', 'SPX Index'). "
                "xbbg requires Bloomberg format tickers."
            )


def _reshape(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Normalise xbbg bdh() output to MultiIndex (ticker, canonical_field) DataFrame.

    xbbg bdh() column format varies:
      - Single ticker, long-form: columns = (date, field, value) → needs pivot
      - Multiple tickers or already-pivoted: columns = MultiIndex (ticker, field)
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Already in (ticker, field) format — just rename fields
        frames = []
        for ticker in tickers:
            if ticker not in df.columns.get_level_values(0):
                logger.warning({"event": "missing_ticker", "ticker": ticker})
                continue
            sub = df[ticker].copy()
            _assert_fields(sub, ticker)
            sub = sub.rename(columns=_RENAME)[_CANONICAL]
            sub.columns = pd.MultiIndex.from_product([[ticker], sub.columns])
            frames.append(sub)

        if not frames:
            raise DatumFetchError("No valid ticker data from Bloomberg")

        result = pd.concat(frames, axis=1).sort_index(axis=1)

    else:
        # Long-form: columns are field names, index is date — pivot if needed
        # Or flat wide form with a single ticker
        if "field" in df.columns and "value" in df.columns:
            # Long-form with explicit field/value columns
            pivot_index = "date" if "date" in df.columns else df.index.name or df.index
            if isinstance(pivot_index, str) and pivot_index in df.columns:
                df = df.pivot(index=pivot_index, columns="field", values="value")
            else:
                df = df.pivot(columns="field", values="value")
        # Now df is flat wide for a single ticker
        if len(tickers) != 1:
            raise DatumSchemaError(
                "Received flat (non-MultiIndex) output from xbbg bdh() for multiple tickers. "
                "This is unexpected — check xbbg version."
            )
        ticker = tickers[0]
        _assert_fields(df, ticker)
        df = df.rename(columns=_RENAME)[_CANONICAL]
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        result = df

    # Ensure UTC DatetimeIndex
    result.index = pd.to_datetime(result.index)
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")

    return result.sort_index()


def _assert_fields(df: pd.DataFrame, ticker: str) -> None:
    missing = [f for f in _BDH_FIELDS if f not in df.columns]
    if missing:
        raise DatumSchemaError(
            f"Bloomberg BDH response for '{ticker}' is missing expected fields: {missing}. "
            "Check entitlements or xbbg version."
        )


register("xbbg", XbbgProvider)
