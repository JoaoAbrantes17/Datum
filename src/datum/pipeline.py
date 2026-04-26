import logging

import numpy as np
import pandas as pd

from datum._types import CalendarLiteral, PolicyLiteral
from datum.exceptions import DatumMissingDataError, DatumSchemaError
from datum.schema import CANONICAL_FIELDS

logger = logging.getLogger("datum")


def process_ticker(
    df: pd.DataFrame,
    ticker: str,
    dropna_policy: PolicyLiteral,
    calendar: CalendarLiteral,
    timezone: str,
    metadata: dict,
) -> pd.DataFrame:
    """
    Validates and normalises a single-ticker flat DataFrame.

    Input:  flat columns matching CANONICAL_FIELDS, any DatetimeIndex.
    Output: flat columns matching CANONICAL_FIELDS, UTC DatetimeIndex.
    Mutates metadata in place (coverage, missing_data_actions).
    """
    df = _normalize_index(df, timezone)
    df = _enforce_schema(df, ticker)
    df = _coerce_types(df, ticker)
    _validate_volume(df, ticker)
    df = _apply_missing_policy(df, ticker, dropna_policy, metadata)
    df = _apply_calendar(df, calendar)

    metadata["coverage"][ticker] = {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    }

    return df


# ---------------------------------------------------------------------------
# Internal steps
# ---------------------------------------------------------------------------

def _normalize_index(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _enforce_schema(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    for col in CANONICAL_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
    return df[CANONICAL_FIELDS]


def _coerce_types(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    if any(df.dtypes == "object"):
        raise DatumSchemaError(f"Object dtype after numeric coercion for {ticker}")
    return df


def _validate_volume(df: pd.DataFrame, ticker: str) -> None:
    if (df["volume"] < 0).any():
        raise ValueError(f"Negative volume detected for {ticker}")
    zero_count = int((df["volume"] == 0).sum())
    if zero_count > 0:
        logger.warning({"event": "zero_volume_days", "ticker": ticker, "count": zero_count})


def _apply_missing_policy(
    df: pd.DataFrame,
    ticker: str,
    policy: PolicyLiteral,
    metadata: dict,
) -> pd.DataFrame:
    if not df.isna().any().any():
        return df

    if policy == "strict":
        raise DatumMissingDataError(f"Missing data detected for {ticker}")

    if policy == "forward":
        df = df.ffill()
        metadata["missing_data_actions"].append({"ticker": ticker, "action": "forward_fill"})

    if policy == "interpolate":
        df = df.interpolate(method="time")
        metadata["missing_data_actions"].append({"ticker": ticker, "action": "interpolate"})

    return df


def _apply_calendar(df: pd.DataFrame, calendar: CalendarLiteral) -> pd.DataFrame:
    if calendar == "business":
        idx = pd.date_range(df.index.min(), df.index.max(), freq="B", tz="UTC")
        df = df.reindex(idx)
    return df
