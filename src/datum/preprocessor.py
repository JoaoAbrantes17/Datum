import logging
from typing import Any

import numpy as np
import pandas as pd

from datum.exceptions import DatumSchemaError
from datum.schema import CANONICAL_FIELDS

logger = logging.getLogger("datum")


class Preprocessor:
    """
    Quality analysis and cleaning for OHLCV price frames produced by Datum.

    Input: MultiIndex (ticker, field) DataFrame with fields = CANONICAL_FIELDS
    and a UTC DatetimeIndex.

    Workflow:
        p = Preprocessor(prices)
        p.coverage_report()
        p.missing_report()
        p.outlier_report()
        p.quality_report()
        clean = p.clean()        # winsorized OHLCV
        report = p.get_report()  # everything bundled
    """

    def __init__(self, prices: pd.DataFrame, zscore_threshold: float = 4.0):
        self._validate(prices)
        self.prices = prices.copy()
        self.zscore_threshold = float(zscore_threshold)
        self.tickers: list[str] = list(prices.columns.get_level_values(0).unique())
        self._cleaned: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(prices: pd.DataFrame) -> None:
        if not isinstance(prices.columns, pd.MultiIndex) or prices.columns.nlevels != 2:
            raise DatumSchemaError("Preprocessor expects a MultiIndex (ticker, field) DataFrame")
        fields = set(prices.columns.get_level_values(1))
        missing = set(CANONICAL_FIELDS) - fields
        if missing:
            raise DatumSchemaError(f"Missing canonical fields: {sorted(missing)}")
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise DatumSchemaError("Index must be a DatetimeIndex")

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def coverage_report(self) -> dict[str, dict[str, Any]]:
        """Per-ticker start, end, observation count, and gap count vs business-day calendar."""
        out: dict[str, dict[str, Any]] = {}
        for t in self.tickers:
            close = self.prices[(t, "close")].dropna()
            if close.empty:
                out[t] = {"start": None, "end": None, "n_obs": 0, "expected_obs": 0, "gap_days": 0}
                continue
            expected = pd.date_range(close.index.min(), close.index.max(), freq="B", tz=close.index.tz)
            out[t] = {
                "start": str(close.index.min()),
                "end": str(close.index.max()),
                "n_obs": int(close.shape[0]),
                "expected_obs": int(expected.shape[0]),
                "gap_days": int(max(expected.shape[0] - close.shape[0], 0)),
            }
        return out

    def missing_report(self) -> pd.DataFrame:
        """Per (ticker, field) missing counts and percentages."""
        n = len(self.prices)
        rows = []
        for t in self.tickers:
            for f in CANONICAL_FIELDS:
                col = self.prices[(t, f)]
                n_missing = int(col.isna().sum())
                rows.append({
                    "ticker": t,
                    "field": f,
                    "n_missing": n_missing,
                    "pct_missing": (n_missing / n) if n else 0.0,
                })
        return pd.DataFrame(rows)

    def outlier_report(self) -> pd.DataFrame:
        """Z-score outliers on log returns of close, per ticker."""
        rows = []
        for t in self.tickers:
            z = self._log_return_zscores(t)
            mask = z.abs() > self.zscore_threshold
            for date, zval in z[mask].items():
                rows.append({
                    "ticker": t,
                    "date": date,
                    "log_return": float(np.log(self.prices[(t, "close")]).diff().loc[date]),
                    "zscore": float(zval),
                })
        return pd.DataFrame(rows, columns=["ticker", "date", "log_return", "zscore"])

    def quality_report(self) -> dict[str, Any]:
        """Bundle structural checks: negative prices, broken OHLC, zero volume, duplicate/non-monotonic index."""
        checks: dict[str, Any] = {
            "duplicate_index_rows": int(self.prices.index.duplicated().sum()),
            "monotonic_index": bool(self.prices.index.is_monotonic_increasing),
            "per_ticker": {},
        }
        for t in self.tickers:
            o = self.prices[(t, "open")]
            h = self.prices[(t, "high")]
            l = self.prices[(t, "low")]
            c = self.prices[(t, "close")]
            v = self.prices[(t, "volume")]
            checks["per_ticker"][t] = {
                "negative_price_rows": int(((o < 0) | (h < 0) | (l < 0) | (c < 0)).sum()),
                "high_lt_low_rows": int((h < l).sum()),
                "close_outside_hl_rows": int(((c > h) | (c < l)).sum()),
                "open_outside_hl_rows": int(((o > h) | (o < l)).sum()),
                "zero_volume_rows": int((v == 0).sum()),
                "negative_volume_rows": int((v < 0).sum()),
            }
        return checks

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self) -> pd.DataFrame:
        """
        Winsorize OHLCV by clipping close-return outliers to ±zscore_threshold·σ,
        rescaling that day's OHL by (new_close / old_close) so intraday shape is preserved.
        Volume is left untouched.
        """
        cleaned = self.prices.copy()
        for t in self.tickers:
            close = cleaned[(t, "close")]
            log_ret = np.log(close).diff()
            sigma = log_ret.std(skipna=True)
            if not np.isfinite(sigma) or sigma == 0:
                continue
            limit = self.zscore_threshold * sigma
            clipped_ret = log_ret.clip(lower=-limit, upper=limit)

            # Rebuild close from clipped returns, anchored at first observation.
            new_close = close.copy()
            mask = clipped_ret != log_ret
            if not mask.any():
                continue
            # Reconstruct only affected days; cascade adjustment forward.
            scale = pd.Series(1.0, index=close.index)
            for date in close.index[mask.fillna(False)]:
                prev = new_close.shift(1).loc[date]
                new_val = prev * np.exp(clipped_ret.loc[date])
                scale.loc[date] = new_val / close.loc[date]
                new_close.loc[date] = new_val

            cleaned[(t, "close")] = new_close
            cleaned[(t, "open")] = cleaned[(t, "open")] * scale
            cleaned[(t, "high")] = cleaned[(t, "high")] * scale
            cleaned[(t, "low")] = cleaned[(t, "low")] * scale

        self._cleaned = cleaned
        return cleaned

    def get_clean_prices(self) -> pd.DataFrame:
        if self._cleaned is None:
            return self.clean()
        return self._cleaned.copy()

    def get_report(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage_report(),
            "missing": self.missing_report().to_dict(orient="records"),
            "outliers": self.outlier_report().to_dict(orient="records"),
            "quality": self.quality_report(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log_return_zscores(self, ticker: str) -> pd.Series:
        close = self.prices[(ticker, "close")]
        log_ret = np.log(close).diff()
        sigma = log_ret.std(skipna=True)
        if not np.isfinite(sigma) or sigma == 0:
            return pd.Series(0.0, index=log_ret.index)
        return (log_ret - log_ret.mean(skipna=True)) / sigma
