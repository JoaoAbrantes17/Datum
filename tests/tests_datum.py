import numpy as np
import pandas as pd
import pytest

import datum


def mock_download(*args, **kwargs):
    idx = pd.date_range("2020-01-01", periods=3, freq="D")

    data = pd.DataFrame(
        {
            ("AAPL", "Open"): [10, 11, 12],
            ("AAPL", "High"): [11, 12, 13],
            ("AAPL", "Low"): [9, 10, 11],
            ("AAPL", "Close"): [10, 11, 12],
            ("AAPL", "Adj Close"): [5, 5.5, 6],
            ("AAPL", "Volume"): [100, 100, 100],
            ("AAPL", "Dividends"): [0, 0, 0],
            ("AAPL", "Stock Splits"): [0, 0, 0],
        },
        index=idx,
    )
    data.columns = pd.MultiIndex.from_tuples(data.columns)
    return data


def test_adjustment_correctness(monkeypatch):
    monkeypatch.setattr("yfinance.download", mock_download)

    d = datum.Datum(
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-01-04",
        freq="1d",
    )

    prices = d.get_prices()
    close_adj = prices[("AAPL", "close_adj")]

    expected = pd.Series(
        [5.0, 5.5, 6.0], index=close_adj.index
    )

    assert np.allclose(close_adj, expected)


def test_missing_data_strict(monkeypatch):
    def bad_download(*args, **kwargs):
        df = mock_download()
        df[("AAPL", "Close")].iloc[1] = np.nan
        return df

    monkeypatch.setattr("yfinance.download", bad_download)

    with pytest.raises(ValueError):
        datum.Datum(
            tickers=["AAPL"],
            start="2020-01-01",
            end="2020-01-04",
            freq="1d",
            dropna_policy="strict",
        )


def test_multi_ticker_alignment(monkeypatch):
    def multi_download(*args, **kwargs):
        idx = pd.date_range("2020-01-01", periods=2)
        data = {}

        for t in ["AAPL", "MSFT"]:
            for f in [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]:
                data[(t, f)] = [10, 11]

        df = pd.DataFrame(data, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    monkeypatch.setattr("yfinance.download", multi_download)

    d = datum.Datum(
        tickers=["MSFT", "AAPL"],
        start="2020-01-01",
        end="2020-01-03",
        freq="1d",
    )

    prices = d.get_prices()
    assert list(prices.columns.get_level_values(0)) == ["AAPL", "MSFT"]


def test_invalid_ticker(monkeypatch):
    monkeypatch.setattr("yfinance.download", mock_download)

    d = datum.Datum(
        tickers=["AAPL", "INVALID"],
        start="2020-01-01",
        end="2020-01-04",
        freq="1d",
    )

    prices = d.get_prices()
    assert "INVALID" not in prices.columns.get_level_values(0)
