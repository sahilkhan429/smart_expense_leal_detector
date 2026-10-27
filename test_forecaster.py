import pandas as pd

from forecaster import forecast_next_month


def test_forecast_with_multiple_months():
    df = pd.DataFrame({
        "Date": [
            "2026-09-01",
            "2026-10-01",
            "2026-11-01",
        ],
        "Amount": [1000, 2000, 3000],
    })

    result = forecast_next_month(df)

    assert result is not None
    assert result["Last observed month"] == "2026-11"
    assert result["Forecast month"] == "2026-12"
    assert result["Forecast amount"] == 2000
    assert result["Months of data"] == 3


def test_forecast_with_only_one_month():
    df = pd.DataFrame({
        "Date": ["2026-09-01", "2026-09-15"],
        "Amount": [1000, 2000],
    })

    result = forecast_next_month(df)

    assert result is None


def test_forecast_with_empty_data():
    df = pd.DataFrame({
        "Date": [],
        "Amount": [],
    })

    result = forecast_next_month(df)

    assert result is None


def test_forecast_does_not_modify_original_data():
    df = pd.DataFrame({
        "Date": ["2026-09-01", "2026-10-01"],
        "Amount": [1000, 2000],
    })

    original = df.copy(deep=True)

    forecast_next_month(df)

    pd.testing.assert_frame_equal(df, original)