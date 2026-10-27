
import pandas as pd
import pytest

from ml_detector import detect_ml_anomalies


def make_expenses(amounts):
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=len(amounts)),
        "Description": [
            f"Expense {i}" for i in range(len(amounts))
        ],
        "Category": ["Food"] * len(amounts),
        "Amount": amounts,
    })


def test_empty_dataframe_returns_empty():
    df = pd.DataFrame(
        columns=["Date", "Description", "Category", "Amount"]
    )

    result = detect_ml_anomalies(df)

    assert result.empty


def test_small_dataset_returns_empty():
    df = make_expenses([100, 200, 300])

    result = detect_ml_anomalies(df)

    assert result.empty


def test_invalid_amounts_are_handled():
    df = make_expenses(["bad", None, "bad", None, "bad"])

    result = detect_ml_anomalies(df)

    assert result.empty


def test_missing_column_raises_error():
    df = pd.DataFrame({
        "Description": ["Expense A"],
        "Amount": [100],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        detect_ml_anomalies(df)


def test_extreme_amount_is_detected():
    df = make_expenses([
        100, 110, 120, 130, 140,
        150, 160, 170, 180, 5000,
    ])

    result = detect_ml_anomalies(df)

    assert not result.empty
    assert 5000 in result["Amount"].tolist()
    assert "ML Detection" in result.columns
    assert "Detection Reason" in result.columns


def test_normal_amounts_not_flagged_in_sample():
    df = make_expenses([
        100, 110, 120, 130, 140,
        150, 160, 170, 180, 5000,
    ])

    result = detect_ml_anomalies(df)

    flagged_amounts = result["Amount"].tolist()

    assert 100 not in flagged_amounts
    assert 180 not in flagged_amounts


def test_constant_amounts_return_empty():
    df = make_expenses([100, 100, 100, 100, 100, 100])

    result = detect_ml_anomalies(df)

    assert result.empty


def test_negative_and_invalid_amounts_are_not_flagged():
    df = make_expenses([
        100, 110, 120, 130, 140,
        150, 160, 170, 180, 5000,
    ])

    df.loc[0, "Amount"] = -100

    result = detect_ml_anomalies(df)

    assert -100 not in result["Amount"].tolist()
    assert 5000 in result["Amount"].tolist()
