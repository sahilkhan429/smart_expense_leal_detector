import pandas as pd

from analyzer import (
    calculate_total,
    category_summary,
    average_expense,
    monthly_summary,
)


def test_calculate_total():
    df = pd.DataFrame({"Amount": [100, 200, 300]})
    assert calculate_total(df) == 600


def test_category_summary():
    df = pd.DataFrame({
        "Category": ["Food", "Food", "Transport"],
        "Amount": [100, 200, 150],
    })

    result = category_summary(df)

    assert result["Food"] == 300
    assert result["Transport"] == 150


def test_average_expense():
    df = pd.DataFrame({"Amount": [100, 200, 300]})
    assert average_expense(df) == 200


def test_average_empty_data():
    df = pd.DataFrame({"Amount": []})
    assert average_expense(df) == 0


def test_monthly_summary():
    df = pd.DataFrame({
        "Date": ["2026-09-01", "2026-09-15", "2026-10-01"],
        "Amount": [100, 200, 400],
    })

    result = monthly_summary(df)

    assert result.sum() == 700
    assert len(result) == 2