import pandas as pd

from savings_advisor import suggest_savings


def test_category_above_limit_is_included():
    df = pd.DataFrame({
        "Category": ["Food", "Food", "Transport"],
        "Amount": [3000, 3000, 1000],
    })

    result = suggest_savings(df, spending_limit=5000)

    assert len(result) == 1
    assert result.iloc[0]["Category"] == "Food"
    assert result.iloc[0]["Total Spent"] == 6000
    assert result.iloc[0]["Possible Reduction (10%)"] == 600


def test_category_at_limit_is_not_included():
    df = pd.DataFrame({
        "Category": ["Food"],
        "Amount": [5000],
    })

    result = suggest_savings(df, spending_limit=5000)

    assert result.empty


def test_empty_data():
    df = pd.DataFrame({
        "Category": [],
        "Amount": [],
    })

    result = suggest_savings(df)

    assert result.empty


def test_original_data_is_not_modified():
    df = pd.DataFrame({
        "Category": ["Food", "Food"],
        "Amount": [3000, 4000],
    })

    original = df.copy(deep=True)

    suggest_savings(df)

    pd.testing.assert_frame_equal(df, original)