import pandas as pd

from leak_detector import detect_unusual_expenses


def test_empty_data():
    df = pd.DataFrame({
        "Date": [],
        "Description": [],
        "Category": [],
        "Amount": [],
    })

    result = detect_unusual_expenses(df)

    assert result.empty


def test_normal_expenses_not_flagged():
    df = pd.DataFrame({
        "Date": ["2026-09-01", "2026-09-02", "2026-09-03"],
        "Description": ["Food 1", "Food 2", "Food 3"],
        "Category": ["Food", "Food", "Food"],
        "Amount": [100, 110, 105],
    })

    result = detect_unusual_expenses(df)

    assert result.empty


def test_category_with_fewer_than_three_records():
    df = pd.DataFrame({
        "Date": ["2026-09-01", "2026-09-02"],
        "Description": ["Food", "Food"],
        "Category": ["Food", "Food"],
        "Amount": [100, 10000],
    })

    result = detect_unusual_expenses(df)

    assert result.empty


def test_original_data_is_not_modified():
    df = pd.DataFrame({
        "Date": ["2026-09-01", "2026-09-02", "2026-09-03"],
        "Description": ["Food 1", "Food 2", "Food 3"],
        "Category": ["Food", "Food", "Food"],
        "Amount": [100, 110, 105],
    })

    original = df.copy(deep=True)

    detect_unusual_expenses(df)

    pd.testing.assert_frame_equal(df, original)


def test_detects_unusual_expense():
    df = pd.DataFrame({
        "Date": [
            "2026-09-01",
            "2026-09-02",
            "2026-09-03",
            "2026-09-04",
            "2026-09-05",
        ],
        "Description": [
            "Food 1",
            "Food 2",
            "Food 3",
            "Food 4",
            "Expensive Food",
        ],
        "Category": ["Food"] * 5,
        "Amount": [100, 110, 105, 95, 10000],
    })

    result = detect_unusual_expenses(df)

    assert len(result) == 1
    assert result.iloc[0]["Description"] == "Expensive Food"
    assert result.iloc[0]["Amount"] == 10000