import pandas as pd


def detect_unusual_expenses(df):
    df = df.copy()

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount", "Category"])
    df["Category"] = df["Category"].astype(str).str.strip()
    df = df[df["Category"] != ""]

    output_columns = [
        "Date", "Description", "Category", "Amount", "Threshold"
    ]

    if df.empty:
        return pd.DataFrame(columns=output_columns)

    results = []

    for category, group in df.groupby("Category"):
        amounts = group["Amount"]

        if len(group) < 4:
            continue

        median = amounts.median()
        mad = (amounts - median).abs().median()

        if mad > 0:
            threshold = median + 3 * 1.4826 * mad
        else:
            q1 = amounts.quantile(0.25)
            q3 = amounts.quantile(0.75)
            threshold = q3 + 1.5 * (q3 - q1)

        unusual = group[group["Amount"] > threshold].copy()

        if not unusual.empty:
            unusual["Threshold"] = threshold
            results.append(unusual)

    if not results:
        return pd.DataFrame(columns=output_columns)

    return pd.concat(results, ignore_index=True)[output_columns]


if __name__ == "__main__":
    df = pd.read_csv("data/monthly_test.csv")
    result = detect_unusual_expenses(df)
    print("Category-wise unusual expenses:")
    print(result.to_string(index=False) if not result.empty else "No unusual expenses detected.")