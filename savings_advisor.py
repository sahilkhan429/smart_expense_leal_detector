import pandas as pd


def suggest_savings(df, spending_limit=5000):
    df = df.copy()

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Category", "Amount"])

    category_totals = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    suggestions = []

    for category, total in category_totals.items():
        if total > spending_limit:
            suggestions.append({
                "Category": category,
                "Total Spent": total,
                "Possible Reduction (10%)": round(total * 0.10, 2)
            })

    return pd.DataFrame(suggestions)


if __name__ == "__main__":
    df = pd.read_csv("data/monthly_test.csv")

    result = suggest_savings(df)

    print("Possible Savings Suggestions:")
    print(result.to_string(index=False) if not result.empty else "No categories exceed the limit.")