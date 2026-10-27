import pandas as pd


def forecast_next_month(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df = df.dropna(subset=["Date", "Amount"])
    df = df[df["Amount"] >= 0]

    if df.empty:
        return None

    df["Month"] = df["Date"].dt.to_period("M")

    monthly_totals = df.groupby("Month")["Amount"].sum().sort_index()

    if len(monthly_totals) < 2:
        return None

    # Simple baseline: average of observed monthly totals
    forecast = monthly_totals.mean()

    last_month = monthly_totals.index[-1]
    next_month = last_month + 1

    return {
        "Last observed month": str(last_month),
        "Forecast month": str(next_month),
        "Forecast amount": round(float(forecast), 2),
        "Months of data": len(monthly_totals)
    }


if __name__ == "__main__":
    df = pd.read_csv("data/monthly_test.csv")

    result = forecast_next_month(df)

    if result is None:
        print("More valid monthly data is needed.")
    else:
        print("Expense Forecast:")
        for key, value in result.items():
            print(f"{key}: {value}")