import pandas as pd


def calculate_total(df):
    return df["Amount"].sum()


def category_summary(df):
    return df.groupby("Category")["Amount"].sum()


def average_expense(df):
    if len(df) == 0:
        return 0

    return df["Amount"].mean()
def largest_expense(df):
    if len(df) == 0:
        return None

    return df.loc[df["Amount"].idxmax()]

def monthly_summary(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Amount"])
    df["Month"] = df["Date"].dt.to_period("M")
    return df.groupby("Month")["Amount"].sum()


if __name__ == "__main__":
    df = pd.read_csv("data/monthly_test.csv")

    print("Total expenses:", calculate_total(df))

    print("\nCategory summary:")
    print(category_summary(df))

    print("\nAverage expense:", average_expense(df))
    largest = largest_expense(df)
    if largest is not None:
        print("\nLargest expense:")
        print(largest)
    else:
        print("\nNo expenses found.")

    print("\nMonthly summary:")
    print(monthly_summary(df))
    
