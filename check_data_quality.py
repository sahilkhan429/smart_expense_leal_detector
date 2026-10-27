import pandas as pd

df = pd.read_csv("data/sample_expenses.csv")

# 1. Check missing values
print("Missing values:")
print(df.isnull().sum())

# 2. Check duplicate rows
duplicate_rows = df.duplicated().sum()
print("\nDuplicate rows:", duplicate_rows)

# 3. Convert dates into proper date format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

invalid_dates = df["Date"].isna().sum()
print("Invalid dates:", invalid_dates)

# 4. Check invalid or negative expenses
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

invalid_amounts = (
    df["Amount"].isna() | (df["Amount"] < 0)
).sum()

print("Invalid amounts:", invalid_amounts)