import pandas as pd

df = pd.read_csv("data/sample_expenses.csv")

# Convert columns into correct data types
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

# Remove rows with missing required values
df = df.dropna(subset=["Date", "Description", "Category", "Amount"])

# Remove negative expenses
df = df[df["Amount"] >= 0]

# Remove duplicate rows
df = df.drop_duplicates()

# Reset row numbering
df = df.reset_index(drop=True)

print("Cleaned Expenses:")
print(df)

print("\nTotal clean rows:", len(df))
print("Total clean expenses:", df["Amount"].sum())