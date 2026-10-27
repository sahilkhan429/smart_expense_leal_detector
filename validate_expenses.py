import pandas as pd
df = pd.read_csv('data/sample_expenses.csv')    #READ THE CSV FILE INTO A DATAFRAME
required_columns = ['Date','Description', 'Category', 'Amount']    #DEFINE THE REQUIRED COLUMNS
missing_columns = [column for column in required_columns if column not in df.columns]    #CHECK FOR MISSING COLUMNS
if missing_columns:
    print("Missing columns:", missing_columns)
else:
    print("All required columns are present.")

df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

invalid_amounts = df["Amount"].isna()

if invalid_amounts.any():
    print("Error: Some amounts are missing or invalid.")
else:
    print("All amounts are valid numbers.")