import pandas as pd    #IMPORT PANDAS FOR DATA MANIPULATION
df = pd.read_csv('data/sample_expenses.csv')    #READ THE CSV FILE INTO A DATAFRAME
print("ALL EXPENSES:")     #PRINT ALL EXPENSES
print(df)

total_expenses = df['Amount'].sum()   #CALCULATE THE TOTAL EXPENSES BY SUMMING THE 'Amount' COLUMN
print("Total Expenses:", total_expenses)    #PRINT THE TOTAL EXPENSES


category_totals = df.groupby('Category')['Amount'].sum()    #GROUP THE DATA BY 'Category' AND SUM THE 'Amount' FOR EACH CATEGORY
print("Expenses by Category:")    #PRINT THE EXPENSES BY CATEGORY
print(category_totals)



