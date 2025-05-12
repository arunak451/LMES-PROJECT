import pandas as pd

# Basic CSV read
df = pd.read_csv('dataset.csv')
print(df.head(7))

# Read Excel file
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
print(df.tail(6))

# Read multiple sheets
xls = pd.ExcelFile('file.xlsx')
df1 = pd.read_excel(xls, 'Sheet1')
df2 = pd.read_excel(xls, 'Sheet2')