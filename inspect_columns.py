import pandas as pd

# Load dataset and skip the index column
df = pd.read_csv('emails.csv', index_col=0)

# Print column names
print("Column names:", df.columns.tolist())
