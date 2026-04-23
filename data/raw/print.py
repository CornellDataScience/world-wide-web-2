import pandas as pd

# Load the JSONL file
df = pd.read_json('data/raw/train_live.jsonl', lines=True, nrows = 50)

# Print the list of column names
print(list(df.columns))
print(df.at[10, "output"])

# Messages is equivalent to actions I think? Output and prompt