import pandas as pd
import numpy as np
from oipd.io import CSVReader

# Test the reader directly
reader = CSVReader()

# Column mapping for bitcoin data
column_mapping = {
    "Strike": "strike",
    "Last": "last_price", 
    "Bid": "bid",
    "Ask": "ask",
    "OptionType": "option_type",
}

# Read the CSV file
df = reader.read(
    "data/bitcoin_date20250830_strike20251226_price108864.csv",
    column_mapping=column_mapping
)

print("DataFrame after CSVReader.read():")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:")
print(df.dtypes)
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast price column type: {df['last_price'].dtype}")
print(f"Sample last_price values (first 10):")
print(df['last_price'].head(10).tolist())

# Check for string values
print(f"\nChecking for non-numeric values in last_price:")
if df['last_price'].dtype == 'object':
    print("WARNING: last_price is still object type (strings)!")
    unique_non_numeric = df['last_price'][df['last_price'].apply(lambda x: isinstance(x, str))].unique()
    print(f"Unique string values: {unique_non_numeric}")
else:
    print(f"last_price is numeric type: {df['last_price'].dtype}")
    
# Check option_type column
print(f"\nOption type column:")
print(f"Unique values: {df['option_type'].unique() if 'option_type' in df.columns else 'Column not found'}")

# Test parity detection
from oipd.core.parity import detect_parity_opportunity
has_parity = detect_parity_opportunity(df)
print(f"\nParity detection result: {has_parity}")
