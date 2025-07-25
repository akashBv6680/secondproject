import yaml
import pandas as pd
import os

# --- Configuration ---
# This is the base directory where you will extract your data.rar content.
# For example, if you extract data.rar to 'my_stock_data',
# then inside 'my_stock_data' you'd have '2023-10', '2023-11', etc.
# UPDATED PATH BASED ON YOUR INPUT:
BASE_DATA_DIR = r'C:\Users\user\Desktop\secondproject' # <--- UPDATED PATH

# --- Simulate Data Structure for Testing (if you don't have actual files yet) ---
# This dictionary mimics the structure after you extract data.rar
# and assumes YAML files like '2023-10-03.yaml' exist within month folders.
# You will replace this with actual file system reading.
dummy_data_structure = {
    '2023-10': {
        '2023-10-03.yaml': """
- Ticker: SBIN
  close: 602.95
  date: '2023-10-03 05:30:00'
  high: 604.9
  low: 589.6
  month: 2023-10
  open: 596.6
  volume: 15322196
- Ticker: BAJFINANCE
  close: 7967.6
  date: '2023-10-03 05:30:00'
  high: 7975.5
  low: 7755.0
  month: 2023-10
  open: 7780.8
  volume: 944555
- Ticker: RELIANCE
  close: 1159.08
  date: '2023-10-03 05:30:00'
  high: 1167.8
  low: 1158.0
  month: 2023-10
  open: 1164.97
  volume: 8859056
- Ticker: TCS
  close: 3513.85
  date: '2023-10-03 05:30:00'
  high: 3534.2
  low: 3480.1
  month: 2023-10
  open: 3534.2
  volume: 1948148
"""
    },
    '2023-11': {
        '2023-11-01.yaml': """
- Ticker: SBIN
  close: 610.0
  date: '2023-11-01 05:30:00'
  high: 615.0
  low: 600.0
  month: 2023-11
  open: 605.0
  volume: 16000000
- Ticker: RELIANCE
  close: 1170.0
  date: '2023-11-01 05:30:00'
  high: 1180.0
  low: 1160.0
  month: 2023-11
  open: 1165.0
  volume: 9000000
"""
    }
}

# --- Function to read data from actual files ---
def read_data_from_files(base_dir):
    """
    Reads YAML data from the specified directory structure (month_folder/date_file.yaml).
    Returns a list of dictionaries, where each dict is a stock entry for a specific day.
    """
    all_stock_entries = []
    if not os.path.exists(base_dir):
        print(f"Error: Base data directory '{base_dir}' not found. Please ensure the path is correct.")
        print("Using dummy data for demonstration.")
        # Fallback to dummy data if directory not found
        for month_folder, date_files in dummy_data_structure.items():
            for filename, content in date_files.items():
                try:
                    # Load each YAML content string as a list of dictionaries
                    daily_entries = yaml.safe_load(content)
                    all_stock_entries.extend(daily_entries)
                except yaml.YAMLError as e:
                    print(f"Error parsing dummy YAML content for {month_folder}/{filename}: {e}")
        return all_stock_entries

    print(f"Reading data from '{base_dir}'...")
    # List directories and sort them to ensure consistent processing order (e.g., '2023-10' before '2023-11')
    for month_folder in sorted(os.listdir(base_dir)):
        month_path = os.path.join(base_dir, month_folder)
        if os.path.isdir(month_path): # Ensure it's a directory
            print(f"Processing month: {month_folder}")
            # List files within the month directory and sort them
            for date_file in sorted(os.listdir(month_path)):
                if date_file.endswith('.yaml') or date_file.endswith('.yml'):
                    file_path = os.path.join(month_path, date_file)
                    try:
                        with open(file_path, 'r') as f:
                            # Each YAML file is a list of stock entries for that date
                            daily_entries = yaml.safe_load(f)
                            if daily_entries: # Ensure the loaded data is not empty
                                all_stock_entries.extend(daily_entries)
                    except FileNotFoundError:
                        print(f"File not found: {file_path}")
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML file {file_path}: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred with {file_path}: {e}")
    return all_stock_entries

# --- Main Execution ---
output_dir = 'stock_data_csvs'
# Create the output directory if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)

# Read all data from files (or use dummy data if BASE_DATA_DIR not found)
all_raw_data = read_data_from_files(BASE_DATA_DIR)

if not all_raw_data:
    print("No data loaded. Exiting data extraction process.")
    # If no data is loaded, we can't proceed with DataFrame creation or saving.
    exit()

# Convert the list of dictionaries to a pandas DataFrame
master_df = pd.DataFrame(all_raw_data)

# Rename columns to be consistent with common conventions (optional but good practice)
master_df.rename(columns={
    'Ticker': 'Symbol',
    'close': 'Close',
    'high': 'High',
    'low': 'Low',
    'open': 'Open',
    'volume': 'Volume'
}, inplace=True)

# Convert 'date' column to datetime objects and extract just the date part
# Using .dt.date converts it to a Python date object, which is fine for storage,
# but for further calculations with pandas, it's often better to keep it as datetime.
# Let's keep it as datetime for now, and only extract date if strictly needed later.
master_df['date'] = pd.to_datetime(master_df['date'])
master_df.rename(columns={'date': 'Date'}, inplace=True)

# Drop the 'month' column as it's redundant now that we have 'Date'
if 'month' in master_df.columns:
    master_df.drop(columns=['month'], inplace=True)

# Sort by Symbol and Date for consistency
master_df = master_df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)

print(f"\nMaster DataFrame created with {len(master_df)} rows.")
print("Sample of Master DataFrame:")
print(master_df.head())

# --- Save individual CSV files for each symbol ---
print(f"\nSaving individual CSV files to '{output_dir}' directory...")
unique_symbols = master_df['Symbol'].unique()

for symbol in unique_symbols:
    # Select data for the current symbol
    symbol_df = master_df[master_df['Symbol'] == symbol].copy()
    csv_filename = os.path.join(output_dir, f'{symbol}.csv')
    # Save the DataFrame to a CSV file without the index
    symbol_df.to_csv(csv_filename, index=False)
    print(f"Saved {symbol}.csv with {len(symbol_df)} rows.")

print("\nData extraction and transformation complete!")
print(f"You can find the generated CSV files in the '{output_dir}' directory.")
