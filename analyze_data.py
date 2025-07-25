import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Added for better heatmaps, ensure you have it installed: pip install seaborn

# --- Configuration ---
# Directory where the individual stock CSVs are stored (from previous step)
STOCK_DATA_DIR = 'stock_data_csvs' # This folder will be created in your project directory
# File path for the sector data CSV
SECTOR_DATA_PATH = r'C:\Users\user\Desktop\secondproject\Sector_data - Sheet1.csv' # <--- UPDATED PATH

# --- 1. Load and Consolidate Stock Data ---
all_stock_data = []
stock_symbols = []

print(f"Loading stock data from '{STOCK_DATA_DIR}'...")
# Ensure the STOCK_DATA_DIR path is relative to where you run this script,
# or provide an absolute path if running from a different location.
# Assuming you run this script from C:\Users\user\Desktop\secondproject
full_stock_data_path = os.path.join(r'C:\Users\user\Desktop\secondproject', STOCK_DATA_DIR)

if not os.path.exists(full_stock_data_path):
    print(f"Error: Stock data directory '{full_stock_data_path}' not found.")
    print("Please ensure 'extract_data.py' was run successfully and created this directory.")
    exit()

for filename in os.listdir(full_stock_data_path):
    if filename.endswith('.csv'):
        symbol = os.path.splitext(filename)[0]
        stock_symbols.append(symbol)
        filepath = os.path.join(full_stock_data_path, filename)
        try:
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Symbol'] = symbol # Add a symbol column to identify the stock
            all_stock_data.append(df)
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

if not all_stock_data:
    print(f"No stock data found in '{full_stock_data_path}'. Please ensure the previous step generated CSVs.")
    exit()

# Concatenate all individual stock DataFrames into one master DataFrame
master_df = pd.concat(all_stock_data, ignore_index=True)
master_df = master_df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)

# --- NEW FIX: Drop duplicate rows based on Symbol and Date ---
# This ensures unique date entries for each stock, preventing reindexing errors.
initial_rows = len(master_df)
master_df.drop_duplicates(subset=['Symbol', 'Date'], inplace=True)
if len(master_df) < initial_rows:
    print(f"Warning: Removed {initial_rows - len(master_df)} duplicate 'Symbol'-'Date' entries from master_df.")


print(f"\nMaster DataFrame created with {len(master_df)} rows.")
print("Sample of Master DataFrame:")
print(master_df.head())

# --- Load Sector Data ---
sector_df = pd.DataFrame() # Initialize as empty DataFrame
if os.path.exists(SECTOR_DATA_PATH):
    try:
        sector_df = pd.read_csv(SECTOR_DATA_PATH)
        print(f"\nLoaded sector data from '{SECTOR_DATA_PATH}'.")
        print("Sample of Sector DataFrame (before symbol extraction):")
        print(sector_df.head())

        # --- NEW: Extract actual Symbol from the 'Symbol' column in sector_df ---
        # This assumes the format is "COMPANY NAME: TICKER_SYMBOL"
        # We will split by ': ' and take the second part.
        if 'Symbol' in sector_df.columns:
            # Apply a function to safely extract the ticker symbol
            # If split fails or results in less than 2 parts, keep original or set to NaN
            sector_df['Extracted_Symbol'] = sector_df['Symbol'].apply(
                lambda x: x.split(': ')[1].strip() if isinstance(x, str) and ': ' in x and len(x.split(': ')) > 1 else x
            )
            # Now, we will merge using 'Extracted_Symbol' from sector_df
            # and 'Symbol' from master_df
            print("\nSample of Sector DataFrame (after symbol extraction):")
            print(sector_df[['Symbol', 'Extracted_Symbol', 'sector']].head())
        else:
            print("\nWarning: 'Symbol' column not found in Sector Data. Cannot extract ticker symbols for merging.")
            sector_df = pd.DataFrame() # Clear sector_df if Symbol column is missing

    except Exception as e:
        print(f"Error loading sector data from {SECTOR_DATA_PATH}: {e}")
        sector_df = pd.DataFrame() # Clear sector_df on error
else:
    print(f"Warning: Sector data file '{SECTOR_DATA_PATH}' not found. Sector-wise analysis will not be performed.")

# Merge master_df with sector_df to add sector information
if not sector_df.empty and 'Extracted_Symbol' in sector_df.columns:
    # Ensure 'sector' column (lowercase) exists in sector_df before merging
    if 'sector' in sector_df.columns:
        # Merge on master_df['Symbol'] and sector_df['Extracted_Symbol']
        master_df = pd.merge(master_df, sector_df[['Extracted_Symbol', 'sector']].drop_duplicates(subset=['Extracted_Symbol']),
                             left_on='Symbol', right_on='Extracted_Symbol', how='left')
        # Drop the redundant 'Extracted_Symbol' column after merge
        master_df.drop(columns=['Extracted_Symbol'], inplace=True)
        # Rename the 'sector' column to 'Sector' (capital 'S') for consistency in analysis
        master_df.rename(columns={'sector': 'Sector'}, inplace=True)
        print("\nMaster DataFrame after merging with Sector Data (sample with Sector column):")
        print(master_df.head())
    else:
        print("\nWarning: 'sector' column (lowercase) not found in Sector Data. Skipping merge with sector data.")
else:
    print("\nSkipping merge with sector data as it was not loaded or 'Extracted_Symbol' column is missing.")


# --- 2. Calculate Key Metrics ---

# Calculate Daily Returns for each stock
# Using .copy() to avoid SettingWithCopyWarning
master_df['Daily_Return'] = master_df.groupby('Symbol')['Close'].pct_change()

# Calculate Yearly Return for each stock
# This assumes 'yearly' means from the first available date to the last available date in the dataset
yearly_returns = {}
for symbol in stock_symbols:
    stock_df = master_df[master_df['Symbol'] == symbol].copy()
    if not stock_df.empty and len(stock_df) > 1: # Ensure there's enough data for a return calculation
        first_close = stock_df.iloc[0]['Close']
        last_close = stock_df.iloc[-1]['Close']
        if first_close != 0: # Avoid division by zero
            yearly_return = ((last_close - first_close) / first_close) * 100
            yearly_returns[symbol] = yearly_return
        else:
            yearly_returns[symbol] = 0.0 # Assign 0.0 if first_close is 0
    else:
        yearly_returns[symbol] = np.nan # Assign NaN if not enough data

yearly_returns_df = pd.DataFrame(list(yearly_returns.items()), columns=['Symbol', 'Yearly_Return_Pct'])
yearly_returns_df = yearly_returns_df.dropna(subset=['Yearly_Return_Pct']) # Remove stocks with no valid yearly return
print("\nYearly Returns for Stocks:")
print(yearly_returns_df.sort_values(by='Yearly_Return_Pct', ascending=False).head())

# Identify Top 10 Green Stocks (Best Performing)
top_10_green_stocks = yearly_returns_df.sort_values(by='Yearly_Return_Pct', ascending=False).head(10)
print("\nTop 10 Best Performing (Green) Stocks:")
print(top_10_green_stocks)

# Identify Top 10 Red Stocks (Worst Performing)
top_10_red_stocks = yearly_returns_df.sort_values(by='Yearly_Return_Pct', ascending=True).head(10)
print("\nTop 10 Worst Performing (Red) Stocks:")
print(top_10_red_stocks)

# Market Summary
total_stocks_with_returns = len(yearly_returns_df)
green_stocks_count = len(yearly_returns_df[yearly_returns_df['Yearly_Return_Pct'] >= 0])
red_stocks_count = total_stocks_with_returns - green_stocks_count # Assuming anything not green is red

average_close_price = master_df['Close'].mean()
average_volume = master_df['Volume'].mean()

print("\n--- Market Summary ---")
print(f"Total Stocks Analyzed (with valid yearly returns): {total_stocks_with_returns}")
print(f"Number of Green Stocks (Positive Yearly Return): {green_stocks_count}")
print(f"Number of Red Stocks (Negative Yearly Return): {red_stocks_count}")
if total_stocks_with_returns > 0:
    print(f"Percentage of Green Stocks: {green_stocks_count / total_stocks_with_returns * 100:.2f}%")
    print(f"Percentage of Red Stocks: {red_stocks_count / total_stocks_with_returns * 100:.2f}%")
else:
    print("Cannot calculate percentages as no stocks with valid returns were found.")
print(f"Average Close Price Across All Stocks: {average_close_price:.2f}")
print(f"Average Volume Across All Stocks: {average_volume:.2f}")

# --- 3. Volatility Analysis ---
# Calculate the standard deviation of daily returns for each stock
# Dropna to remove NaNs from pct_change at the beginning of each stock's data
volatility = master_df.groupby('Symbol')['Daily_Return'].std().dropna()
volatility_df = volatility.reset_index(name='Volatility')

# Sort to get the top 10 most volatile stocks
top_10_volatile_stocks = volatility_df.sort_values(by='Volatility', ascending=False).head(10)
print("\nTop 10 Most Volatile Stocks:")
print(top_10_volatile_stocks)

# Visualization: Top 10 Most Volatile Stocks
plt.figure(figsize=(12, 7))
plt.bar(top_10_volatile_stocks['Symbol'], top_10_volatile_stocks['Volatility'], color='skyblue')
plt.xlabel('Stock Symbol')
plt.ylabel('Volatility (Standard Deviation of Daily Returns)')
plt.title('Top 10 Most Volatile Stocks')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# --- 4. Cumulative Return Over Time ---
print("\n--- Cumulative Return Over Time Analysis ---")

# Calculate Cumulative Return for each stock
# The cumulative return is calculated as (1 + daily_return).cumprod()
# Use transform to ensure the output aligns with the original DataFrame's index
master_df['Cumulative_Return'] = master_df.groupby('Symbol')['Daily_Return'].transform(lambda x: (1 + x).cumprod() - 1)

# Get the final cumulative return for each stock
final_cumulative_returns = master_df.groupby('Symbol')['Cumulative_Return'].last().sort_values(ascending=False)

# Identify the top 5 performing stocks based on final cumulative return
top_5_performing_symbols = final_cumulative_returns.head(5).index.tolist()
print(f"\nTop 5 Performing Stocks by Cumulative Return: {top_5_performing_symbols}")
print(final_cumulative_returns.head(5))

# Visualization: Cumulative Return for Top 5 Performing Stocks
plt.figure(figsize=(14, 8))
for symbol in top_5_performing_symbols:
    # Filter data for the current top performing stock
    stock_data = master_df[master_df['Symbol'] == symbol].copy()
    # Plot Date vs Cumulative_Return
    plt.plot(stock_data['Date'], stock_data['Cumulative_Return'], label=symbol)

plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Return Over Time for Top 5 Performing Stocks')
plt.legend(title='Stock Symbol')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\nCumulative Return Over Time Analysis complete!")


# --- 5. Sector-wise Performance ---
print("\n--- Sector-wise Performance Analysis ---")

# Ensure 'Sector' column exists and is not entirely NaN
if 'Sector' in master_df.columns and not master_df['Sector'].isnull().all():
    # Calculate yearly return for each stock and merge with sector info
    # We already have yearly_returns_df from previous steps
    sector_performance_df = pd.merge(yearly_returns_df, master_df[['Symbol', 'Sector']].drop_duplicates(),
                                     on='Symbol', how='left')

    # Drop stocks where sector information is missing (NaN)
    sector_performance_df.dropna(subset=['Sector'], inplace=True)

    if not sector_performance_df.empty:
        # Calculate the average yearly return for each sector
        average_yearly_return_by_sector = sector_performance_df.groupby('Sector')['Yearly_Return_Pct'].mean().sort_values(ascending=False)
        print("\nAverage Yearly Return by Sector:")
        print(average_yearly_return_by_sector)

        # Visualization: Average Yearly Return by Sector
        plt.figure(figsize=(14, 8))
        average_yearly_return_by_sector.plot(kind='bar', color='lightgreen')
        plt.xlabel('Sector')
        plt.ylabel('Average Yearly Return (%)')
        plt.title('Average Yearly Return by Sector')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        print("\nSector-wise Performance Analysis complete!")
    else:
        print("\nNo valid sector data found after dropping missing values. Skipping Sector-wise Performance analysis.")
else:
    print("\n'Sector' column not available or contains all NaN values. Skipping Sector-wise Performance analysis.")


# --- 6. Stock Price Correlation ---
print("\n--- Stock Price Correlation Analysis ---")

# Pivot the master_df to get 'Close' prices with dates as index and symbols as columns
# This is needed for correlation calculation across different stocks
close_prices_pivot = master_df.pivot(index='Date', columns='Symbol', values='Close')

# Calculate daily returns for correlation (more common than raw prices)
daily_returns_pivot = close_prices_pivot.pct_change().dropna()

if not daily_returns_pivot.empty:
    # Calculate the correlation matrix
    correlation_matrix = daily_returns_pivot.corr()
    print("\nCorrelation Matrix (sample):")
    print(correlation_matrix.head())

    # Visualization: Stock Price Correlation Heatmap
    plt.figure(figsize=(16, 14)) # Adjust size for better readability with many stocks
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Stock Price Correlation Heatmap (Based on Daily Returns)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print("\nStock Price Correlation Analysis complete!")
else:
    print("\nNot enough data to calculate stock price correlation. Skipping correlation analysis.")


# --- 7. Top 5 Gainers and Losers (Month-wise) ---
print("\n--- Monthly Top 5 Gainers and Losers Analysis ---")

# Ensure 'Date' column is datetime and extract month-year
master_df['YearMonth'] = master_df['Date'].dt.to_period('M')

# Calculate monthly returns for each stock
# Get the first and last close price for each stock for each month
monthly_first_last_close = master_df.groupby(['Symbol', 'YearMonth'])['Close'].agg(['first', 'last'])

# --- NEW FIX: Calculate monthly returns directly and reset index for a flat DataFrame ---
# Calculate monthly return directly on the aggregated DataFrame
# Handle division by zero by replacing 0 with NaN, then fillna(0) for final result
monthly_first_last_close['Monthly_Return_Pct'] = (
    (monthly_first_last_close['last'] - monthly_first_last_close['first']) /
    monthly_first_last_close['first'].replace(0, np.nan)
)
# Reset index to make Symbol and YearMonth regular columns
monthly_returns = monthly_first_last_close.reset_index()
# Fill any inf/-inf or NaN values from division by zero or missing data with 0
monthly_returns['Monthly_Return_Pct'] = monthly_returns['Monthly_Return_Pct'].fillna(0)


if not monthly_returns.empty:
    # Get unique months for iteration
    unique_months = sorted(monthly_returns['YearMonth'].unique())

    for month in unique_months:
        monthly_data = monthly_returns[monthly_returns['YearMonth'] == month].copy()
        if not monthly_data.empty:
            # Sort for gainers and losers
            top_5_gainers = monthly_data.sort_values(by='Monthly_Return_Pct', ascending=False).head(5)
            top_5_losers = monthly_data.sort_values(by='Monthly_Return_Pct', ascending=True).head(5)

            # Combine for plotting
            plot_data = pd.concat([top_5_gainers, top_5_losers]).sort_values(by='Monthly_Return_Pct', ascending=False)

            if not plot_data.empty:
                plt.figure(figsize=(10, 6))
                colors = ['green' if x >= 0 else 'red' for x in plot_data['Monthly_Return_Pct']]
                plt.bar(plot_data['Symbol'], plot_data['Monthly_Return_Pct'] * 100, color=colors) # Convert to percentage
                plt.xlabel('Stock Symbol')
                plt.ylabel('Monthly Return (%)')
                plt.title(f'Top 5 Gainers and Losers for {month.strftime("%B %Y")}')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
            else:
                print(f"No data to plot for {month}.")
        else:
            print(f"No data for {month} to determine gainers/losers.")
    print("\nMonthly Top 5 Gainers and Losers Analysis complete!")
else:
    print("\nNot enough data to calculate monthly returns. Skipping monthly gainers/losers analysis.")


print("\nAll analysis steps (Key Metrics, Volatility, Cumulative Return, Sector-wise Performance, Stock Price Correlation, Monthly Gainers/Losers) complete!")
