import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set Matplotlib style for better visuals in Streamlit
plt.style.use('seaborn-v0_8-darkgrid') # Using a modern seaborn style
# plt.rcParams['font.family'] = 'Inter' # Commented out: Findfont: Font family 'Inter' not found error

# --- Custom CSS for Background Color ---
st.markdown(
    """
    <style>
    body {
        background-color: #264653; /* Deep Teal background */
        color: #f4f4f4; /* Light text for contrast on dark background */
    }
    .stApp {
        background-color: #264653; /* Ensures the main app area also has the background */
        color: #f4f4f4; /* Light text for contrast on dark background */
    }
    /* Adjust header colors for better visibility on dark background */
    h1, h2, h3, h4, h5, h6 {
        color: #e9c46a; /* A contrasting golden yellow for headers */
    }
    /* Adjust dataframe text color for readability */
    .dataframe {
        color: #f4f4f4; /* Light text for dataframe content */
    }
    /* Adjust info/success/warning box colors if needed for contrast */
    .stAlert {
        color: #333333; /* Darker text for alerts to stand out */
        background-color: #fef9ef; /* Lighter background for alerts */
    }

    /* Other attractive color options to try: */
    /*
    # Dark Blue/Navy:
    body { background-color: #1a2a3a; color: #f0f0f0; }
    h1, h2, h3 { color: #87ceeb; } /* Sky Blue for headers */

    # Dark Green/Forest:
    body { background-color: #2f4f4f; color: #f8f8f8; }
    h1, h2, h3 { color: #90ee90; } /* Light Green for headers */

    # Charcoal Gray:
    body { background-color: #36454F; color: #f0f0f0; }
    h1, h2, h3 { color: #ADD8E6; } /* Light Blue for headers */
    */
    </style>
    """,
    unsafe_allow_html=True
)

# --- Configuration ---
# Define paths relative to the app.py script's location (C:\Users\user\Desktop\secondproject)
STOCK_DATA_DIR = 'stock_data_csvs'
SECTOR_DATA_PATH = 'Sector_data - Sheet1.csv' # Ensure this file is in the same directory as app.py

# --- Function to load all stock data ---
@st.cache_data # Cache the data loading for better performance
def load_all_stock_data(data_dir):
    """Loads all individual stock CSVs into a single DataFrame."""
    all_stock_data = []
    stock_symbols = []

    # Construct the full path to the stock data directory
    # os.getcwd() gets the current working directory where app.py is run from
    full_data_path = os.path.join(os.getcwd(), data_dir)

    if not os.path.exists(full_data_path):
        st.error(f"Stock data directory not found: '{full_data_path}'. Please ensure 'extract_data.py' was run and 'stock_data_csvs' exists.")
        return pd.DataFrame(), []

    for filename in os.listdir(full_data_path):
        if filename.endswith('.csv'):
            symbol = os.path.splitext(filename)[0]
            stock_symbols.append(symbol)
            filepath = os.path.join(full_data_path, filename)
            try:
                df = pd.read_csv(filepath)
                df['Date'] = pd.to_datetime(df['Date'])
                df['Symbol'] = symbol # Add a symbol column
                all_stock_data.append(df)
            except Exception as e:
                st.warning(f"Could not load {filename}: {e}")

    if not all_stock_data:
        st.warning(f"No CSV files found in '{full_data_path}'. Please ensure the previous step generated CSVs.")
        return pd.DataFrame(), []

    master_df = pd.concat(all_stock_data, ignore_index=True)
    master_df = master_df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)

    # Drop duplicates here as well, crucial for time-series calculations
    initial_rows = len(master_df)
    master_df.drop_duplicates(subset=['Symbol', 'Date'], inplace=True)
    if len(master_df) < initial_rows:
        st.warning(f"Removed {initial_rows - len(master_df)} duplicate 'Symbol'-'Date' entries during data loading.")

    return master_df, stock_symbols

# --- Function to load and preprocess sector data ---
@st.cache_data # Cache the data loading for better performance
def load_and_preprocess_sector_data(file_path):
    """Loads and preprocesses the sector data CSV."""
    # Construct the full path to the sector data file
    full_file_path = os.path.join(os.getcwd(), file_path)

    if not os.path.exists(full_file_path):
        st.error(f"Sector data file not found: '{full_file_path}'. Please ensure 'Sector_data - Sheet1.csv' is in the correct path.")
        return pd.DataFrame()
    try:
        sector_df = pd.read_csv(full_file_path)

        # --- Extract actual Symbol from the 'Symbol' column in sector_df ---
        if 'Symbol' in sector_df.columns:
            sector_df['Extracted_Symbol'] = sector_df['Symbol'].apply(
                lambda x: x.split(': ')[1].strip() if isinstance(x, str) and ': ' in x and len(x.split(': ')) > 1 else x
            )
        else:
            st.warning("Warning: 'Symbol' column not found in Sector Data. Cannot extract ticker symbols for merging.")
            sector_df['Extracted_Symbol'] = sector_df['Symbol'] # Fallback

        # Ensure 'sector' column (lowercase) exists before returning
        if 'sector' not in sector_df.columns:
             st.error("Error: 'sector' column (lowercase) not found in Sector Data. Please ensure your CSV has this column.")
             return pd.DataFrame()

        # Drop duplicates on Extracted_Symbol to ensure unique merge keys
        sector_df.drop_duplicates(subset=['Extracted_Symbol'], inplace=True)

        return sector_df
    except Exception as e:
        st.error(f"Error loading or preprocessing sector data from '{full_file_path}': {e}")
        return pd.DataFrame()

# --- Main Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")

st.title("📊 Data-Driven Stock Analysis Dashboard")

# --- Load Data ---
st.header("1. Data Loading")
with st.spinner("Loading stock data..."):
    master_df, stock_symbols = load_all_stock_data(STOCK_DATA_DIR)
with st.spinner("Loading sector data..."):
    sector_df = load_and_preprocess_sector_data(SECTOR_DATA_PATH)

# Merge master_df with sector_df to add sector information
if not master_df.empty and not sector_df.empty and 'Extracted_Symbol' in sector_df.columns and 'sector' in sector_df.columns:
    master_df = pd.merge(master_df, sector_df[['Extracted_Symbol', 'sector']].drop_duplicates(subset=['Extracted_Symbol']),
                         left_on='Symbol', right_on='Extracted_Symbol', how='left')
    master_df.drop(columns=['Extracted_Symbol'], inplace=True)
    master_df.rename(columns={'sector': 'Sector'}, inplace=True)
    
    # Check if merge resulted in NaNs and warn
    if master_df['Sector'].isnull().any():
        st.warning("Some stocks have missing sector information after merging. Please ensure all stock symbols in your CSVs have a corresponding entry in 'Sector_data - Sheet1.csv' with a valid 'sector' value.")
    
    st.success("Data loaded and merged successfully!")
    st.subheader("Combined Stock Data Sample (with Sector)")
    st.dataframe(master_df.head())
else:
    st.error("Failed to load or merge all necessary data. Please check the data files and paths.")
    st.stop() # Stop execution if data loading fails

st.markdown("---")

# --- 2. Key Metrics Calculation ---
st.header("2. Key Performance Metrics")

# Calculate Daily Returns
master_df['Daily_Return'] = master_df.groupby('Symbol')['Close'].pct_change()

# Calculate Yearly Return for each stock
yearly_returns = {}
for symbol in stock_symbols:
    stock_df = master_df[master_df['Symbol'] == symbol].copy()
    if not stock_df.empty and len(stock_df) > 1:
        first_close = stock_df.iloc[0]['Close']
        last_close = stock_df.iloc[-1]['Close']
        if first_close != 0:
            yearly_return = ((last_close - first_close) / first_close) * 100
            yearly_returns[symbol] = yearly_return
        else:
            yearly_returns[symbol] = 0.0
    else:
        yearly_returns[symbol] = np.nan

yearly_returns_df = pd.DataFrame(list(yearly_returns.items()), columns=['Symbol', 'Yearly_Return_Pct'])
yearly_returns_df = yearly_returns_df.dropna(subset=['Yearly_Return_Pct'])

st.subheader("Yearly Returns for Stocks")
st.dataframe(yearly_returns_df.sort_values(by='Yearly_Return_Pct', ascending=False).head(10))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Green Stocks (Best Performing)")
    top_10_green_stocks = yearly_returns_df.sort_values(by='Yearly_Return_Pct', ascending=False).head(10)
    st.dataframe(top_10_green_stocks)

with col2:
    st.subheader("Top 10 Red Stocks (Worst Performing)")
    top_10_red_stocks = yearly_returns_df.sort_values(by='Yearly_Return_Pct', ascending=True).head(10)
    st.dataframe(top_10_red_stocks)

st.subheader("Market Summary")
total_stocks_with_returns = len(yearly_returns_df)
green_stocks_count = len(yearly_returns_df[yearly_returns_df['Yearly_Return_Pct'] >= 0])
red_stocks_count = total_stocks_with_returns - green_stocks_count

if total_stocks_with_returns > 0:
    st.info(f"Total Stocks Analyzed (with valid yearly returns): **{total_stocks_with_returns}**")
    st.success(f"Number of Green Stocks (Positive Yearly Return): **{green_stocks_count}** ({green_stocks_count / total_stocks_with_returns * 100:.2f}%)")
    st.error(f"Number of Red Stocks (Negative Yearly Return): **{red_stocks_count}** ({red_stocks_count / total_stocks_with_returns * 100:.2f}%)")
else:
    st.warning("Cannot calculate percentages as no stocks with valid returns were found.")

average_close_price = master_df['Close'].mean()
average_volume = master_df['Volume'].mean()
st.write(f"Average Close Price Across All Stocks: **{average_close_price:.2f}**")
st.write(f"Average Volume Across All Stocks: **{average_volume:.2f}**")

st.markdown("---")

# --- 3. Volatility Analysis ---
st.header("3. Volatility Analysis")

volatility = master_df.groupby('Symbol')['Daily_Return'].std().dropna()
volatility_df = volatility.reset_index(name='Volatility')

st.subheader("Top 10 Most Volatile Stocks")
top_10_volatile_stocks = volatility_df.sort_values(by='Volatility', ascending=False).head(10)
st.dataframe(top_10_volatile_stocks)

fig_volatility, ax_volatility = plt.subplots(figsize=(12, 7))
ax_volatility.bar(top_10_volatile_stocks['Symbol'], top_10_volatile_stocks['Volatility'], color='skyblue')
ax_volatility.set_xlabel('Stock Symbol')
ax_volatility.set_ylabel('Volatility (Standard Deviation of Daily Returns)')
ax_volatility.set_title('Top 10 Most Volatile Stocks')
ax_volatility.tick_params(axis='x', rotation=45)
ax_volatility.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig_volatility) # Display plot in Streamlit

st.markdown("---")

# --- 4. Cumulative Return Over Time ---
st.header("4. Cumulative Return Over Time")

master_df['Cumulative_Return'] = master_df.groupby('Symbol')['Daily_Return'].transform(lambda x: (1 + x).cumprod() - 1)
final_cumulative_returns = master_df.groupby('Symbol')['Cumulative_Return'].last().sort_values(ascending=False)
top_5_performing_symbols = final_cumulative_returns.head(5).index.tolist()

st.subheader("Top 5 Performing Stocks by Cumulative Return")
st.dataframe(final_cumulative_returns.head(5))

fig_cumulative, ax_cumulative = plt.subplots(figsize=(14, 8))
for symbol in top_5_performing_symbols:
    stock_data = master_df[master_df['Symbol'] == symbol].copy()
    ax_cumulative.plot(stock_data['Date'], stock_data['Cumulative_Return'], label=symbol)

ax_cumulative.set_xlabel('Date')
ax_cumulative.set_ylabel('Cumulative Return')
ax_cumulative.set_title('Cumulative Return Over Time for Top 5 Performing Stocks')
ax_cumulative.legend(title='Stock Symbol')
ax_cumulative.grid(True, linestyle='--', alpha=0.7)
ax_cumulative.tick_params(axis='x', rotation=45)
st.pyplot(fig_cumulative) # Display plot in Streamlit

st.markdown("---")

# --- 5. Sector-wise Performance ---
st.header("5. Sector-wise Performance Analysis")

if 'Sector' in master_df.columns and not master_df['Sector'].isnull().all():
    sector_performance_df = pd.merge(yearly_returns_df, master_df[['Symbol', 'Sector']].drop_duplicates(),
                                     on='Symbol', how='left')
    sector_performance_df.dropna(subset=['Sector'], inplace=True)

    if not sector_performance_df.empty:
        average_yearly_return_by_sector = sector_performance_df.groupby('Sector')['Yearly_Return_Pct'].mean().sort_values(ascending=False)
        
        st.subheader("Average Yearly Return by Sector")
        st.dataframe(average_yearly_return_by_sector)

        fig_sector, ax_sector = plt.subplots(figsize=(14, 8))
        average_yearly_return_by_sector.plot(kind='bar', color='lightgreen', ax=ax_sector)
        ax_sector.set_xlabel('Sector')
        ax_sector.set_ylabel('Average Yearly Return (%)')
        ax_sector.set_title('Average Yearly Return by Sector')
        ax_sector.tick_params(axis='x', rotation=45)
        ax_sector.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig_sector) # Display plot in Streamlit
    else:
        st.warning("No valid sector data found after dropping missing values. Skipping Sector-wise Performance analysis.")
else:
    st.warning("'Sector' column not available or contains all NaN values. Skipping Sector-wise Performance analysis.")

st.markdown("---")

# --- 6. Stock Price Correlation ---
st.header("6. Stock Price Correlation Analysis")

close_prices_pivot = master_df.pivot(index='Date', columns='Symbol', values='Close')
daily_returns_pivot = close_prices_pivot.pct_change().dropna()

if not daily_returns_pivot.empty:
    correlation_matrix = daily_returns_pivot.corr()
    
    st.subheader("Stock Price Correlation Matrix (Sample)")
    st.dataframe(correlation_matrix.head()) # Display a sample of the matrix

    fig_corr, ax_corr = plt.subplots(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Stock Price Correlation Heatmap (Based on Daily Returns)')
    ax_corr.tick_params(axis='x', rotation=90)
    ax_corr.tick_params(axis='y', rotation=0)
    st.pyplot(fig_corr) # Display plot in Streamlit
else:
    st.warning("Not enough data to calculate stock price correlation. Skipping correlation analysis.")

st.markdown("---")

# --- 7. Top 5 Gainers and Losers (Month-wise) ---
st.header("7. Monthly Top 5 Gainers and Losers")

master_df['YearMonth'] = master_df['Date'].dt.to_period('M')
monthly_first_last_close = master_df.groupby(['Symbol', 'YearMonth'])['Close'].agg(['first', 'last'])

monthly_first_last_close['Monthly_Return_Pct'] = (
    (monthly_first_last_close['last'] - monthly_first_last_close['first']) /
    monthly_first_last_close['first'].replace(0, np.nan)
)
monthly_returns = monthly_first_last_close.reset_index()
monthly_returns['Monthly_Return_Pct'] = monthly_returns['Monthly_Return_Pct'].fillna(0)

if not monthly_returns.empty:
    unique_months = sorted(monthly_returns['YearMonth'].unique())
    
    selected_month = st.selectbox("Select a Month to View Gainers/Losers:", unique_months)

    monthly_data = monthly_returns[monthly_returns['YearMonth'] == selected_month].copy()
    if not monthly_data.empty:
        top_5_gainers = monthly_data.sort_values(by='Monthly_Return_Pct', ascending=False).head(5)
        top_5_losers = monthly_data.sort_values(by='Monthly_Return_Pct', ascending=True).head(5)

        plot_data = pd.concat([top_5_gainers, top_5_losers]).sort_values(by='Monthly_Return_Pct', ascending=False)

        if not plot_data.empty:
            fig_monthly, ax_monthly = plt.subplots(figsize=(10, 6))
            colors = ['green' if x >= 0 else 'red' for x in plot_data['Monthly_Return_Pct']]
            ax_monthly.bar(plot_data['Symbol'], plot_data['Monthly_Return_Pct'] * 100, color=colors)
            ax_monthly.set_xlabel('Stock Symbol')
            ax_monthly.set_ylabel('Monthly Return (%)')
            ax_monthly.set_title(f'Top 5 Gainers and Losers for {selected_month.strftime("%B %Y")}')
            ax_monthly.tick_params(axis='x', rotation=45)
            ax_monthly.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig_monthly) # Display plot in Streamlit
        else:
            st.info(f"No data to plot for {selected_month}.")
    else:
        st.info(f"No data for {selected_month} to determine gainers/losers.")
else:
    st.warning("Not enough data to calculate monthly returns. Skipping monthly gainers/losers analysis.")

st.markdown("---")
st.success("All analysis and visualizations are integrated into the dashboard!")
