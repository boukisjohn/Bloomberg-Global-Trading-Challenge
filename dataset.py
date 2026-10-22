import yfinance as yf
import numpy as np
import pandas as pd

market_index = 'ACWI'
data = yf.download(tickers=['NVDA', 'AMD', 'TSLA', 'AAPL',
                            'META', 'JPM', 'XOM', 'LLY', 'CAT', 'KO',
                            market_index], period='1y', auto_adjust=True)

# Keep only the Daily Closing Price
daily_prices = data['Close']

# Mon-Fri and if not Fri, then the previous day.
weekly_prices = daily_prices.resample('W-FRI').last()
# print(weekly_prices.head())
print(f'We expect 53 weeks for 10 (+1) stocks: {weekly_prices.shape}.\n')

# pct_change: percentage change tomorrow/today - 1.
weekly_returns = weekly_prices.pct_change(1)
# print(weekly_returns.head())
next_week_returns = weekly_returns.shift(-1)
# print(next_week_returns.head())

# To see if the stocks "won" the market index next week or not.
# (Doesn't mean that they performed better, it could be that the market went down.)
excess = next_week_returns.sub(next_week_returns[market_index], axis=0)
label = (excess > 0).astype(int)
# Otherwise the code "predicts" what will happen next week.
label = label.where(excess.notna())

# FEATURE EXTRACTION

# WEEKLY FEATURES

# Feature 1.
# Excess return (stock - market) of THIS week.
excess_1w = weekly_returns.sub(weekly_returns[market_index], axis=0)

# Feature 2 & 3.
# Comparing the Returns Now with the ones from 4 weeks and 12 weeks ago (Momentum).
return_4w = weekly_prices.pct_change(4)
return_12w = weekly_prices.pct_change(12)

# DAILY FEATURES

# Feature 4.
# 20 Days Volatility.

# Calculating Daily Returns
daily_returns = daily_prices.pct_change()
# Last 20 days to calculate stdev, *sqrt(252) to do it yearly.
volatility_daily = daily_returns.rolling(20).std()*np.sqrt(252)
volatility_20d = volatility_daily.resample('W-FRI').last()

# Feature 5.
# Beta. Shows how fast the stock moves compared to the market.
# =1 same speed, =2 for 1% up the stock goes 2%, =0.5 half etc.
# Beta = covariance(stock, market)/ variance(market).
# We use a sample space of 60 days and for every day onwards we calculate beta every day.

# Covariance of each stock with the market.
covariance = daily_returns.rolling(60).cov(daily_returns[market_index])

# Variance of the Market Index.
variance = daily_returns[market_index].rolling(60).var()

# Beta calculation.
beta_daily = covariance.div(variance, axis=0)

# End of the week Beta.
beta_60d = beta_daily.resample('W-FRI').last()

# Feature table creation.
tables = {
    'excess_1w': excess_1w,
    'return_4w': return_4w,
    'return_12w': return_12w,
    'volatility_20d': volatility_20d,
    'beta_60d': beta_60d,
    'label': label
}

dataset = pd.concat({name: df.drop(columns=market_index).stack() for name, df in tables.items()}, axis=1)
dataset.index.names = ['date', 'ticker']

dataset = dataset.reset_index()
sectors = {'NVDA': 'Tech', 'AMD': 'Tech', 'AAPL': 'Tech', 'META': 'Comm. Services',
           'TSLA': 'Consumer Disc.', 'JPM': 'Financials', 'XOM': 'Energy',
           'LLY': 'Health Care', 'CAT': 'Industrials', 'KO': 'Consumer Staples'}

dataset['sector'] = dataset['ticker'].map(sectors)
dataset = dataset.sort_values(['date', 'ticker']).reset_index(drop=True)

dataset = dataset[['date', 'ticker', 'excess_1w', 'return_4w', 'return_12w',
                   'volatility_20d', 'beta_60d', 'sector', 'label']]

# Save dataset to an Excel file (xlsx).
# dataset.to_excel('dataset_1st.xlsx', index=False)
features = ['excess_1w', 'return_4w', 'return_12w','volatility_20d', 'beta_60d', 'sector']

# Drop any rows with a NaN value.
dataset = dataset.dropna(subset=features).reset_index(drop=True)

# Because the algorithm will not understand the string type for sector,
# I chose to assign a value to every one of them according to
# how much they are affected by the economy.

sector_score = {'Consumer Staples': 1, 'Health Care':1,
                'Tech': 2, 'Comm. Services': 2, 'Industrials': 2, 'Energy': 2,
                'Consumer Disc.': 3, 'Financials': 3}

dataset['cyclicality'] = dataset['sector'].map(sector_score)
dataset = dataset.drop(columns='sector')
dataset = dataset[['date', 'ticker', 'excess_1w', 'return_4w', 'return_12w',
                   'volatility_20d', 'beta_60d', 'cyclicality', 'label']]

# dataset.to_excel('dataset_clean.xlsx', index=False)

