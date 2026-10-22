# Bloomberg-Global-Trading-Challenge

Heyy, so I built a working prototype in Python:

* Downloaded 1 year of daily prices from Yahoo Finance for 10 US stocks, plus ACWI as a proxy for the WLS index (WLS isn’t on Yahoo).
* Converted the prices to weekly data and built one row per stock per week with 6 features: last-week excess return vs the index, 4-week and 12-week returns, 20-day volatility, 60-day beta, and a sector “cyclicality” score (defensive / sensitive / cyclical).
* Label = 1 if the stock beat the index the following week, 0 otherwise.
* Split by time (older weeks for training, most recent 30% for testing, no shuffling) and trained a logistic regression and a random forest.

Results Neither model beat the simple baseline. Test accuracy was 50–57%, against 60% for always predicting “beats the index”. That isn’t surprising: ~270 training rows from 10 stocks over one year is far too little data for weekly returns, which are very noisy. The useful outcome is that the full pipeline works (prices → features → labels → model → predictions), so we can now scale it up.

What we need next
1. More and better data. Many more stocks (100+, ideally global, since the benchmark is global), at least 3–5 years of history, and ideally data from Bloomberg: the actual WLS index, historical index members, earnings dates and analyst revisions.
2. Better features. We should pick features that the research shows actually predict returns, rather than guessing. Strong candidates: 12-month momentum (skipping the last month), short-term reversal, distance from the 52-week high, sector momentum, and upcoming earnings announcements. A good starting point is Gu, Kelly & Xiu (2020), “Empirical Asset Pricing via Machine Learning”.
