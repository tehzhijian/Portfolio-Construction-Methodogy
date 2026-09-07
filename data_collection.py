import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ==========================================
# PART 1: ROBUST DATA FETCHER (YFinance)
# ==========================================
class RobustDataFetcher:
    def __init__(self, min_delay=2, max_delay=5):
        self.min_delay = min_delay
        self.max_delay = max_delay

    def chunk_tickers(self, tickers, n=20):
        for i in range(0, len(tickers), n):
            yield tickers[i:i + n]

    def _apply_buffer(self):
        sleep_time = random.uniform(self.min_delay, self.max_delay)
        time.sleep(sleep_time)

    def fetch_data(self, tickers, start_date, end_date, interval='1d'):
        all_data_list = []
        unique_tickers = list(set(tickers))
        chunks = list(self.chunk_tickers(unique_tickers, n=20)) 
        
        print(f"Initializing download for {len(unique_tickers)} unique tickers...")
        
        for i, chunk in enumerate(tqdm(chunks, desc=f"Downloading {interval}", unit="batch")):
            chunk_str = " ".join(chunk)
            try:
                data = yf.download(
                    chunk_str, 
                    start=start_date, 
                    end=end_date, 
                    group_by='ticker', 
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    threads=False 
                )
                if not data.empty:
                    all_data_list.append(data)
                self._apply_buffer()
            except Exception as e:
                tqdm.write(f"  Warning: Batch {i+1} failed: {e}")
                self._apply_buffer()
        
        if not all_data_list:
            return pd.DataFrame()
        return pd.concat(all_data_list, axis=1)

# ==========================================
# PART 2: FUNDAMENTAL SCRAPER (Finviz)
# ==========================================
class FundamentalScraper:
    def __init__(self):
        # Header required to mimic a browser and avoid 403 Forbidden errors
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
        }
        self.base_url = 'https://finviz.com/quote.ashx?t='

    def _clean_value(self, val_str):
        """Converts string '20.5%' -> float 20.5 or '-' -> 0.0"""
        if not val_str or val_str == '-':
            return 0.0
        try:
            return float(val_str.strip('%'))
        except ValueError:
            return 0.0

    def fetch_ticker_data(self, ticker):
        """Scrapes specific ratios for a single ticker."""
        try:
            response = requests.get(self.base_url + ticker, headers=self.headers, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            data = {'ticker': ticker}
            
            # Map Finviz Labels to DataFrame Columns
            metrics_map = {
                'ROA': 'ROA',
                'ROE': 'ROE',
                'ROI': 'ROI',
                'Gross Margin': 'Gross Margin',
                'Oper. Margin': 'Operating Margin',
                'Profit Margin': 'Profit Margin'
            }

            for label, col_name in metrics_map.items():
                td = soup.find('td', string=label)
                if td:
                    val_str = td.find_next_sibling('td').text.strip()
                    data[col_name] = self._clean_value(val_str)
                else:
                    data[col_name] = 0.0
            
            return data
        except Exception as e:
            # Silent fail for individual tickers to keep process moving
            return None

    def fetch_batch(self, tickers):
        """Iterates through tickers with a progress bar."""
        results = []
        print(f"\n--- Scraping Fundamentals (Finviz) for {len(tickers)} tickers ---")
        
        for ticker in tqdm(tickers, unit="ticker"):
            data = self.fetch_ticker_data(ticker)
            if data:
                results.append(data)
            # Politeness delay to prevent blocking
            time.sleep(random.uniform(0.5, 1.0))
            
        return pd.DataFrame(results)

# ==========================================
# PART 3: CAPM PROCESSOR (Integrated)
# ==========================================
class CAPMProcessor:
    def __init__(self, tickers, markets, start_date, end_date, risk_free_rate=0.0365):
        # Deduplicate inputs
        self.tickers = sorted(list(set(tickers)))
        self.markets = sorted(list(set(markets)))
        
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        
        # Tools
        self.fetcher = RobustDataFetcher()
        self.scraper = FundamentalScraper()
        
        # Data Stores
        self.daily_stock_prices = None
        self.daily_market_prices = None
        self.monthly_stock_prices = None
        self.monthly_market_prices = None
        self.fundamental_df = None
        self.metrics_df = None

    def _extract_close(self, data, ticker_list):
        if data.empty: return pd.DataFrame()
        
        # Case 1: Single Ticker
        if len(ticker_list) == 1:
            ticker = ticker_list[0]
            if ticker in data.columns:
                val = data[ticker]
                if isinstance(val, pd.DataFrame) and 'Close' in val.columns:
                    return pd.DataFrame({ticker: val['Close']})
                elif isinstance(val, pd.Series):
                    return pd.DataFrame({ticker: val})
            if 'Close' in data.columns:
                 return pd.DataFrame({ticker: data['Close']})
            return pd.DataFrame()

        # Case 2: Multiple Tickers
        try:
            clean_data = {}
            for t in ticker_list:
                if t in data.columns:
                    ticker_data = data[t]
                    if 'Close' in ticker_data.columns:
                        close_vals = ticker_data['Close']
                        if isinstance(close_vals, pd.DataFrame):
                            close_vals = close_vals.iloc[:, 0]
                        clean_data[t] = close_vals
            return pd.DataFrame(clean_data)
        except Exception as e:
            print(f"Error extracting close prices: {e}")
            return pd.DataFrame()

    def fetch_market_data(self):
        print("\n=== Fetching Market Data ===")
        raw_daily = self.fetcher.fetch_data(self.markets, self.start_date, self.end_date, '1d')
        self.daily_market_prices = self._extract_close(raw_daily, self.markets)
        
        raw_monthly = self.fetcher.fetch_data(self.markets, self.start_date, self.end_date, '1mo')
        self.monthly_market_prices = self._extract_close(raw_monthly, self.markets)
        print("=== Market Data Fetch Complete ===\n")

    def fetch_stock_data(self):
        print("\n=== Fetching Stock Data ===")
        raw_daily = self.fetcher.fetch_data(self.tickers, self.start_date, self.end_date, '1d')
        self.daily_stock_prices = self._extract_close(raw_daily, self.tickers)
        
        raw_monthly = self.fetcher.fetch_data(self.tickers, self.start_date, self.end_date, '1mo')
        self.monthly_stock_prices = self._extract_close(raw_monthly, self.tickers)
        print("=== Stock Data Fetch Complete ===\n")

    def fetch_fundamental_data(self):
        """Scrapes fundamental data from Finviz."""
        print("\n=== Fetching Fundamental Data ===")
        self.fundamental_df = self.scraper.fetch_batch(self.tickers)
        print("=== Fundamental Data Fetch Complete ===\n")

    def calculate_metrics(self):
        # Validation
        if self.daily_stock_prices is None or self.monthly_stock_prices is None:
            print("Error: Missing Stock Data. Run .fetch_stock_data()")
            return pd.DataFrame()
        if self.daily_market_prices is None or self.monthly_market_prices is None:
            print("Error: Missing Market Data. Run .fetch_market_data()")
            return pd.DataFrame()

        # Returns Calculation
        daily_stock_ret = self.daily_stock_prices.diff().iloc[1:].fillna(0)
        monthly_stock_ret = self.monthly_stock_prices.diff().iloc[1:].fillna(0)
        monthly_market_ret = self.monthly_market_prices.diff().iloc[1:].fillna(0)

        valid_tickers = [t for t in self.tickers if t in daily_stock_ret.columns]
        valid_markets = [m for m in self.markets if m in monthly_market_ret.columns]
        
        common_dates = monthly_stock_ret.index.intersection(monthly_market_ret.index)
        monthly_stock_ret = monthly_stock_ret.loc[common_dates]
        monthly_market_ret = monthly_market_ret.loc[common_dates]

        results = []
        market_vars = monthly_market_ret[valid_markets].var()

        for ticker in valid_tickers:
            prices = self.daily_stock_prices[ticker].dropna()
            if prices.empty: continue

            latest_price = prices.iloc[-1]
            daily_mean = daily_stock_ret[ticker].mean()
            daily_var = daily_stock_ret[ticker].var()
            
            monthly_mean = monthly_stock_ret[ticker].mean()
            monthly_std = monthly_stock_ret[ticker].std()

            record = {
                'ticker': ticker,
                'latest_price': latest_price,
                'expected_return': daily_mean,
                'daily_variance': daily_var,
                'monthly_sharpe': np.nan,
                'monthly_var': monthly_stock_ret[ticker].var()
            }

            if monthly_std > 1e-9:
                record['monthly_sharpe'] = (monthly_mean - self.risk_free_rate) / monthly_std

            for market in valid_markets:
                m_var = market_vars[market]
                cov = np.cov(monthly_market_ret[market], monthly_stock_ret[ticker])[0][1]
                beta = cov / m_var if m_var > 1e-9 else np.nan
                treynor = (monthly_mean - self.risk_free_rate) / beta if (beta and abs(beta) > 1e-9) else np.nan
                
                clean_m = market.replace('^', '').lower()
                record[f'beta_{clean_m}'] = beta
                record[f'treynor_{clean_m}'] = treynor

            results.append(record)

        self.metrics_df = pd.DataFrame(results)

        # Merge Fundamental Data if available
        if self.fundamental_df is not None and not self.fundamental_df.empty:
            print("Merging Fundamental Data into Results...")
            # Left join to keep all CAPM results even if scraping failed for some
            self.metrics_df = self.metrics_df.merge(self.fundamental_df, on='ticker', how='left')
            # Fill missing fundamentals with 0
            fund_cols = ['ROA', 'ROE', 'ROI', 'Gross Margin', 'Operating Margin', 'Profit Margin']
            for col in fund_cols:
                if col in self.metrics_df.columns:
                    self.metrics_df[col] = self.metrics_df[col].fillna(0)

        return self.metrics_df