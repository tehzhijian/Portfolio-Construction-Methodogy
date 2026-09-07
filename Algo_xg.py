import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from data_collection import CAPMProcessor
from opti_port import PortfolioOptimizer

class SmartPortfolioBuilder(CAPMProcessor):
    """
    A portfolio builder that optimizes data fetching (Daily ONLY) and uses 
    XGBoost for rolling variance feature selection.
    """

    def __init__(self, tickers, markets, start_date, end_date, risk_free_rate=0.0365):
        super().__init__(tickers, markets, start_date, end_date, risk_free_rate)
        self.optimizer = PortfolioOptimizer()
        self.rolling_volatility = None
        self.selected_tickers = []

    def fetch_stock_data(self):
        """
        Overrides the parent method to fetch ONLY daily data ('1d').
        Strictly skips monthly data fetching.
        """
        print("\n=== Fetching Daily Stock Data (Optimized) ===")
        # Fetch only 1d interval
        raw_daily = self.fetcher.fetch_data(self.tickers, self.start_date, self.end_date, '1d')
        
        # Extract Close prices using parent helper method
        self.daily_stock_prices = self._extract_close(raw_daily, self.tickers)
        
        # Explicitly set monthly to Empty to ensure no monthly calls are made
        self.monthly_stock_prices = pd.DataFrame()
        
        print("=== Daily Stock Data Fetch Complete ===\n")

    def calculate_rolling_variance(self, window=90):
        """
        Calculates the aggregate rolling volatility (target) using only daily data.
        """
        # Auto-fetch if data is missing
        if self.daily_stock_prices is None or self.daily_stock_prices.empty:
            self.fetch_stock_data()

        # 1. Calculate Daily Returns
        # Drop the first NaN row created by diff()
        t_change = self.daily_stock_prices.diff().iloc[1:].fillna(0)
        
        # 2. Calculate Target: Aggregate Rolling Volatility
        # Logic: Rolling STD of returns -> Mean across all stocks = Market Noise Level
        self.rolling_volatility = t_change.rolling(window=window).std().mean(axis=1)
        
        # 3. Align Features and Target
        # The rolling window creates NaNs at the start. Trim X to start where y becomes valid.
        X = t_change.iloc[window-1:]
        y = self.rolling_volatility.dropna()
        
        # Ensure exact index alignment
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        return X, y

    def get_feature_importance(self, window=90, n_selection=25):
        """
        Uses XGBoost to find stocks that contribute LEAST to aggregate volatility.
        """
        print(f"\n=== Analyzing Feature Importance (Window: {window} days) ===")
        
        X, y = self.calculate_rolling_variance(window)
        
        if X.empty:
            print("Error: Insufficient data for the specified rolling window.")
            return []

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost Regressor
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        # Extract Importance
        importances = xgb_model.feature_importances_
        
        feat_df = pd.DataFrame({
            'ticker': X.columns,
            'importance': importances
        })
        
        # Select bottom N tickers (Lowest importance = Least predictive of market noise)
        self.selected_tickers = feat_df.sort_values(by='importance', ascending=True)['ticker'].head(n_selection).tolist()
        
        print(f"Top 5 Low-Volatility Selections: {self.selected_tickers[:5]}")
        return self.selected_tickers

    def _get_stock_statistics(self):
        """
        Internal helper to calculate statistics using ONLY daily data.
        """
        stats = {}
        if self.daily_stock_prices is None:
            return stats
            
        # Daily returns for variance calculation
        daily_ret = self.daily_stock_prices.diff().iloc[1:].fillna(0)
        
        for ticker in self.selected_tickers:
            if ticker in self.daily_stock_prices.columns:
                # Get Latest Price
                price = self.daily_stock_prices[ticker].iloc[-1]
                
                # Get Variance of returns over the period
                variance = daily_ret[ticker].var()
                
                stats[ticker] = {
                    'price': price,
                    'variance': variance
                }
        return stats

    def optimize_portfolio(self, capital, price_threshold_ratio=0.05, n_range=(0.01, 0.10), m_range=(0.10, 0.50)):
        """
        Allocates weights and returns a DataFrame of the portfolio.
        """
        if not self.selected_tickers:
            print("No stocks selected. Running feature selection...")
            self.get_feature_importance()

        print("\n=== Optimizing Portfolio Weights ===")
        
        stats = self._get_stock_statistics()
        
        # Filter by Price Threshold
        price_limit = capital * price_threshold_ratio
        eligible_data = {}
        
        for ticker, data in stats.items():
            if data['price'] <= price_limit:
                # Structure: [Price, Expected Return (0 placeholder), Variance]
                eligible_data[ticker] = [data['price'], 0.0, data['variance']]
                
        final_tickers = list(eligible_data.keys())
        print(f"Eligible stocks (Price < {price_limit:.2f}): {len(final_tickers)}")
        
        if not final_tickers:
            print("No stocks suitable for optimization.")
            return pd.DataFrame(), capital

        # Prepare Optimizer Inputs
        variances = np.array([v[2] for v in eligible_data.values()])
        prices = {k: v[0] for k, v in eligible_data.items()}
        n_stocks = len(final_tickers)

        # 1. Run Optimization to find Best Bounds (n, m)
        best_n, best_m, _ = self.optimizer.test_opti_tune(
            n_stocks=n_stocks,
            variances=variances,
            stocks=eligible_data,
            prices=prices,
            capital=capital,
            n_range=n_range,
            m_range=m_range
        )
        
        if best_n is None:
            print("Optimization found no valid solution.")
            return pd.DataFrame(), capital

        print(f"\nOptimization Successful.")
        print(f"Best Bounds -> Min %: {best_n:.2f}, Max %: {best_m:.2f}")

        # 2. Re-calculate Specific Weights to Build the Table
        # (This logic reconstructs the portfolio details since opti_tune only returned remaining capital)
        cov_matrix = np.diag(variances)
        initial_weights = np.array([1/n_stocks] * n_stocks)
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(best_n, best_m) for _ in range(n_stocks)]
        
        result = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_weights = result.x
        
        # 3. Construct the DataFrame
        portfolio_data = []
        for idx, ticker in enumerate(final_tickers):
            price = prices[ticker]
            weight = opt_weights[idx]
            allocated_amt = weight * capital
            shares = int(np.floor(allocated_amt / price))
            cost = shares * price
            
            if shares > 0:
                portfolio_data.append({
                    "Ticker": ticker,
                    "Price": price,
                    "Optimized Weight": round(weight, 4),
                    "Shares": shares,
                    "Actual Cost": cost
                })
        
        df_portfolio = pd.DataFrame(portfolio_data)
        
        # Calculate final remaining capital from the dataframe to be exact
        total_invested = df_portfolio['Actual Cost'].sum() if not df_portfolio.empty else 0
        remaining_cap = capital - total_invested
        
        print(f"Projected Unused Capital: ${remaining_cap:.2f}")
        
        return df_portfolio, remaining_cap