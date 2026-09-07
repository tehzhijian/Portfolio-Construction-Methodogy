import pandas as pd
import numpy as np
from scipy.optimize import minimize
from opti_port import PortfolioOptimizer

class VarianceSelector:
    """
    Implements 'Algorithm 1':
    1. Filters stocks based on a price threshold (5% of capital).
    2. Ranks remaining stocks by Variance (lowest to highest).
    3. Selects the top N (default 25) stocks.
    4. Optimizes allocation using PortfolioOptimizer.
    """

    def __init__(self, capital=5000, price_threshold_ratio=0.05, top_n=25):
        self.capital = capital
        self.threshold = capital * price_threshold_ratio
        self.top_n = top_n
        self.optimizer = PortfolioOptimizer()

    def process(self, df_results):
        """
        Ingests the analysis dataframe, applies filters/ranking, and runs optimization.
        
        :param df_results: DataFrame with columns: 
                           ['ticker', 'latest_price', 'expected_return', 'daily_variance', ...]
        :return: Tuple (DataFrame of final allocation, float remaining_capital)
        """
        # --- Step 1: Data Standardization ---
        # Map your specific columns to the internal names expected by the algorithm
        # using 'daily_variance' for risk ranking as per standard CAPM workflows
        df_std = df_results.copy()
        
        # Ensure 'ticker' is the index for easy lookup
        if 'ticker' in df_std.columns:
            df_std.set_index('ticker', inplace=True)
            
        # Check for necessary columns
        required_map = {
            'latest_price': 'Price', 
            'expected_return': 'Mean Return', 
            'daily_variance': 'Variance'
        }
        
        # validate columns exist
        missing_cols = [col for col in required_map.keys() if col not in df_std.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")
            
        # Rename for internal consistency
        df_std.rename(columns=required_map, inplace=True)

        # --- Step 2: Filtering & Ranking (Algorithm 1 Logic) ---
        
        # Filter 1: Price Threshold
        # -> "t_dic[i][0] <= threshold"
        eligible_df = df_std[df_std['Price'] <= self.threshold].copy()
        
        if eligible_df.empty:
            print(f"No stocks found under price threshold ${self.threshold:.2f}")
            return pd.DataFrame(), self.capital

        # Filter 2: Rank by Variance (Lowest to Highest)
        # -> "sorted(values_at_position_2, key=lambda x: x[1])"
        ranked_df = eligible_df.sort_values(by='Variance', ascending=True)
        
        # Select Top N
        portfolio_subset = ranked_df.head(self.top_n)
        
        # --- Step 3: Format Data for Optimizer ---
        # Convert to Dictionary: {Ticker: [Price, Mean, Variance]}
        stocks_dict = {}
        for ticker, row in portfolio_subset.iterrows():
            stocks_dict[ticker] = [row['Price'], row['Mean Return'], row['Variance']]
            
        tickers = list(stocks_dict.keys())
        n_stocks = len(tickers)
        variances = np.array([x[2] for x in stocks_dict.values()]) # Index 2 is Variance
        prices = {k: v[0] for k, v in stocks_dict.items()}         # Index 0 is Price

        print(f"Selected {n_stocks} stocks for optimization (Filter: Price <= ${self.threshold:.2f}, Rank: Lowest Daily Var).")

        # --- Step 4: Run Grid Search Optimization ---
        # Finds best N (min weight) and M (max weight) to minimize leftover cash
        best_n, best_m, remaining_capital = self.optimizer.test_opti_tune(
            n_stocks, variances, stocks_dict, prices, self.capital
        )
        
        print(f"Optimal Constraints: Min % {best_n:.2%} | Max % {best_m:.2%}")

        # --- Step 5: Final Allocation Calculation ---
        final_allocation_df = self._get_final_allocation(
            n_stocks, variances, stocks_dict, prices, best_n, best_m
        )
        
        # Verify exact spend
        if not final_allocation_df.empty:
            total_spent = final_allocation_df['Actual Cost'].sum()
            final_remaining = self.capital - total_spent
        else:
            final_remaining = self.capital

        return final_allocation_df, final_remaining

    def _get_final_allocation(self, n_stocks, variances, stocks, prices, n, m):
        """
        Internal helper to re-run minimization with best parameters 
        to get exact share counts.
        """
        # - Replicating minimize logic to extract weights
        initial_weights = np.array([1/n_stocks] * n_stocks)
        cov_matrix = np.diag(variances)
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = [(n, m) for _ in range(n_stocks)]
        
        result = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        allocation_data = []
        optimized_weights = dict(zip(stocks.keys(), result.x))
        
        for ticker, weight in optimized_weights.items():
            share_price = prices[ticker]
            allocated_amt = weight * self.capital
            num_shares = int(np.floor(allocated_amt / share_price))
            actual_cost = num_shares * share_price
            
            # Only include stocks we actually buy
            if num_shares > 0:
                allocation_data.append({
                    "Ticker": ticker,
                    "Price": share_price,
                    "Optimized Weight": round(weight, 4),
                    "Shares": num_shares,
                    "Actual Cost": actual_cost
                })
        
        return pd.DataFrame(allocation_data)