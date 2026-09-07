import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

class MVPTOptimizer:
    def __init__(self, price_df):
        """
        Initializes with a DataFrame of daily stock close prices.
        """
        self.price_df = price_df
        # Standardize data: Daily percentage change for better variance scaling
        self.df_change = self.price_df.pct_change().iloc[1:].fillna(0)
        self.tickers = self.df_change.columns.tolist()

    def _portfolio_variance(self, weights, cov_matrix):
        """Objective function: $w^T \Sigma w$"""
        return weights.T @ cov_matrix @ weights

    def _get_allocation_details(self, selected_tickers, weights, capital):
        """
        Generates a DataFrame with shares, value, and adjusted weights.
        """
        if not selected_tickers or len(weights) == 0:
            return pd.DataFrame(), 0.0

        latest_prices = self.price_df[selected_tickers].iloc[-1].values
        allocated_capital = capital * np.array(weights)
        num_shares = np.floor(allocated_capital / latest_prices).astype(int)
        
        actual_cost = num_shares * latest_prices
        actual_capital_used = np.sum(actual_cost)
        
        adjusted_weights = actual_cost / actual_capital_used if actual_capital_used > 0 else np.zeros_like(weights)

        allocation_df = pd.DataFrame({
            'Ticker': selected_tickers,
            'Original_Weight': weights,
            'Shares': num_shares,
            'Price': latest_prices,
            'Actual Cost': actual_cost, 
            'Adjusted_Weight': adjusted_weights
        })
        
        return allocation_df[allocation_df['Shares'] > 0].sort_values(by='Adjusted_Weight', ascending=False), actual_capital_used

    def optimize_ledoit_wolf(self, capital=5000):
        """
        Version 1: Ledoit-Wolf Shrinkage MVP.
        Best for high-dimensional data where number of stocks > observations.
        """
        lw = LedoitWolf()
        cov_matrix = lw.fit(self.df_change).covariance_
        
        n = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n)]
        
        res = minimize(self._portfolio_variance, np.full(n, 1/n), args=(cov_matrix,), 
                       method='SLSQP', bounds=bounds, constraints=constraints)
        
        allocation_df, actual_cap = self._get_allocation_details(self.tickers, res.x, capital)
        return {"method": "Ledoit-Wolf", "allocation": allocation_df, "capital_used": actual_cap}

    def optimize_greedy_unrestricted(self, capital=5000, num_stocks=5):
        """
        Version 2: Greedy Optimization with no weight restrictions.
        Concentrates capital into the most variance-reducing assets.
        """
        return self._run_greedy_logic(capital, num_stocks, max_weight=1.0, method_name="Greedy Unrestricted")

    def optimize_greedy_restricted(self, capital=5000, num_stocks=5):
        """
        Version 3: Greedy Optimization with 25% weight restriction per asset.
        Forces diversification even in a greedy selection process.
        """
        return self._run_greedy_logic(capital, num_stocks, max_weight=0.25, method_name="Greedy Restricted (25% Max)")

    def _run_greedy_logic(self, capital, num_stocks, max_weight, method_name):
        """
        Internal execution logic for greedy selection.
        """
        selected_tickers = []
        remaining_tickers = self.tickers.copy()
        final_weights = []
        
        for _ in range(min(num_stocks, len(self.tickers))):
            best_ticker = None
            min_var = float('inf')
            current_best_weights = []
            
            for ticker in remaining_tickers:
                trial_tickers = selected_tickers + [ticker]
                subset_cov = self.df_change[trial_tickers].cov().values
                
                n_subset = len(trial_tickers)
                bounds = [(0, max_weight) for _ in range(n_subset)]
                constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                
                res = minimize(self._portfolio_variance, np.full(n_subset, 1/n_subset), 
                               args=(subset_cov,), method='SLSQP', bounds=bounds, constraints=constraints)
                
                if res.fun < min_var:
                    min_var = res.fun
                    best_ticker = ticker
                    current_best_weights = res.x
            
            if best_ticker:
                selected_tickers.append(best_ticker)
                remaining_tickers.remove(best_ticker)
                final_weights = current_best_weights

        allocation_df, actual_cap = self._get_allocation_details(selected_tickers, final_weights, capital)
        return {"method": method_name, "allocation": allocation_df, "capital_used": actual_cap}