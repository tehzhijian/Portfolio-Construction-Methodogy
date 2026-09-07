import pandas as pd
import numpy as np
from Algo_1 import VarianceSelector
from Algo_2 import FundamentalScoreSelector
from Algo_xg import SmartPortfolioBuilder
from Algo_4 import TechnicalScoreSelector

# --- 1. Helper to Bypass Grid Search ---
class NaiveOptimizerStub:
    def test_opti_tune(self, n_stocks, variances, stocks, prices, capital, *args, **kwargs):
        if n_stocks == 0:
            return 0, 0, capital
        weight = 1.0 / n_stocks
        return weight, weight, 0.0 

# --- 2. Naive Algorithm 1 ---
class NaiveVarianceSelector(VarianceSelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = NaiveOptimizerStub()

    def _get_final_allocation(self, n_stocks, variances, stocks, prices, n, m):
        target_weight = n
        allocation_data = []
        for ticker in stocks.keys():
            share_price = prices[ticker]
            allocated_amt = target_weight * self.capital
            num_shares = int(np.floor(allocated_amt / share_price))
            actual_cost = num_shares * share_price
            if num_shares > 0:
                allocation_data.append({
                    "Ticker": ticker,
                    "Price": share_price,
                    "Optimized Weight": round(target_weight, 4),
                    "Shares": num_shares,
                    "Actual Cost": actual_cost,
                    "Strategy": "Naive 1/N"
                })
        return pd.DataFrame(allocation_data)

# --- 3. Naive Algorithm 2 ---
class NaiveFundamentalSelector(FundamentalScoreSelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = NaiveOptimizerStub()

    def _reconstruct_allocation(self, n_stocks, variances, stocks, prices, n, m):
        target_weight = n
        allocation_data = []
        for ticker in stocks.keys():
            share_price = prices[ticker]
            allocated_amt = target_weight * self.capital
            num_shares = int(np.floor(allocated_amt / share_price))
            actual_cost = num_shares * share_price
            if num_shares > 0:
                allocation_data.append({
                    "Ticker": ticker,
                    "Price": share_price,
                    "Optimized Weight": round(target_weight, 4),
                    "Shares": num_shares,
                    "Actual Cost": actual_cost,
                    "Total Score": self.scores.get(ticker, 0),
                    "Strategy": "Naive Fundamental"
                })
        return pd.DataFrame(allocation_data)

# --- 4. Naive Algo XG ---
class NaiveSmartPortfolioBuilder(SmartPortfolioBuilder):
    def __init__(self, tickers, markets, start_date, end_date, risk_free_rate=0.0365):
        super().__init__(tickers, markets, start_date, end_date, risk_free_rate)
        self.optimizer = NaiveOptimizerStub()

# --- 5. Naive Algo 4 (FIXED) ---
class NaiveTechnicalScoreSelector(TechnicalScoreSelector):
    """
    Uses Technical Indicators to SELECT stocks, but allocates Capital EQUALLY (1/N).
    """
    def __init__(self, tickers):
        super().__init__(tickers)
        # We don't need the stub necessarily, but good for consistency
        self.optimizer = NaiveOptimizerStub()

    def construct_portfolio(self, capital, top_n=25, **kwargs):
        # 1. Reuse Parent Selection Logic
        # We still want the "Smart Selection" (Top N based on Technical Score)
        top_stocks, _ = self.optimize_score(top_n)
        
        if top_stocks.empty:
            return pd.DataFrame(), capital

        # 2. Force Naive 1/N Allocation (Overriding Parent Complex Logic)
        selected_tickers = top_stocks['stocks'].tolist()
        n_stocks = len(selected_tickers)
        weight = 1.0 / n_stocks
        
        print(f"\n--- Constructing NAIVE Portfolio (1/{n_stocks} each) ---")
        
        portfolio_data = []
        total_cost = 0
        
        for ticker in selected_tickers:
            if ticker in self.data_map and not self.data_map[ticker].empty:
                price = self.data_map[ticker]['close'].iloc[-1]
                
                # Naive Math
                allocated = weight * capital
                shares = int(np.floor(allocated / price))
                cost = shares * price
                
                if shares > 0:
                    portfolio_data.append({
                        "Ticker": ticker,
                        "Price": price,
                        "Weight": round(weight, 4),
                        "Shares": shares,
                        "Actual Cost": cost,
                        "Score": "N/A (Naive)"
                    })
                    total_cost += cost
        
        final_df = pd.DataFrame(portfolio_data)
        remaining_cap = capital - total_cost
        
        return final_df, remaining_cap