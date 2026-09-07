import pandas as pd
import numpy as np
from scipy.optimize import minimize
from opti_port import PortfolioOptimizer

class FundamentalScoreSelector:
    """
    Implements 'Algorithm 2':
    1. Filters stocks by Price (<= 5% of Capital).
    2. Ranks stocks across 15 metrics (Risk, Return, Beta, Fundamentals).
    3. Aggregates scores (Cumulative Scoring).
    4. Selects Top 25 highest scorers.
    5. Optimizes weights using Variance minimization via PortfolioOptimizer.
    """

    def __init__(self, capital=5000, price_threshold_ratio=0.05, top_n=25):
        self.capital = capital
        self.threshold = capital * price_threshold_ratio
        self.top_n = top_n
        self.optimizer = PortfolioOptimizer()
        self.scores = {}  # Dictionary to hold cumulative scores

    def process(self, df_results):
        """
        Ingests dataframe, applies Algo 2 scoring logic, and runs optimization.

        :param df_results: DataFrame containing price, risk metrics, and fundamental ratios.
        :return: Tuple (DataFrame of final allocation, float remaining_capital)
        """
        # --- Step 1: Data Preparation ---
        df_work = df_results.copy()

        # Ensure Ticker is available as a column
        if 'ticker' not in df_work.columns and df_work.index.name == 'ticker':
            df_work.reset_index(inplace=True)
        elif 'ticker' not in df_work.columns and 'Ticker' in df_work.columns:
            df_work.rename(columns={'Ticker': 'ticker'}, inplace=True)

        if 'ticker' in df_work.columns:
            df_work.set_index('ticker', inplace=True)

        df_work.replace('-', 0, inplace=True)
        df_work.fillna(0, inplace=True)

        # --- Step 2: Price Filtering ---
        # -> "t_dic[i][0] <= threshold"
        # Assuming 'latest_price' is the column name for price
        if 'latest_price' not in df_work.columns:
            # Fallback for 'Price' column if 'latest_price' missing
            if 'Price' in df_work.columns:
                df_work.rename(columns={'Price': 'latest_price'}, inplace=True)
            else:
                raise ValueError("Column 'latest_price' (or 'Price') is required for filtering.")

        eligible_df = df_work[df_work['latest_price'] <= self.threshold].copy()

        if eligible_df.empty:
            print(f"No stocks found under price threshold ${self.threshold:.2f}")
            return pd.DataFrame(), self.capital

        # Initialize scores
        self.scores = {ticker: 0 for ticker in eligible_df.index}

        # --- Step 3: Multi-Factor Ranking ---
        # 1. Expected Return (High to Low)
        self._rank_metric(eligible_df, 'expected_return', ascending=False)

        # 2. Daily Variance (Low to High)
        self._rank_metric(eligible_df, 'daily_variance', ascending=True)

        # 3. Monthly Sharpe (High to Low - Absolute Value)
        self._rank_metric(eligible_df, 'monthly_sharpe', ascending=False, use_abs=True)

        # 4. Betas (Low to High - Absolute Value)
        self._rank_metric(eligible_df, 'beta_ixic', ascending=True, use_abs=True)
        self._rank_metric(eligible_df, 'beta_nya', ascending=True, use_abs=True)
        self._rank_metric(eligible_df, 'beta_gspc', ascending=True, use_abs=True)

        # 5. Treynor Ratios (High to Low - Absolute Value)
        self._rank_metric(eligible_df, 'treynor_ixic', ascending=False, use_abs=True)
        self._rank_metric(eligible_df, 'treynor_nya', ascending=False, use_abs=True)
        self._rank_metric(eligible_df, 'treynor_gspc', ascending=False, use_abs=True)

        # 6. Fundamentals (High to Low)
        fund_metrics = ['ROA', 'ROE', 'ROI', 'Gross Margin', 'Operating Margin', 'Profit Margin']
        for metric in fund_metrics:
            col_match = next((c for c in eligible_df.columns if metric in c), None)
            if col_match:
                self._rank_metric(eligible_df, col_match, ascending=False)

        # --- Step 4: Selection ---
        sorted_tickers = sorted(self.scores, key=self.scores.get, reverse=True)
        top_tickers = sorted_tickers[:self.top_n]

        print(f"Top {len(top_tickers)} selected based on Fundamental Score.")

        # Create subset for optimization
        portfolio_subset = eligible_df.loc[top_tickers]

        # --- Step 5: Format for Optimizer ---
        stocks_dict = {}
        for ticker, row in portfolio_subset.iterrows():
            stocks_dict[ticker] = [row['latest_price'], row['expected_return'], row['daily_variance']]

        n_stocks = len(top_tickers)
        variances = np.array([x[2] for x in stocks_dict.values()])
        prices = {k: v[0] for k, v in stocks_dict.items()}

        # --- Step 6: Run Optimization using External Class ---
        # Uses test_opti_tune from opti_port.py to find best N and M
        best_n, best_m, remaining_capital = self.optimizer.test_opti_tune(
            n_stocks, variances, stocks_dict, prices, self.capital
        )

        print(f"Optimal Constraints: Min % {best_n:.2%} | Max % {best_m:.2%}")

        # --- Step 7: Final Allocation Calculation ---
        # Since opti_port.py only returns remaining_capital, we run the logic one last time
        # locally to extract the specific share counts.
        final_allocation_df = self._reconstruct_allocation(
            n_stocks, variances, stocks_dict, prices, best_n, best_m
        )

        # Recalculate remaining capital exactly based on the final DataFrame
        if not final_allocation_df.empty:
            total_spent = final_allocation_df['Actual Cost'].sum()
            final_remaining = self.capital - total_spent
        else:
            final_remaining = self.capital

        return final_allocation_df, final_remaining

    def _rank_metric(self, df, column_name, ascending=True, use_abs=False):
        """
        Internal helper to perform ranking and update scores.
        """
        if column_name not in df.columns:
            return

        if use_abs:
            values = df[column_name].abs()
        else:
            values = df[column_name]

        items = list(values.items())
        sorted_items = sorted(items, key=lambda x: x[1], reverse=not ascending)

        max_score = len(sorted_items)
        current_score = max_score + 1

        for i, (ticker, val) in enumerate(sorted_items):
            if val == 0:
                self.scores[ticker] += 0
            else:
                current_score -= 1
                self.scores[ticker] += current_score

    def _reconstruct_allocation(self, n_stocks, variances, stocks, prices, n, m):
        """
        Replicates the minimization logic from opti_port.py to return full DataFrame details.
        """
        #
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

            if num_shares > 0:
                allocation_data.append({
                    "Ticker": ticker,
                    "Price": share_price,
                    "Optimized Weight": round(weight, 4),
                    "Shares": num_shares,
                    "Actual Cost": actual_cost,
                    "Total Score": self.scores.get(ticker, 0)
                })

        return pd.DataFrame(allocation_data)