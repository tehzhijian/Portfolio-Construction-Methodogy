import yfinance as yf
import pandas as pd
import numpy as np
from stockstats import wrap
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from opti_port import PortfolioOptimizer

class TechnicalScoreSelector:
    """
    A class to score stocks based on technical indicators and return variance,
    then construct a DYNAMIC score-based portfolio.
    """

    def __init__(self, tickers):
        self.tickers = tickers
        self.data_map = {}
        self.summary_df = pd.DataFrame()
        self.optimizer = PortfolioOptimizer()

    def fetch_data(self, start_date="2021-01-01", end_date="2024-01-01"):
        """
        Fetches historical data for all tickers and calculates indicators in-memory.
        """
        print(f"Fetching data for {len(self.tickers)} tickers...")
        
        for ticker in self.tickers:
            try:
                # 1. Download Data
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                if df.empty:
                    print(f"Warning: No data for {ticker}")
                    continue

                # --- Fix for MultiIndex Columns (Tuple Error) ---
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 2. Reset index & Standardize columns
                df = df.reset_index()
                df.columns = [str(c).lower() for c in df.columns]
                
                # 3. Wrap with StockStats
                df = wrap(df)
                
                # 4. Pre-calculate custom columns
                df['return'] = df['close'].pct_change()
                df['cma'] = df['close'].expanding(min_periods=14).mean()

                self.data_map[ticker] = df

            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        print("Data fetch and indicator calculation complete.")

    def compute_signals(self, lookback_days=90):
        """
        Analyzes the last N days to count buy signals and calculate variance.
        """
        if not self.data_map:
            print("No data available. Please run fetch_data() first.")
            return pd.DataFrame()

        results = []
        print(f"Computing signals (Lookback: {lookback_days} days)...")
        
        for ticker, df in self.data_map.items():
            if df.empty: continue

            # --- Pre-calculate Indicators on the full dataframe ---
            try:
                _ = df['kdjk']; _ = df['rsi']; _ = df['wr']
                _ = df['macdh']; _ = df['boll_lb']
                _ = df['close_14_sma']; _ = df['close_14_ema']
            except KeyError as e:
                continue

            # Slice
            subset = df.iloc[-lookback_days:].copy()
            if subset.empty: continue

            # --- Define Strong Buy Signals ---
            kdj_condition = (subset["kdjk"].shift(1) < 20) & (subset["kdjk"] >= 20)
            
            strong_buy_signals = {
                "RSI": subset["rsi"] < 30,
                "William %R": subset["wr"] < -80,
                "KDJ": kdj_condition,
                "MACD Histogram": subset["macdh"] > 0,
                "Bollinger Bands": subset["close"] < subset["boll_lb"],
                "SMA Crossover": (subset["close"].shift(1) < subset['close_14_sma']) & (subset["close"] >= subset['close_14_sma']),
                "EMA Crossover": (subset["close"].shift(1) < subset['close_14_ema']) & (subset["close"] >= subset['close_14_ema']),
                "CMA Crossover": (subset["close"].shift(1) < subset['cma']) & (subset["close"] >= subset['cma'])
            }

            strong_buy_counts = sum(int(condition.sum()) for condition in strong_buy_signals.values())
            return_variance = float(subset["return"].var())

            results.append({
                "stocks": ticker,
                "buy_count": strong_buy_counts,
                "variance": return_variance
            })

        self.summary_df = pd.DataFrame(results)
        return self.summary_df

    def optimize_score(self, top_n=25):
        """
        Uses Grid Search/Random Forest to score stocks. Returns top N stocks.
        """
        if self.summary_df.empty:
            print("No signals computed. Run compute_signals() first.")
            return pd.DataFrame(), {}

        df_calc = self.summary_df.copy()
        
        # Normalize
        scaler = MinMaxScaler()
        df_calc[["Buy_Signals_Norm", "Variance_Norm"]] = scaler.fit_transform(df_calc[["buy_count", "variance"]])

        def compute_score_vec(alpha, beta, signals, variance):
            return alpha * signals - beta * variance

        # Grid Search logic for Alpha/Beta
        alpha_values = np.linspace(0.5, 2, 50)
        beta_values = np.linspace(0.5, 2, 50)
        
        best_score = -np.inf
        best_params = {"alpha": 1.0, "beta": 1.0}

        for a in alpha_values:
            for b in beta_values:
                current_scores = compute_score_vec(a, b, df_calc["Buy_Signals_Norm"], df_calc["Variance_Norm"])
                if np.mean(current_scores) > best_score:
                    best_score = np.mean(current_scores)
                    best_params = {"alpha": a, "beta": b}

        # Apply Final Scoring
        df_calc["Final_Score"] = compute_score_vec(
            best_params['alpha'], best_params['beta'], 
            df_calc["Buy_Signals_Norm"], df_calc["Variance_Norm"]
        )

        top_stocks = df_calc.sort_values(by="Final_Score", ascending=False).head(top_n)
        return top_stocks, best_params

    def construct_portfolio(self, capital, top_n=25, n_range=(0.01, 0.10), m_range=(0.10, 0.50)):
        """
        Constructs a DYNAMIC portfolio where weights are optimized based on Final Scores.
        """
        # 1. Get Top Stocks
        top_stocks, params = self.optimize_score(top_n)
        if top_stocks.empty:
            return pd.DataFrame(), capital

        print(f"\n--- Constructing Portfolio (Alpha: {params['alpha']:.2f}, Beta: {params['beta']:.2f}) ---")
        
        selected_tickers = top_stocks['stocks'].tolist()
        scores_map = dict(zip(top_stocks['stocks'], top_stocks['Final_Score']))
        
        prices = {}
        scores_list = []
        valid_tickers = []
        
        for t in selected_tickers:
            if t in self.data_map and not self.data_map[t].empty:
                latest_price = self.data_map[t]['close'].iloc[-1]
                prices[t] = latest_price
                scores_list.append(scores_map[t])
                valid_tickers.append(t)

        n_stocks = len(valid_tickers)

        # 2. Define Dynamic Constraints
        # We enforce a small minimum weight (e.g. 1% or 0.5%) to allow the optimizer
        # to push capital to the winners, rather than forcing a naive 1/N.
        
        # Calculate safe floor: (1.0 / N) * 0.2  -> 20% of the average weight
        safe_min = (1.0 / n_stocks) * 0.2
        min_weight = max(0.01, safe_min) # Default to 1% min
        
        # Calculate safe ceiling: 25% or higher
        max_weight = 0.25 
        
        # Adjust if boundaries are physically impossible
        if min_weight * n_stocks > 1.0:
            min_weight = 0.99 / n_stocks
        
        print(f"Dynamic Constraints: Min {min_weight:.2%} | Max {max_weight:.2%}")

        # 3. Optimize for MAX SCORE (Objective: Maximize w * Score)
        # We minimize (- w * Score)
        scores_array = np.array(scores_list)
        initial_weights = np.array([1/n_stocks] * n_stocks)
        
        def objective_function(w):
            return -np.dot(w, scores_array) # Negative because we want to MAXIMIZE score
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(min_weight, max_weight) for _ in range(n_stocks)]
        
        res = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not res.success:
            print(f"Optimization failed: {res.message}")
            # Fallback to naive if optimization breaks
            final_weights = initial_weights
        else:
            final_weights = res.x

        # 4. Build DataFrame
        portfolio_data = []
        total_cost = 0
        
        for i, ticker in enumerate(valid_tickers):
            w = final_weights[i]
            p = prices[ticker]
            
            allocated = w * capital
            shares = int(np.floor(allocated / p))
            cost = shares * p
            total_cost += cost
            
            if shares > 0:
                portfolio_data.append({
                    "Ticker": ticker,
                    "Price": p,
                    "Weight": round(w, 4),
                    "Shares": shares,
                    "Actual Cost": cost,
                    "Score": round(scores_list[i], 2)
                })

        final_df = pd.DataFrame(portfolio_data)
        
        # Sort by Weight High -> Low to show the "Dynamic" allocation clearly
        final_df = final_df.sort_values(by="Weight", ascending=False).reset_index(drop=True)
        
        remaining_cap = capital - total_cost
        print(f"Portfolio Generated. Unused Capital: ${remaining_cap:.2f}")
        return final_df, remaining_cap