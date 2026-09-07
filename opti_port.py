import numpy as np
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    A class to handle portfolio optimization tasks. 
    It allows for optimizing allocation weights to minimize variance and 
    tuning constraint parameters (n, m) to maximize capital utilization.
    """

    def __init__(self):
        pass

    def opti_tune(self, n_stocks, variances, stocks, prices, capital, n, m):
        """
        Optimizes portfolio weights to minimize variance subject to constraints.
        
        Parameters:
        - n_stocks: Number of stocks in the portfolio.
        - variances: A list or array of variances for the assets.
        - stocks: Dictionary of stock data (keys are tickers).
        - prices: Dictionary of stock prices (keys are tickers).
        - capital: Total capital available for investment.
        - n: Lower bound for asset weight.
        - m: Upper bound for asset weight.

        Returns:
        - remaining_capital: The amount of capital left unallocated after integer share calculation.
        """
        initial_weights = np.array([1/n_stocks] * n_stocks)
        
        # Create covariance matrix (Assumes variances only/diagonal matrix based on original code)
        cov_matrix = np.diag(variances)
        
        # Define the objective function (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Define the constraint: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        
        # Define bounds for weights: non-negative and upper bound for diversification
        bounds = [(n, m) for _ in range(n_stocks)]
        
        # Minimize the portfolio variance
        optimized_result = minimize(
            portfolio_variance, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        # Extract the optimized weights
        optimized_weights = optimized_result.x
        
        # Map weights to stocks
        optimized_portfolio = dict(zip(stocks.keys(), optimized_weights))
        weights = {stock: float(weight) for stock, weight in optimized_portfolio.items()}
        
        # Calculate the capital allocated to each stock based on the weights
        allocated_capital = {stock: weights[stock] * capital for stock in weights}
        
        # Calculate the number of shares for each stock (using integer shares)
        shares = {stock: int(np.floor(allocated_capital[stock] / prices[stock])) for stock in prices}
        
        # Calculate the total capital used
        total_capital_used = sum(shares[stock] * prices[stock] for stock in shares)
        
        # Adjust if necessary to use up the entire capital
        remaining_capital = capital - total_capital_used

        return remaining_capital

    def test_opti_tune(self, n_stocks, variances, stocks, prices, capital, n_range=(0.01, 0.10), m_range=(0.10, 0.50)):
        """
        Iterates through ranges of weight bounds to find the best combination.
        CRITICAL FIX: Ensures the minimum weight 'n' allows for optimization freedom.
        """
        best_n = 0.01 # Default to 1% if nothing better found
        best_m = 0.25 # Default to 25%
        best_remaining_capital = capital 
        
        # Calculate the maximum 'n' that still allows for optimization.
        # We want the sum of minimums (n * stocks) to be AT MOST 75% of the portfolio.
        # This leaves 25% of the capital 'free' to be moved to the best stocks.
        max_valid_n = 0.75 / n_stocks 
        
        # Clip the user's n_range to respect this mathematical limit
        valid_n_end = min(n_range[1], max_valid_n)
        
        # If the range is invalid (e.g., min > max), force a safe small number
        if n_range[0] >= valid_n_end:
             # Fallback: Just search small weights like 0.5% to 2%
             search_range_n = np.arange(0.005, 0.02, 0.005)
        else:
             search_range_n = np.arange(n_range[0], valid_n_end, 0.005)

        # Iterate n (Min Weight)
        for n in search_range_n:
            # Check if this 'n' is mathematically possible (n * stocks <= 1)
            if n * n_stocks > 0.95: 
                continue
                
            # Iterate m (Max Weight)
            for m in np.arange(m_range[0], m_range[1], 0.05):
                # Valid Constraints Check:
                # 1. Min < Max
                # 2. Max is high enough to actually take up the slack (m * 1 + n * (stocks-1) >= 1)
                if n >= m: continue
                
                try:
                    remaining = self.opti_tune(n_stocks, variances, stocks, prices, capital, n, m)
                    
                    # Logic: We want min capital remaining, BUT we prefer smaller 'n' 
                    # if the capital difference is negligible.
                    
                    # If we found a new best capital efficiency
                    if remaining < best_remaining_capital:
                        best_n = n
                        best_m = m
                        best_remaining_capital = remaining
                        
                    # Tie-breaker: If capital is same, prefer the one with LOWER 'n' (More freedom)
                    elif remaining == best_remaining_capital:
                        if n < best_n:
                            best_n = n
                            best_m = m
                            
                except Exception:
                    continue
        
        print(f"  [OptiTune] Selected Bounds -> n: {best_n:.2%} | m: {best_m:.2%} | Unused: ${best_remaining_capital:.2f}")
        return best_n, best_m, best_remaining_capital