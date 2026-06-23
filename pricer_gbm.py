
import numpy as np
import yfinance as yf
from scipy.stats import norm
 
 
# market data
 
def get_spy_market_params() -> tuple[float, float, float]:
    """
    Pull live SPY spot, 1-year realised vol, and 3-month T-bill rate.
    Returns (S0, sigma, r).
    """
    spy = yf.Ticker("SPY")
    hist = spy.history(period="1y")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    sigma = float(log_returns.std() * np.sqrt(252))
 
    S0 = float(spy.fast_info["last_price"])
 
    irx = yf.Ticker("^IRX")
    tbill = irx.fast_info.get("last_price", None)
    r = float(tbill) / 100.0 if tbill else 0.045   # fallback 4.5%
 
    print(f"[GBM] SPY spot: {S0:.2f}  |  realised vol: {sigma:.4f}  |  r: {r:.4f}")
    return S0, sigma, r
 
 
# core gbm pricer
 
def price_european_gbm(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 100_000,
    option_type: str = "call",
) -> float:
    """
    Monte Carlo European option price under GBM (risk-neutral measure).
    Uses antithetic variates for variance reduction.
    """
    Z = np.random.standard_normal(N // 2)
    Z = np.concatenate([Z, -Z])                          # antithetic pairs
 
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
 
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
 
    return float(np.exp(-r * T) * payoffs.mean())
 
 
# closed form black scholes
 
def bs_price(S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
 
 
# bisection gave implied volatility
 
def iv_bisection(
    market_price: float,
    S0: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float | None:
    """
    Recover implied vol from a market price using bisection on BS formula.
    Returns None if no solution found.
    """
    intrinsic = max(S0 - K, 0) if option_type == "call" else max(K - S0, 0)
    if market_price <= intrinsic + 1e-8:
        return None
 
    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        diff = bs_price(S0, K, r, mid, T, option_type) - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0
 

 
if __name__ == "__main__":
    S0, sigma, r = get_spy_market_params()
    T = 0.25        # 3-month option
    K = S0          # ATM
 
    mc_call  = price_european_gbm(S0, K, r, sigma, T, N=200_000, option_type="call")
    bs_call  = bs_price(S0, K, r, sigma, T, option_type="call")
    mc_put   = price_european_gbm(S0, K, r, sigma, T, N=200_000, option_type="put")
    bs_put   = bs_price(S0, K, r, sigma, T, option_type="put")
 
    print(f"\n{'':>10} {'MC':>10} {'BS':>10} {'Error %':>10}")
    print(f"{'Call':>10} {mc_call:>10.4f} {bs_call:>10.4f} {abs(mc_call-bs_call)/bs_call*100:>10.3f}%")
    print(f"{'Put':>10}  {mc_put:>10.4f}  {bs_put:>10.4f} {abs(mc_put-bs_put)/bs_put*100:>10.3f}%")


