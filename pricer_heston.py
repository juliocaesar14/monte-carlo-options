"""
Heston (1993) stochastic volatility model, Monte Carlo implementation.

Model dynamics under risk-neutral measure:
  dS = r * S * dt + sqrt(v) * S * dW1
  dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
  dW1 * dW2 = rho * dt

Parameters:
  kappa  - mean-reversion speed of variance
  theta  - long-run variance (theta = long_vol^2)
  xi     - vol of vol
  rho    - correlation between stock and vol shocks (negative => put skew)
  V0     - initial variance

Calibration approach:
  V0    = realised var from 1-month SPY daily returns (recent regime)
  theta = realised var from 1-year SPY returns (long-run)
  kappa, xi, rho are set to typical SPY-fitted values from literature.
  Full market calibration (minimising IV RMSE across strike grid) is the
  natural next step — see compare.py for the overlay that motivates it.

Exports:
  simulate_heston_paths(S0, r, V0, kappa, theta, xi, rho, T, N, steps) -> ndarray
  price_european_heston(S0, K, r, V0, kappa, theta, xi, rho, T, N, steps, option_type) -> float
  get_heston_params_from_spy() -> dict
"""

import numpy as np
import yfinance as yf


# heston paramters from spy history

def get_heston_params_from_spy() -> dict:
    """
    Pull SPY history and set Heston parameters:
      V0    — 1-month realised variance (current vol regime)
      theta — 1-year realised variance (long-run target)
      kappa, xi, rho — literature values for S&P 500 index options

    Returns a dict with keys: S0, r, V0, kappa, theta, xi, rho
    """
    spy  = yf.Ticker("SPY")
    hist = spy.history(period="1y")
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

    # 1-year realised variance -> theta 
    annual_sigma_1y = float(log_ret.std() * np.sqrt(252))
    theta = annual_sigma_1y ** 2

    # 1-month realised variance -> V0 
    log_ret_1m = log_ret.iloc[-21:]
    annual_sigma_1m = float(log_ret_1m.std() * np.sqrt(252))
    V0 = annual_sigma_1m ** 2

    S0 = float(spy.fast_info["last_price"])

    irx = yf.Ticker("^IRX")
    tbill = irx.fast_info.get("last_price", None)
    r = float(tbill) / 100.0 if tbill else 0.045

    # Standard SPY-calibrated Heston parameters from empirical literature
    # kappa=2.0: variance mean-reverts with ~6-month half-life
    # xi=0.30:   vol-of-vol typical for equity index
    # rho=-0.70: strong negative correlation -> puts are expensive (skew)
    params = {
        "S0":    S0,
        "r":     r,
        "V0":    V0,
        "kappa": 2.0,
        "theta": theta,
        "xi":    0.30,
        "rho":   -0.70,
    }

    print(f"[Heston] SPY spot: {S0:.2f}  |  V0 (1m): {V0:.4f} ({np.sqrt(V0)*100:.1f}% vol)"
          f"  |  theta (1y): {theta:.4f} ({np.sqrt(theta)*100:.1f}% vol)  |  r: {r:.4f}")
    return params


# simulate path

def simulate_heston_paths(
    S0: float,
    r: float,
    V0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    N: int = 50_000,
    steps: int = 252,
) -> np.ndarray:
    """
    Euler-Maruyama discretisation of Heston SDE.
    Uses full truncation scheme: max(v, 0) before taking sqrt.
    Returns array of shape (N,) — terminal stock prices only.

    Correlated Brownian increments:
      dW1 = Z1
      dW2 = rho*Z1 + sqrt(1-rho^2)*Z2
    where Z1, Z2 ~ N(0,1) independent.
    """

    dt   = T / steps
    S    = np.full(N, S0, dtype=np.float64)
    v    = np.full(N, V0, dtype=np.float64)
    sqrt_dt = np.sqrt(dt)

    for _ in range(steps):
        Z1 = np.random.standard_normal(N)
        Z2 = np.random.standard_normal(N)
        dW1 = Z1
        dW2 = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2

        v_pos = np.maximum(v, 0.0)               # full truncation
        sqrt_v = np.sqrt(v_pos)

        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * dW2
        S = S * np.exp((r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * dW1)

    return S


# pricer

def price_european_heston(
    S0: float,
    K: float,
    r: float,
    V0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    N: int = 50_000,
    steps: int = 252,
    option_type: str = "call",
) -> float:
    """
    European option price under Heston model via Monte Carlo.
    """
    S_T = simulate_heston_paths(S0, r, V0, kappa, theta, xi, rho, T, N, steps)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    return float(np.exp(-r * T) * payoffs.mean())



if __name__ == "__main__":
    p = get_heston_params_from_spy()
    S0, r = p["S0"], p["r"]
    T  = 0.25   # 3-month
    K  = S0     # ATM

    call = price_european_heston(option_type="call", K=K, T=T, **p)
    put  = price_european_heston(option_type="put",  K=K, T=T, **p)

    print(f"\nHeston ATM (3M)  |  Call: {call:.4f}  |  Put: {put:.4f}")


