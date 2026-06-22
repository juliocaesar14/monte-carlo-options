import numpy as np
from scipy.stats import norm

# Parameters
S0    = 100    # spot price
K     = 105    # strike price
r     = 0.05   # risk-free rate
sigma = 0.20   # volatility
T     = 1.0    # time to expiry (years)
N     = 100_000  # number of simulations

# Monte Carlo pricing
np.random.seed(42)
Z   = np.random.standard_normal(N)
S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

call_payoffs = np.maximum(S_T - K, 0)
put_payoffs  = np.maximum(K - S_T, 0)

mc_call = np.exp(-r * T) * call_payoffs.mean()
mc_put  = np.exp(-r * T) * put_payoffs.mean()


d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
bs_put  = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

print(f"MC  Call: {mc_call:.4f}  |  BS Call: {bs_call:.4f}")
print(f"MC  Put:  {mc_put:.4f}  |  BS Put:  {bs_put:.4f}")

for name, mc, bs in [("Call", mc_call, bs_call), ("Put", mc_put, bs_put)]:
    err = abs(mc - bs) / bs * 100
    se  = np.exp(-r * T) * (call_payoffs if name=="Call" else put_payoffs).std() / np.sqrt(N)
    print(f"  {name}: {err:.3f}% error, SE=±{se:.4f}  {'PASS' if err < 0.5 else 'FAIL'}")
