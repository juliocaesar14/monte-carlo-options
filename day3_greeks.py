import numpy as np
from scipy.stats import norm

# Parameters
S0    = 100.0
K     = 100.0
r     = 0.05
sigma = 0.20
T     = 1.0
N     = 200_000

np.random.seed(42)
Z = np.random.standard_normal(N) 
#reuse for all bumps

def mc_price(S0, K, r, sigma, T, Z, option_type="call"):
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    return np.exp(-r * T) * np.mean(payoffs)

print(mc_price(S0, K, r, sigma, T, Z, "call"))


# Delta lower S0 up and down by 1%
h_S = 0.01 * S0

P_up = mc_price(S0 + h_S, K, r, sigma, T, Z, "call")
P_dn = mc_price(S0 - h_S, K, r, sigma, T, Z, "call")

delta = (P_up - P_dn) / (2 * h_S)
print(f"Delta: {delta:.6f}")

# Gamma,is second derivative, how fast Delta changes
P0 = mc_price(S0, K, r, sigma, T, Z, "call")

gamma = (P_up - 2 * P0 + P_dn) / (h_S ** 2)
print(f"Gamma: {gamma:.6f}")

# Vega- we do bumping sigma by 0.001
h_sigma = 0.001

P_vup = mc_price(S0, K, r, sigma + h_sigma, T, Z, "call")
P_vdn = mc_price(S0, K, r, sigma - h_sigma, T, Z, "call")

vega = (P_vup - P_vdn) / (2 * h_sigma)
print(f"Vega:  {vega:.6f}")


# Theta- we perform bumping T by one trading day
h_T = 1 / 252

P_tup = mc_price(S0, K, r, sigma, T + h_T, Z, "call")
P_tdn = mc_price(S0, K, r, sigma, T - h_T, Z, "call")

theta = (P_tdn - P_tup) / (2 * h_T)
print(f"Theta: {theta:.6f}")

# Rho- we do bumping r by 1 basis point
h_r = 0.0001

P_rup = mc_price(S0, K, r + h_r, sigma, T, Z, "call")
P_rdn = mc_price(S0, K, r - h_r, sigma, T, Z, "call")

rho = (P_rup - P_rdn) / (2 * h_r)
print(f"Rho:   {rho:.6f}")


# Black-Scholes analytical Greeks to compare
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

bs_delta = norm.cdf(d1)
bs_gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
bs_vega  = S0 * norm.pdf(d1) * np.sqrt(T)
bs_theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
bs_rho   = K * T * np.exp(-r * T) * norm.cdf(d2)

print("\n" + "="*55)
print(f"{'Greek':<10} {'MC':>12} {'BS Exact':>12} {'Error':>10}")
print("="*55)
print(f"{'Delta':<10} {delta:>12.6f} {bs_delta:>12.6f} {abs(delta-bs_delta):>10.6f}")
print(f"{'Gamma':<10} {gamma:>12.6f} {bs_gamma:>12.6f} {abs(gamma-bs_gamma):>10.6f}")
print(f"{'Vega':<10} {vega:>12.6f} {bs_vega:>12.6f} {abs(vega-bs_vega):>10.6f}")
print(f"{'Theta':<10} {theta:>12.6f} {bs_theta:>12.6f} {abs(theta-bs_theta):>10.6f}")
print(f"{'Rho':<10} {rho:>12.6f} {bs_rho:>12.6f} {abs(rho-bs_rho):>10.6f}")
print("="*55)
