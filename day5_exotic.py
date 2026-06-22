import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

S0 = 100.0       # spot price
K = 100.0        # strike
r = 0.05         # risk-free rate
sigma = 0.20     # volatility
T = 1.0          # time to maturity (years)
steps = 252      # daily monitoring over 1 trading year
N = 100_000      # number of simulated paths

B_up = 120.0     # upper barrier
B_down = 80.0    # lower barrier

rng = np.random.default_rng(42)


def simulate_paths(S0, r, sigma, T, steps, N, rng):
    dt = T / steps
    Z = rng.standard_normal((N, steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(log_paths)
    paths = np.hstack([np.full((N, 1), S0), paths])  # prepend S0 at t=0
    return paths  # shape (N, steps+1)

paths = simulate_paths(S0, r, sigma, T, steps, N, rng)
S_T = paths[:, -1]


# asian call
avg_price = np.mean(paths[:, 1:], axis=1)
asian_call_payoffs = np.maximum(avg_price - K, 0)
asian_call_price = np.exp(-r * T) * np.mean(asian_call_payoffs)
asian_call_se = np.exp(-r * T) * np.std(asian_call_payoffs) / np.sqrt(N)
#up and out call
knocked_out = np.any(paths >= B_up, axis=1)
up_out_payoffs = np.where(knocked_out, 0.0, np.maximum(S_T - K, 0))
up_out_call_price = np.exp(-r * T) * np.mean(up_out_payoffs)
up_out_call_se = np.exp(-r * T) * np.std(up_out_payoffs) / np.sqrt(N)
#down nd in call
activated = np.any(paths <= B_down, axis=1)
down_in_payoffs = np.where(activated, np.maximum(K - S_T, 0), 0.0)
down_in_put_price = np.exp(-r * T) * np.mean(down_in_payoffs)
down_in_put_se = np.exp(-r * T) * np.std(down_in_payoffs) / np.sqrt(N)



#european benchmark
euro_call_payoffs = np.maximum(S_T - K, 0)
euro_call_mc_price = np.exp(-r * T) * np.mean(euro_call_payoffs)
euro_call_mc_se = np.exp(-r * T) * np.std(euro_call_payoffs) / np.sqrt(N)

d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
euro_call_bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


print("=" * 64)
print(f"{'Option':<28}{'Price':>12}{'Std Error':>14}")
print("=" * 64)
print(f"{'European call (MC)':<28}{euro_call_mc_price:>12.4f}{euro_call_mc_se:>14.4f}")
print(f"{'European call (BS, closed-form)':<28}{euro_call_bs_price:>12.4f}{'—':>14}")
print(f"{'Asian call (avg price)':<28}{asian_call_price:>12.4f}{asian_call_se:>14.4f}")
print(f"{'Up-and-out barrier call':<28}{up_out_call_price:>12.4f}{up_out_call_se:>14.4f}")
print(f"{'Down-and-in barrier put':<28}{down_in_put_price:>12.4f}{down_in_put_se:>14.4f}")
print("=" * 64)
print(f"\nAsian call ({asian_call_price:.4f}) < European call ({euro_call_mc_price:.4f}): "
      f"{asian_call_price < euro_call_mc_price}")
print("Reason: averaging the path reduces effective volatility versus using")
print("only the terminal price, and option value is increasing in volatility.")


N_PLOT = 20
t_grid = np.linspace(0, T, steps + 1)
fig, ax = plt.subplots(figsize=(10, 6))

vanilla_label_used = knock_out_label_used = activate_label_used = False
for i in range(N_PLOT):
    path = paths[i]
    if knocked_out[i]:
        color, label, used = "red", "knocked out (up-and-out)", knock_out_label_used
    elif activated[i]:
        color, label, used = "green", "activated (down-and-in)", activate_label_used
    else:
        color, label, used = "grey", "vanilla (neither barrier touched)", vanilla_label_used

    ax.plot(t_grid, path, color=color, alpha=0.7, label=None if used else label)
    if color == "red": knock_out_label_used = True
    elif color == "green": activate_label_used = True
    else: vanilla_label_used = True

ax.axhline(B_up, color="red", linestyle="--", linewidth=1, alpha=0.6, label=f"upper barrier ({B_up})")
ax.axhline(B_down, color="green", linestyle="--", linewidth=1, alpha=0.6, label=f"lower barrier ({B_down})")
ax.axhline(K, color="black", linestyle=":", linewidth=1, alpha=0.5, label=f"strike ({K})")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Price")
ax.set_title(f"{N_PLOT} Sample GBM Paths — Barrier Behaviour")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("exotic_paths.png", dpi=150)
print("\nSaved plot: exotic_paths.png")



