#using original data from yfinance and not synthetic data



import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
import warnings
import sys
 
warnings.filterwarnings("ignore")
 

 
def bs_price(S0, K, r, sigma, T, option_type="call"):
    """Black-Scholes closed-form price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S0 - K, 0) if option_type == "call" else max(K - S0, 0)
        return intrinsic
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
 
 
def iv_bisection(market_price, S0, K, r, T, option_type="call", tol=1e-6, max_iter=200):
    """
    Recover implied vol via bisection.
    Bracket: [0.001, 5.0]. Returns NaN on failure or bad edge cases.
    """
    intrinsic = max(S0 - K, 0) if option_type == "call" else max(K - S0, 0)
    if market_price <= intrinsic or market_price <= 0 or T <= 0:
        return np.nan
 
    lo, hi = 1e-4, 5.0
    f_lo = bs_price(S0, K, r, lo, T, option_type) - market_price
    f_hi = bs_price(S0, K, r, hi, T, option_type) - market_price
 
    if f_lo * f_hi > 0:
        return np.nan  # root not bracketed
 
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = bs_price(S0, K, r, mid, T, option_type) - market_price
        if abs(f_mid) < tol or (hi - lo) / 2.0 < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0
 
 

 
print("=" * 60)
print("Day 7 — Real Market Data Calibration (SPY)")
print("=" * 60)
 
ticker = yf.Ticker("SPY")
S0 = ticker.fast_info.get("last_price", None)
if S0 is None or S0 != S0:  # NaN check
    # fallback: use previous close
    hist = ticker.history(period="2d")
    S0 = float(hist["Close"].iloc[-1])
 
print(f"\nSPY spot price: ${S0:.2f}")
 
# Fetch all available expiries and pick ~1M, 3M, 6M
all_expiries = ticker.options
print(f"Available expiries ({len(all_expiries)} total): {all_expiries[:10]} ...")
 
import datetime
today = datetime.date.today()
 
def days_to_expiry(exp_str):
    exp = datetime.date.fromisoformat(exp_str)
    return (exp - today).days
 
# Pick expiries closest to 30, 90, 180 days out
targets = [30, 90, 180]
chosen_expiries = []
for t in targets:
    valid = [(abs(days_to_expiry(e) - t), e) for e in all_expiries if days_to_expiry(e) > 7]
    if valid:
        chosen_expiries.append(sorted(valid)[0][1])
 
# Deduplicate while preserving order
seen = set()
chosen_expiries = [e for e in chosen_expiries if not (e in seen or seen.add(e))]
print(f"\nChosen expiries: {chosen_expiries}")
 
r = 0.045  # approximate risk-free rate (Fed funds)
 
 

 
all_results = {}  # expiry -> cleaned DataFrame with recovered IV
 
for exp in chosen_expiries:
    T = days_to_expiry(exp) / 365.0
    chain = ticker.option_chain(exp)
 
    # Use puts for skew analysis (more liquid OTM puts below S0)
    puts = chain.puts.copy()
    calls = chain.calls.copy()
 
    print(f"\n── Expiry {exp}  (T={T:.3f}yr) ──")
    print(f"   Puts raw rows: {len(puts)}, Calls raw rows: {len(calls)}")
 
    # ── Clean puts ──
    puts = puts.dropna(subset=["lastPrice", "impliedVolatility"])
    puts = puts[puts["volume"] > 10]                          # liquidity filter
    puts = puts[puts["lastPrice"] > 0]
    puts = puts[puts["strike"].between(0.75 * S0, 1.10 * S0)]  # near-the-money
    puts = puts[puts["lastPrice"] > puts["strike"].apply(lambda K: max(K - S0, 0))]  # above intrinsic
 
    # ── Clean calls ──
    calls = calls.dropna(subset=["lastPrice", "impliedVolatility"])
    calls = calls[calls["volume"] > 10]
    calls = calls[calls["lastPrice"] > 0]
    calls = calls[calls["strike"].between(0.90 * S0, 1.25 * S0)]
    calls = calls[calls["lastPrice"] > calls["strike"].apply(lambda K: max(S0 - K, 0))]
 
    dropped_puts = len(chain.puts) - len(puts)
    dropped_calls = len(chain.calls) - len(calls)
    print(f"   Dropped puts: {dropped_puts}, Dropped calls: {dropped_calls}  (illiquid / NaN / far OTM)")
 
    # ── Recover IV via bisection ──
    puts = puts.copy()
    calls = calls.copy()
 
    puts["iv_bisection"] = puts.apply(
        lambda row: iv_bisection(row["lastPrice"], S0, row["strike"], r, T, "put"), axis=1
    )
    calls["iv_bisection"] = calls.apply(
        lambda row: iv_bisection(row["lastPrice"], S0, row["strike"], r, T, "call"), axis=1
    )
 
    puts["moneyness"] = puts["strike"] / S0
    calls["moneyness"] = calls["strike"] / S0
    puts["T"] = T
    calls["T"] = T
    puts["option_type"] = "put"
    calls["option_type"] = "call"
 
    import pandas as pd
    combined = pd.concat([puts, calls], ignore_index=True)
    combined = combined.dropna(subset=["iv_bisection"])
    combined = combined[combined["iv_bisection"] > 0]
 
    print(f"   Rows after IV recovery: {len(combined)}")
    all_results[exp] = combined
 
 

 
import pandas as pd
 
fig, axes = plt.subplots(1, len(chosen_expiries), figsize=(6 * len(chosen_expiries), 5), sharey=False)
if len(chosen_expiries) == 1:
    axes = [axes]
 
fig.suptitle("SPY Implied Vol: Bisection Recovery vs yFinance Reported IV", fontsize=13, fontweight="bold")
 
for ax, exp in zip(axes, chosen_expiries):
    df = all_results[exp]
    T = df["T"].iloc[0]
 
    puts_df = df[df["option_type"] == "put"].sort_values("moneyness")
    calls_df = df[df["option_type"] == "call"].sort_values("moneyness")
 
    if len(puts_df):
        ax.plot(puts_df["moneyness"], puts_df["iv_bisection"], "b-o", ms=4, label="Bisection IV (puts)")
        ax.scatter(puts_df["moneyness"], puts_df["impliedVolatility"], c="cyan", s=30, zorder=5, label="yFinance IV (puts)", marker="x")
 
    if len(calls_df):
        ax.plot(calls_df["moneyness"], calls_df["iv_bisection"], "r-o", ms=4, label="Bisection IV (calls)")
        ax.scatter(calls_df["moneyness"], calls_df["impliedVolatility"], c="orange", s=30, zorder=5, label="yFinance IV (calls)", marker="x")
 
    ax.axvline(1.0, color="grey", linestyle="--", alpha=0.6, label="ATM (K/S0=1)")
    ax.set_title(f"Expiry {exp}\n(T={T:.2f}yr)", fontsize=10)
    ax.set_xlabel("Moneyness K/S0")
    ax.set_ylabel("Implied Volatility")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig("spy_iv_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] spy_iv_overlay.png")
 

 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
 
fig = plt.figure(figsize=(11, 7))
ax3d = fig.add_subplot(111, projection="3d")
 
colors_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
 
for idx, exp in enumerate(chosen_expiries):
    df = all_results[exp]
    df_sorted = df.sort_values("moneyness")
    T_val = df_sorted["T"].iloc[0]
 
    # Use all options, sorted by moneyness
    xs = df_sorted["moneyness"].values
    zs = df_sorted["iv_bisection"].values
    ys = np.full_like(xs, T_val)
 
    ax3d.plot(xs, ys, zs, color=colors_cycle[idx % len(colors_cycle)],
              linewidth=2, label=f"{exp} (T={T_val:.2f}yr)")
    ax3d.scatter(xs, ys, zs, color=colors_cycle[idx % len(colors_cycle)], s=18, zorder=5)
 
# Annotate put skew
ax3d.set_xlabel("Moneyness K/S0", labelpad=8)
ax3d.set_ylabel("Time to Expiry (yr)", labelpad=8)
ax3d.set_zlabel("Implied Volatility", labelpad=8)
ax3d.set_title(
    "SPY Real Implied Vol Surface\n(Put skew: IV rises for K < S0 — crash-risk premium)",
    fontsize=11, fontweight="bold"
)
ax3d.legend(fontsize=8, loc="upper right")
ax3d.view_init(elev=22, azim=-55)
 
plt.tight_layout()
plt.savefig("spy_vol_surface.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] spy_vol_surface.png")
 
 
 
print("\n" + "=" * 60)
print("IV Recovery Summary")
print("=" * 60)
print(f"{'Expiry':<14} {'T (yr)':<8} {'Rows':<6} {'ATM IV (bisect)':<18} {'ATM IV (yFin)'}")
print("-" * 60)
 
for exp in chosen_expiries:
    df = all_results[exp]
    T_val = df["T"].iloc[0]
    # Near-ATM = moneyness closest to 1.0
    atm_idx = (df["moneyness"] - 1.0).abs().idxmin()
    atm_row = df.loc[atm_idx]
    print(f"{exp:<14} {T_val:<8.3f} {len(df):<6} "
          f"{atm_row['iv_bisection']:<18.4f} {atm_row['impliedVolatility']:.4f}")
 
print("\nPut skew annotation:")
print("  IV rises for K < S0 because downside puts carry crash-risk premium.")
print("  Dealers short puts must hedge delta by selling spot → demand amplifies.")
print("  Skew steepens for short expiries: near-term tail risk is more discrete.")
 
print("\nDone! Outputs: spy_iv_overlay.png, spy_vol_surface.png")



