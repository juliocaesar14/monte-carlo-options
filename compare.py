"""
GBM vs Heston vs Real SPY Market- implied volatility smile comparison.

What this does:
  1. Pulls live SPY spot, rate, and calibrated params for both models.
  2. Prices a grid of European puts across strikes (80%-120% moneyness)
     under GBM and Heston for a 3-month expiry.
  3. Extracts implied vol from each model's prices via BS bisection.
  4. Pulls real SPY put IV from the nearest 3M options chain via yfinance.
  5. Plots all three on one chart — the vol smile comparison.
  6. Prints a clean comparison table: strike | GBM IV | Heston IV | Market IV

Key insight visible in the chart:
  GBM   -> flat line  (constant vol assumption)
  Heston -> curved / skewed downward for low strikes (rho < 0 generates skew)
  Market -> also skewed, puts with K < S0 are expensive 
  Heston tracks market shape; GBM cannot by construction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
from datetime import datetime, timedelta

from pricer_gbm   import get_spy_market_params, price_european_gbm, iv_bisection, bs_price
from pricer_heston import get_heston_params_from_spy, price_european_heston


# configuration

N_GBM    = 200_000    # paths for GBM (fast, antithetic)
N_HESTON =  50_000    # paths for Heston (slower due to path simulation)
N_STRIKES = 15        # number of strikes in grid
TARGET_T  = 0.25      # target expiry in years (~3 months)


# helper function

def get_spy_chain_iv(S0: float, r: float, T_target: float) -> pd.DataFrame | None:
    """
    Pull the SPY options chain for the expiry closest to T_target.
    Returns DataFrame with columns: strike, market_iv, moneyness
    Filters to near-the-money strikes (80%-120%) and volume > 10.
    """
    spy  = yf.Ticker("SPY")
    exps = spy.options                                  

    # Find expiry whose T is closest to T_target
    today = datetime.today()
    best_exp, best_dt = None, float("inf")
    for exp_str in exps:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
        T_exp  = (exp_dt - today).days / 365.0
        if T_exp < 7 / 365:                       
            continue
        if abs(T_exp - T_target) < best_dt:
            best_dt  = abs(T_exp - T_target)
            best_exp = exp_str
            T_actual = T_exp

    if best_exp is None:
        print("[WARNING] No valid SPY expiry found.")
        return None

    print(f"[Market] Using SPY expiry: {best_exp}  (T = {T_actual:.3f} yr)")
    chain = spy.option_chain(best_exp).puts

    # Clean: drop low volume, zero bid, NaN last price
    chain = chain.copy()
    chain["volume"] = chain["volume"].fillna(0)
    chain = chain[chain["bid"] > 0]
    chain = chain.dropna(subset=["lastPrice", "impliedVolatility"])

    # Filter to near-the-money
    chain = chain[(chain["strike"] >= 0.80 * S0) & (chain["strike"] <= 1.20 * S0)]
    chain["moneyness"] = chain["strike"] / S0

    print(f"[Market] {len(chain)} put contracts after filtering.")

    if chain.empty:
        print("[WARNING] No market puts survived filtering — market IV will be omitted.")
        return chain 


    # Re-derive IV from lastPrice using our bisection
    derived_ivs = []
    for _, row in chain.iterrows():
        iv = iv_bisection(row["lastPrice"], S0, row["strike"], r, T_actual, option_type="put")
        derived_ivs.append(iv)
    chain["market_iv"] = derived_ivs
    chain = chain.dropna(subset=["market_iv"])

    return chain[["strike", "moneyness", "market_iv", "lastPrice", "impliedVolatility"]].reset_index(drop=True)


# comparisons

def run_comparison():
    print("=" * 65)
    print("  GBM vs Heston vs SPY Market — Vol Smile Comparison")
    print("=" * 65)

    # parameters
    S0_gbm, sigma_gbm, r_gbm = get_spy_market_params()
    h = get_heston_params_from_spy()
    S0, r = h["S0"], h["r"]

    # Use common S0 and r (both pulled from same SPY source)
    assert abs(S0_gbm - S0) < 1.0, "S0 mismatch between pricers — check yfinance"

    # Strike grid: 80% to 120% of spot
    moneyness_grid = np.linspace(0.80, 1.20, N_STRIKES)
    strikes        = moneyness_grid * S0
    T              = TARGET_T

    print(f"\nPricing {N_STRIKES} strikes from {strikes[0]:.1f} to {strikes[-1]:.1f}")
    print(f"S0 = {S0:.2f}  |  T = {T:.2f} yr  |  r = {r:.4f}\n")

    # price puts under gbm
    print("Pricing under GBM ...")
    gbm_prices, gbm_ivs = [], []
    for K in strikes:
        p  = price_european_gbm(S0, K, r, sigma_gbm, T, N=N_GBM, option_type="put")
        iv = iv_bisection(p, S0, K, r, T, option_type="put")
        gbm_prices.append(p)
        gbm_ivs.append(iv)

    # price puts under heston
    print("Pricing under Heston ...")
    heston_prices, heston_ivs = [], []
    for K in strikes:
        p  = price_european_heston(
            S0=S0, K=K, r=r, T=T,
            V0=h["V0"], kappa=h["kappa"], theta=h["theta"], xi=h["xi"], rho=h["rho"],
            N=N_HESTON, steps=252, option_type="put",
        )
        iv = iv_bisection(p, S0, K, r, T, option_type="put")
        heston_prices.append(p)
        heston_ivs.append(iv)

    # real market IV
    print("Pulling SPY options chain ...")
    market_df = get_spy_chain_iv(S0, r, TARGET_T)

    has_market = market_df is not None and not market_df.empty

    # print comparison table
    print("\n" + "=" * 65)
    print(f"  {'Moneyness':>10} {'Strike':>8} {'GBM IV':>9} {'Heston IV':>11} {'Market IV':>11}")
    print("  " + "-" * 55)
    for i, (K, mn, giv, hiv) in enumerate(zip(strikes, moneyness_grid, gbm_ivs, heston_ivs)):
        mkt_str = "   --"
        if market_df is not None:
            row = market_df.iloc[(market_df["moneyness"] - mn).abs().argsort()].iloc[0]
            mkt_str = f"{row['market_iv']*100:>8.2f}%"
        g_str = f"{giv*100:.2f}%" if giv else "  N/A"
        h_str = f"{hiv*100:.2f}%" if hiv else "  N/A"
        print(f"  {mn:>10.2f} {K:>8.1f} {g_str:>9} {h_str:>11} {mkt_str:>11}")
    print("=" * 65)

    # plot vol smile
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SPY 3-Month Put Implied Volatility — GBM vs Heston vs Market",
                 fontsize=13, fontweight="bold")

    # Left: vol smile
    ax = axes[0]
    valid_g = [(mn, iv) for mn, iv in zip(moneyness_grid, gbm_ivs)    if iv is not None]
    valid_h = [(mn, iv) for mn, iv in zip(moneyness_grid, heston_ivs) if iv is not None]
    ax.plot([x[0] for x in valid_g], [x[1]*100 for x in valid_g],
            "b--", lw=2, label=f"GBM  (σ={sigma_gbm:.2%})")
    ax.plot([x[0] for x in valid_h], [x[1]*100 for x in valid_h],
            "g-",  lw=2.5, label="Heston  (ρ=−0.70)")
    if market_df is not None:
        ax.scatter(market_df["moneyness"], market_df["market_iv"] * 100,
                   color="red", s=50, zorder=5, label="SPY Market (puts)")
    ax.axvline(1.0, color="grey", lw=1, linestyle=":")
    ax.set_xlabel("Moneyness  (K / S₀)", fontsize=11)
    ax.set_ylabel("Implied Volatility (%)", fontsize=11)
    ax.set_title("Vol Smile — 3-Month Expiry")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(alpha=0.3)

    # Annotate put skew
    ax.annotate(
        "Put skew: OTM puts\ncostly (crash risk)",
        xy=(0.88, ax.get_ylim()[1] * 0.85),
        fontsize=8, color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # Right: price comparison (model vs model)
    ax2 = axes[1]
    ax2.plot(moneyness_grid, gbm_prices,    "b--", lw=2,   label="GBM price")
    ax2.plot(moneyness_grid, heston_prices, "g-",  lw=2.5, label="Heston price")
    if market_df is not None:
        ax2.scatter(market_df["moneyness"], market_df["lastPrice"],
                    color="red", s=50, zorder=5, label="SPY market last price")
    ax2.axvline(1.0, color="grey", lw=1, linestyle=":")
    ax2.set_xlabel("Moneyness  (K / S₀)", fontsize=11)
    ax2.set_ylabel("Put Price  ($)", fontsize=11)
    ax2.set_title("Put Prices — GBM vs Heston vs Market")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("vol_smile_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved: vol_smile_comparison.png")
    plt.show()


if __name__ == "__main__":
    run_comparison()


