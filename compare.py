import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
from datetime import datetime

from pricer_gbm    import get_spy_market_params, price_european_gbm, iv_bisection, bs_price
from pricer_heston import get_heston_params_from_spy, price_european_heston


N_GBM     = 200_000
N_HESTON  =  50_000
N_STRIKES =  15
TARGET_T  =  0.25

IV_MIN    =  0.08
IV_MAX    =  1.50


def get_spy_chain_iv(S0: float, r: float, T_target: float):
    spy  = yf.Ticker("SPY")
    exps = spy.options

    today = datetime.today()
    best_exp, best_dt, T_actual = None, float("inf"), None
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
        print("No valid SPY expiry found.")
        return None

    print(f"Market using SPY expiry. {best_exp}, T. {T_actual:.3f} yr")
    chain = spy.option_chain(best_exp).puts.copy()

    for col in ["bid", "ask", "lastPrice", "impliedVolatility", "volume"]:
        chain[col] = pd.to_numeric(chain[col], errors="coerce")
    chain["volume"]   = chain["volume"].fillna(0)
    chain["lastPrice"] = chain["lastPrice"].fillna(0)

    chain = chain.dropna(subset=["impliedVolatility"])
    chain = chain[chain["impliedVolatility"] > 0.01]
    chain = chain[(chain["strike"] >= 0.80 * S0) & (chain["strike"] <= 1.02 * S0)]
    chain["moneyness"] = chain["strike"] / S0
    chain["mid_price"] = (chain["bid"] + chain["ask"]) / 2

    # Require a live, reasonably tight two-sided market. Falling back to
    # lastPrice for quotes with no current bid/ask (or a very wide spread)
    # is what produced jagged spikes in the recovered IV curve, since a
    # stale lastPrice can sit far from the option's current fair value.
    # Spread threshold loosened to 60% (from an initial 25%) since SPY
    # near-term/far-OTM strikes can legitimately have wider quoted spreads
    # even when liquid; 25% was dropping every contract.
    n_before = len(chain)
    has_quote   = (chain["bid"] > 0) & (chain["ask"] > 0)
    tight_quote = has_quote & ((chain["ask"] - chain["bid"]) / chain["mid_price"] < 0.60)
    chain = chain[tight_quote].copy()
    print(f"Quote filter: {n_before} contracts -> {len(chain)} with live, reasonably tight two-sided markets.")

    print(f"Strikes surviving moneyness filter. {sorted(chain['strike'].tolist())}")
    print(f"Market, {len(chain)} OTM and near ATM put contracts available.")

    if chain.empty:
        print("No market puts survived filtering, market IV will be omitted.")
        return chain

    derived_ivs   = []
    price_used    = []

    for _, row in chain.iterrows():
        iv = iv_bisection(row["mid_price"], S0, row["strike"], r, T_actual, option_type="put")
        if iv is not None and IV_MIN < iv < IV_MAX:
            derived_ivs.append(iv)
            price_used.append(row["mid_price"])
        else:
            derived_ivs.append(None)
            price_used.append(row["mid_price"])

    chain["market_iv"]   = derived_ivs
    chain["price_used"]  = price_used
    chain = chain.dropna(subset=["market_iv"])
    chain = chain[(chain["market_iv"] > IV_MIN) & (chain["market_iv"] < IV_MAX)]

    print(f"Market IV recovered for {len(chain)} contracts from live mid-price quotes only.")
    print(f"Market IV range after filtering. {chain['market_iv'].min()*100:.1f}% to {chain['market_iv'].max()*100:.1f}%")

    return chain[["strike", "moneyness", "market_iv", "price_used", "impliedVolatility"]].reset_index(drop=True)


def run_comparison():
    print("GBM vs Heston vs SPY Market, Vol Smile Comparison")

    S0_gbm, sigma_gbm, r_gbm = get_spy_market_params()
    h = get_heston_params_from_spy()
    S0, r = h["S0"], h["r"]

    assert abs(S0_gbm - S0) < 1.0, "S0 mismatch between pricers, check yfinance"

    moneyness_grid = np.linspace(0.80, 1.20, N_STRIKES)
    strikes        = moneyness_grid * S0
    T              = TARGET_T

    print(f"Pricing {N_STRIKES} strikes from {strikes[0]:.1f} to {strikes[-1]:.1f}")
    print(f"S0. {S0:.2f}, T. {T:.2f} yr, r. {r:.4f}")

    print("Pricing under GBM.")
    gbm_prices, gbm_ivs = [], []
    for K in strikes:
        p  = price_european_gbm(S0, K, r, sigma_gbm, T, N=N_GBM, option_type="put")
        iv = iv_bisection(p, S0, K, r, T, option_type="put")
        gbm_prices.append(p)
        gbm_ivs.append(iv)

    print("Pricing under Heston.")
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

    print("Pulling SPY options chain.")
    market_df  = get_spy_chain_iv(S0, r, TARGET_T)
    has_market = market_df is not None and not market_df.empty

    print(f"{'Moneyness':>10} {'Strike':>8} {'GBM IV':>9} {'Heston IV':>11} {'Market IV':>11}")
    for K, mn, giv, hiv in zip(strikes, moneyness_grid, gbm_ivs, heston_ivs):
        mkt_str = "N/A"
        if has_market:
            closest = market_df.iloc[((market_df["moneyness"] - mn).abs()).argsort().iloc[0]]
            mkt_str = f"{closest['market_iv'] * 100:>8.2f}%"
        g_str = f"{giv * 100:.2f}%" if giv else "N/A"
        h_str = f"{hiv * 100:.2f}%" if hiv else "N/A"
        print(f"  {mn:>10.2f} {K:>8.1f} {g_str:>9} {h_str:>11} {mkt_str:>11}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SPY 3 Month Put Implied Volatility, GBM vs Heston vs Market",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    valid_g = [(mn, iv) for mn, iv in zip(moneyness_grid, gbm_ivs)    if iv is not None]
    valid_h = [(mn, iv) for mn, iv in zip(moneyness_grid, heston_ivs) if iv is not None]
    ax.plot([x[0] for x in valid_g], [x[1] * 100 for x in valid_g],
            "b--", lw=2, label=f"GBM, sigma. {sigma_gbm:.2%}")
    ax.plot([x[0] for x in valid_h], [x[1] * 100 for x in valid_h],
            "g-",  lw=2.5, label="Heston, rho. 0.70")
    if has_market:
        ax.scatter(market_df["moneyness"], market_df["market_iv"] * 100,
                   color="red", s=60, zorder=5, label="SPY Market, OTM puts")
    ax.axvline(1.0, color="grey", lw=1, linestyle=":")
    ax.set_xlabel("Moneyness, K divided by S0", fontsize=11)
    ax.set_ylabel("Implied Volatility, percent", fontsize=11)
    ax.set_title("Vol Smile, 3 Month Expiry")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.annotate(
        "Put skew, OTM puts costly, crash risk",
        xy=(0.03, 0.92), xycoords="axes fraction",
        fontsize=8, color="red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax2 = axes[1]
    ax2.plot(moneyness_grid, gbm_prices,    "b--", lw=2,   label="GBM price")
    ax2.plot(moneyness_grid, heston_prices, "g-",  lw=2.5, label="Heston price")
    if has_market:
        ax2.scatter(market_df["moneyness"], market_df["price_used"],
                    color="red", s=60, zorder=5, label="SPY market last traded price")
    ax2.axvline(1.0, color="grey", lw=1, linestyle=":")
    ax2.set_xlabel("Moneyness, K divided by S0", fontsize=11)
    ax2.set_ylabel("Put Price in dollars", fontsize=11)
    ax2.set_title("Put Prices, GBM vs Heston vs Market")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("vol_smile_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved. vol smile comparison.png")
    plt.show()

    print("GBM prices OTM puts using the same flat vol as ATM options.")
    print("Hestons negative rho causes variance to spike when prices fall, making OTM puts systematically more expensive.")
    print("This is the volatility skew, visible in real SPY chains and reproduced by Heston but structurally impossible under constant vol GBM.")


if __name__ == "__main__":
    run_comparison()


    