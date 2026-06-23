
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

S0 = 476.0   # illustrative SPY-like spot, NOT a live quote
r  = 0.045   # illustrative risk-free rate, NOT a live quote
 
def bs_price(S0, K, r, sigma, T, opt="call"):
    if T <= 0 or sigma <= 0:
        return max(S0-K,0) if opt=="call" else max(K-S0,0)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt == "call":
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
 
def iv_bisection(mkt, S0, K, r, T, opt="call", tol=1e-6):
    intrinsic = max(S0-K,0) if opt=="call" else max(K-S0,0)
    if mkt <= intrinsic or mkt <= 0 or T <= 0: return np.nan
    lo, hi = 1e-4, 5.0
    flo = bs_price(S0,K,r,lo,T,opt) - mkt
    fhi = bs_price(S0,K,r,hi,T,opt) - mkt
    if flo*fhi > 0: return np.nan
    for _ in range(200):
        mid = (lo+hi)/2
        fm  = bs_price(S0,K,r,mid,T,opt) - mkt
        if abs(fm) < tol: return mid
        if flo*fm < 0: hi = mid; fhi = fm
        else:           lo = mid; flo = fm
    return (lo+hi)/2
 
def spy_vol_smile(moneyness, T):
    """Realistic SPY smile: ATM ~18%, steep put skew, mild call skew, term structure."""
    atm  = 0.18 - 0.02*np.sqrt(T)           # ATM vol decreases with T
    skew = -0.12 + 0.04*np.sqrt(T)           # negative skew, flattens with T
    curv =  0.08 - 0.02*T                    # smile curvature
    x = moneyness - 1.0
    return atm + skew*x + curv*x**2
 
# Grid
expiries = [1/12, 3/12, 6/12, 12/12]   # 1M, 3M, 6M, 12M
exp_labels = ["1M","3M","6M","12M"]
moneyness_grid = np.linspace(0.80, 1.20, 25)
strikes = moneyness_grid * S0
 
# Build IV grids
IV_surface = np.zeros((len(expiries), len(moneyness_grid)))
for i, T in enumerate(expiries):
    for j, m in enumerate(moneyness_grid):
        IV_surface[i, j] = spy_vol_smile(m, T)
 
# Round-trip: price -> bisection -> recover IV
IV_recovered = np.zeros_like(IV_surface)
for i, T in enumerate(expiries):
    for j, (K, iv_true) in enumerate(zip(strikes, IV_surface[i])):
        opt = "put" if K < S0 else "call"
        mkt = bs_price(S0, K, r, iv_true, T, opt)
        iv_rec = iv_bisection(mkt, S0, K, r, T, opt)
        IV_recovered[i, j] = iv_rec if not np.isnan(iv_rec) else iv_true
 
# bisection vs market plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
 
for i, (T, label, ax, c) in enumerate(zip(expiries, exp_labels, axes, colors)):
    ax.plot(moneyness_grid, IV_surface[i]*100,   "o--", color=c, ms=4, lw=1.5, label="Synthetic 'true' smile (input)")
    ax.plot(moneyness_grid, IV_recovered[i]*100, "s-",  color=c, ms=3, lw=2,   alpha=0.7, label="Bisection-recovered IV (output)")
    ax.axvline(1.0, color="grey", ls="--", alpha=0.5, label="ATM")
    ax.fill_between(moneyness_grid[moneyness_grid<1],
                    (IV_surface[i]*100)[moneyness_grid<1],
                    (IV_recovered[i]*100)[moneyness_grid<1],
                    alpha=0.08, color=c)
    ax.set_title(f"Expiry {label} (T={T:.2f}yr)", fontsize=10)
    ax.set_xlabel("Moneyness K/S0", fontsize=9)
    ax.set_ylabel("Implied Volatility (%)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(10, 34)
    # Annotate put skew
    ax.annotate("Put skew:\ncrash-risk\npremium",
                xy=(0.85, IV_surface[i][np.argmin(np.abs(moneyness_grid-0.85))]*100),
                xytext=(0.82, 28), fontsize=6.5, color=c,
                arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
 
fig.suptitle("Synthetic Vol Smile: Bisection Round-Trip Recovery Test\n(Hand-specified smile function — NOT real market data, see day7_market_data.py for that)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("synthetic_iv_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] synthetic_iv_overlay.png (synthetic data, see day7_market_data.py for real SPY data)")
 
# 3D surface plot
M_grid, T_grid = np.meshgrid(moneyness_grid, expiries)
fig = plt.figure(figsize=(11, 7))
ax3d = fig.add_subplot(111, projection="3d")
surf = ax3d.plot_surface(M_grid, T_grid, IV_recovered*100,
                          cmap="viridis", alpha=0.88, edgecolor="none")
fig.colorbar(surf, ax=ax3d, shrink=0.5, label="IV (%)")
 
# Highlight the put skew ridge
for i, T in enumerate(expiries):
    mask = moneyness_grid <= 1.0
    ax3d.plot(moneyness_grid[mask], np.full(mask.sum(), T),
              IV_recovered[i][mask]*100, "r-", lw=1.5, alpha=0.6)
 
ax3d.set_xlabel("Moneyness K/S0", labelpad=8, fontsize=10)
ax3d.set_ylabel("Time to Expiry (yr)", labelpad=8, fontsize=10)
ax3d.set_zlabel("Implied Vol (%)", labelpad=8, fontsize=10)
ax3d.set_title("Synthetic Vol Surface — Put Skew by Construction\n"
               "(Hand-specified smile function, NOT real market data)",
               fontsize=11, fontweight="bold")
ax3d.view_init(elev=25, azim=-50)
plt.tight_layout()
plt.savefig("synthetic_vol_surface.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] synthetic_vol_surface.png (synthetic data, see day7_market_data.py for real SPY data)")