
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta
 
end   = datetime.today()
start = end - timedelta(days=5*365)
 
spy = yf.download("SPY", start=start, end=end, progress=False)
vix = yf.download("^VIX", start=start, end=end, progress=False)
 
spy_close = spy["Close"].squeeze()
vix_close = vix["Close"].squeeze()
 
log_returns  = np.log(spy_close / spy_close.shift(1))
realized_vol = log_returns.rolling(window=30).std() * np.sqrt(252) * 100
implied_vol  = vix_close.reindex(realized_vol.index)
 
df = pd.DataFrame({
    "Realized Vol (30d)": realized_vol,
    "Implied Vol (VIX)":  implied_vol,
}).dropna()
 
vrp = df["Implied Vol (VIX)"] - df["Realized Vol (30d)"]
 
print(f"Avg Implied Vol.  {df['Implied Vol (VIX)'].mean():.1f}%")
print(f"Avg Realized Vol. {df['Realized Vol (30d)'].mean():.1f}%")
print(f"Avg VRP.          {vrp.mean():.1f} vol points")
print(f"IV above RV.      {(vrp > 0).mean()*100:.0f}% of trading days")
 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                gridspec_kw={"height_ratios": [3, 1.2]},
                                sharex=True)
fig.patch.set_facecolor("#0f0f1a")
for ax in [ax1, ax2]:
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="#aaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
 
ax1.fill_between(df.index,
                 df["Implied Vol (VIX)"], df["Realized Vol (30d)"],
                 where=df["Implied Vol (VIX)"] >= df["Realized Vol (30d)"],
                 alpha=0.18, color="#4e9af1", label="_nolegend_")
ax1.fill_between(df.index,
                 df["Implied Vol (VIX)"], df["Realized Vol (30d)"],
                 where=df["Implied Vol (VIX)"] < df["Realized Vol (30d)"],
                 alpha=0.35, color="#ff4f4f", label="_nolegend_")
 
ax1.plot(df.index, df["Implied Vol (VIX)"],
         color="#4e9af1", lw=1.6, label="Implied Vol, VIX")
ax1.plot(df.index, df["Realized Vol (30d)"],
         color="#f1c04e", lw=1.4, label="Realized Vol, 30d rolling")
 
crisis_mask = vrp < -5
if crisis_mask.any():
    rv_crisis = df["Realized Vol (30d)"][crisis_mask]
    peak_idx  = rv_crisis.idxmax()
    ax1.annotate("RV above IV. Crisis.",
                 xy=(peak_idx, df["Realized Vol (30d)"][peak_idx]),
                 xytext=(peak_idx - pd.Timedelta(days=200),
                         df["Realized Vol (30d)"][peak_idx] + 8),
                 fontsize=8, color="#ff4f4f",
                 arrowprops=dict(arrowstyle="->", color="#ff4f4f", lw=1.0))
 
mid_date = df.index[len(df)//3]
mid_iv   = df["Implied Vol (VIX)"].iloc[len(df)//3]
mid_rv   = df["Realized Vol (30d)"].iloc[len(df)//3]
if mid_iv - mid_rv > 2:
    ax1.annotate(f"Vol Risk Premium. Avg. {vrp.mean():.1f} vol pts.",
                 xy=(mid_date, (mid_iv + mid_rv) / 2),
                 xytext=(mid_date + pd.Timedelta(days=120), mid_iv + 6),
                 fontsize=8, color="#4e9af1",
                 arrowprops=dict(arrowstyle="->", color="#4e9af1", lw=1.0))
 
ax1.set_ylabel("Annualised Volatility (%)", color="#ccc", fontsize=11)
ax1.legend(fontsize=9, loc="upper right",
           facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
ax1.set_title("SPY. Realized Vol vs Implied Vol (VIX).\nBlue fill. Vol Risk Premium. Red fill. Crisis regime.",
              color="white", fontsize=12, fontweight="bold", pad=10)
ax1.yaxis.label.set_color("#ccc")
ax1.grid(True, alpha=0.12, color="#555")
 
colors_vrp = ["#4e9af1" if v >= 0 else "#ff4f4f" for v in vrp]
ax2.bar(df.index, vrp, color=colors_vrp, alpha=0.75, width=1.5)
ax2.axhline(0, color="#aaa", lw=0.8, ls="--")
ax2.axhline(vrp.mean(), color="#4e9af1", lw=1.0, ls=":",
            label=f"Avg VRP. {vrp.mean():.1f} vol pts.")
ax2.set_ylabel("IV minus RV (vol pts)", color="#ccc", fontsize=9)
ax2.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc", loc="upper right")
ax2.grid(True, alpha=0.10, color="#555")
ax2.yaxis.label.set_color("#ccc")
 
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), color="#aaa")
 
plt.tight_layout(h_pad=0.5)
plt.savefig("realized_vs_implied_vol.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("Saved. realized_vs_implied_vol.png")
 


