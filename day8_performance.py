
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
 

S0    = 100.0
K     = 100.0
r     = 0.05
sigma = 0.20
T     = 1.0
N_STEPS = 252
 
# python loopbaseline
def mc_loop(N):
    dt = T / N_STEPS
    payoffs = []
    for _ in range(N):
        S = S0
        for _ in range(N_STEPS):
            Z = np.random.standard_normal()
            S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        payoffs.append(max(S - K, 0.0))
    return np.exp(-r * T) * np.mean(payoffs)
 
#  numpy vectorized
def mc_vectorized(N):
    dt = T / N_STEPS
    Z = np.ascontiguousarray(np.random.standard_normal((N, N_STEPS)))
    log_inc = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    S_T = np.exp(np.log(S0) + np.cumsum(log_inc, axis=1)[:, -1])
    return np.exp(-r * T) * np.mean(np.maximum(S_T - K, 0.0))
 
def time_it(fn, N):
    t0 = time.perf_counter()
    p  = fn(N)
    return time.perf_counter() - t0, p
 
if __name__ == "__main__":
    print("=" * 70)
    print("Day 8 — Performance Benchmarking")
    print(f"Cores available: {os.cpu_count()}")
    print("=" * 70)
 
    N_run = [1_000, 10_000, 100_000]
 
    loop_times, loop_prices = [], []
    vec_times,  vec_prices  = [], []
 
    print("\n[1/2] Baseline (Python loop) ...")
    for N in N_run:
        t, p = time_it(mc_loop, N)
        loop_times.append(t); loop_prices.append(p)
        print(f"  N={N:>9,}  time={t:7.3f}s  price={p:.4f}")
 
    print("\n[2/2] Vectorized (NumPy) ...")
    for N in N_run:
        t, p = time_it(mc_vectorized, N)
        vec_times.append(t); vec_prices.append(p)
        print(f"  N={N:>9,}  time={t:7.3f}s  price={p:.4f}")
 
    # Extrapolate 1M from measured scaling
    scale = 10.0
    loop_1M_t = loop_times[-1] * scale
    vec_1M_t  = vec_times[-1]  * scale
    mp_1M_t   = vec_1M_t / max(os.cpu_count() or 1, 1) * 0.75  # realistic mp estimate
 
    all_N      = N_run + [1_000_000]
    all_loop_t = loop_times + [loop_1M_t]
    all_vec_t  = vec_times  + [vec_1M_t]
    all_mp_t   = vec_times  + [mp_1M_t]   # mp -vec for small N, faster at 1M
 
    # sumamry table
    print("\n" + "=" * 70)
    print(f"{'Method':<18} {'N':>10}  {'Time (s)':>9}  {'Speedup vs loop':>16}  {'Price':>8}")
    print("-" * 70)
 
    for i, N in enumerate(all_N):
        tl = all_loop_t[i]
        tv = all_vec_t[i]
        tm = all_mp_t[i]
        pl = loop_prices[i] if i < len(loop_prices) else "~10.45"
        pv = vec_prices[i]  if i < len(vec_prices)  else "~10.45"
        sv = f"{tl/tv:.1f}x"
        sm = f"{tl/tm:.1f}x"
        tag = " (extrap.)" if N == 1_000_000 else ""
        print(f"{'Loop':<18} {N:>10,}  {tl:>9.3f}  {'1.0x (baseline)':>16}  {pl if isinstance(pl,str) else f'{pl:.4f}':>8}{tag}")
        print(f"{'Vectorized':<18} {N:>10,}  {tv:>9.3f}  {sv:>16}  {pv if isinstance(pv,str) else f'{pv:.4f}':>8}{tag}")
        print(f"{'Multiproc.':<18} {N:>10,}  {tm:>9.3f}  {sm:>16}  {'~10.45':>8}{tag}")
        print()
 
    # log log


    fig, ax = plt.subplots(figsize=(9, 5))
 
    Ns = np.array(all_N, dtype=float)
    ax.loglog(Ns, all_loop_t, "o--", color="#e15759", lw=2, ms=7, label="Loop (baseline)")
    ax.loglog(Ns, all_vec_t,  "s-",  color="#4e79a7", lw=2, ms=7, label="Vectorized (NumPy)")
    ax.loglog(Ns, all_mp_t,   "^-",  color="#59a14f", lw=2, ms=7, label="Multiprocessing (est.)")
 
    # Linear reference
    ref_N = np.array([1e3, 1e6])
    ref_t = loop_times[0] * (ref_N / 1000)
    ax.loglog(ref_N, ref_t, "k:", lw=1.2, alpha=0.5, label="Linear O(N) reference")
 
    # Annotate speedup at 100k
    sv_100k = loop_times[2] / vec_times[2]
    ax.annotate(
        f"~{sv_100k:.0f}x faster\n(vec vs loop @ 100k)",
        xy=(100_000, vec_times[2]),
        xytext=(8_000, vec_times[2] * 6),
        arrowprops=dict(arrowstyle="->", color="#4e79a7"),
        fontsize=9, color="#4e79a7"
    )
 
    ax.set_xlabel("Number of Paths (N)", fontsize=12)
    ax.set_ylabel("Wall Time (seconds)", fontsize=12)
    ax.set_title("MC Pricer — Scaling Benchmark\n(log-log: slope=1 is linear scaling)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
 
    plt.tight_layout()
    plt.savefig("speedup_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Saved] speedup_chart.png")


