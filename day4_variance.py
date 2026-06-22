import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Parameters
S0    = 100.0
K     = 100.0
r     = 0.05
sigma = 0.20
T     = 1.0

np.random.seed(42)

# Plain MC pricer (baseline)
def mc_plain(N):
    Z   = np.random.standard_normal(N)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(S_T - K, 0)
    price   = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs, ddof=1) / np.sqrt(N)
    return price, std_err

price, se = mc_plain(100_000)
print(f"Plain MC:  price={price:.4f}  std_err={se:.4f}")

# Antithetic Variates
def mc_antithetic(N):
    Z    = np.random.standard_normal(N // 2)  # half the draws
    Z_av = -Z                                  # mirror them

    S_T1 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    S_T2 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_av)

    payoffs1 = np.maximum(S_T1 - K, 0)
    payoffs2 = np.maximum(S_T2 - K, 0)

    # average the pair BEFORE discounting
    payoffs  = (payoffs1 + payoffs2) / 2

    price    = np.exp(-r * T) * np.mean(payoffs)
    std_err  = np.exp(-r * T) * np.std(payoffs, ddof=1) / np.sqrt(N // 2)
    return price, std_err

price_av, se_av = mc_antithetic(100_000)
print(f"Antithetic: price={price_av:.4f}  std_err={se_av:.4f}")


# Control Variates
def mc_control_variate(N):
    Z   = np.random.standard_normal(N)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T - K, 0)

    # S_T is our control — we know its true mean exactly
    expected_S_T = S0 * np.exp(r * T)

    # optimal coefficient c*
    c_star = -np.cov(payoffs, S_T)[0, 1] / np.var(S_T)

    # adjusted payoffs
    payoffs_cv = payoffs + c_star * (S_T - expected_S_T)

    price   = np.exp(-r * T) * np.mean(payoffs_cv)
    std_err = np.exp(-r * T) * np.std(payoffs_cv, ddof=1) / np.sqrt(N)
    return price, std_err

price_cv, se_cv = mc_control_variate(100_000)
print(f"Control CV: price={price_cv:.4f}  std_err={se_cv:.4f}")


# Comparison table
print("\n" + "="*60)
print(f"{'Method':<15} {'Price':>10} {'Std Error':>12} {'Var Ratio':>12}")
print("="*60)
print(f"{'Plain MC':<15} {price:>10.4f} {se:>12.4f} {'1.00x':>12}")
print(f"{'Antithetic':<15} {price_av:>10.4f} {se_av:>12.4f} {(se/se_av)**2:>11.2f}x")
print(f"{'Control CV':<15} {price_cv:>10.4f} {se_cv:>12.4f} {(se/se_cv)**2:>11.2f}x")
print("="*60)


# Convergence plot
N_values = [100, 500, 1000, 5000, 10000, 50000, 100000]

se_plain, se_anti, se_cv = [], [], []

for n in N_values:
    _, se1 = mc_plain(n)
    _, se2 = mc_antithetic(n)
    _, se3 = mc_control_variate(n)
    se_plain.append(se1)
    se_anti.append(se2)
    se_cv.append(se3)

plt.figure(figsize=(10, 6))
plt.loglog(N_values, se_plain, 'b-o', label='Plain MC')
plt.loglog(N_values, se_anti,  'r-o', label='Antithetic')
plt.loglog(N_values, se_cv,    'g-o', label='Control Variate')
plt.xlabel('Number of Paths (N)')
plt.ylabel('Standard Error')
plt.title('Variance Reduction Convergence')
plt.legend()
plt.grid(True)
plt.savefig('variance_reduction.png')
plt.show()
print("Plot saved as variance_reduction.png")