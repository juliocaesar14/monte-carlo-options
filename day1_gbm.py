import numpy as np
import matplotlib.pyplot as plt

# Parameters,these are the only inputs GBM needs
S0 = 100      #starting stock price
mu = 0.08     #annual drift (8%)
sigma = 0.20     #annual volatility (20%)
T = 1.0      #time horizon in years
steps = 252      #trading days in a year
N = 1000     #number of simulated paths

#Time step
dt = T / steps

# Shape: (steps, N),each column is one path
Z = np.random.standard_normal((steps, N))          # random shocks
increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
log_returns = np.cumsum(increments, axis=0)        # cumulative sum

# Full price paths: prepend S0 at time 0
price_paths = S0 * np.exp(np.vstack([np.zeros(N), log_returns]))
# Shape is now(steps+1, N)

# check all now
S_T = price_paths[-1, :]                          
 # terminal prices
print(f"Simulated mean S(T):    {S_T.mean():.2f}")
print(f"Theoretical mean S(T):  {S0 * np.exp(mu * T):.2f}") 
 # should match!
print(f"Simulated std S(T):     {S_T.std():.2f}")

#all plots
plt.figure(figsize=(10, 5))
plt.plot(price_paths[:, :200], alpha=0.15, linewidth=0.7)   # plot 200 paths
plt.axhline(S0, color='black', linewidth=1, linestyle='--', label='S₀ = 100')
plt.title(f'GBM simulation — {N} paths (μ={mu}, σ={sigma}, T={T}y)')
plt.xlabel('Trading day')
plt.ylabel('Stock price')
plt.legend()
plt.tight_layout()
plt.savefig('day1_paths.png', dpi=150)
plt.show()

