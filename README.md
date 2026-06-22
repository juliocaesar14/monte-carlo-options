# Monte Carlo Option Pricing — Day 1: GBM Simulation

This script simulates stock price paths using Geometric Brownian Motion (GBM) as the first step project on Monte Carlo methods in quantitative finance. At each time step, the stock price evolves as S(t+dt) = S(t) × exp((μ − σ²/2)×dt + σ×√dt×Z), where Z is a standard normal random draw. Using 1000 paths over 1 trading year (252 steps), the simulation verifies itself by comparing the simulated mean of terminal prices against the known theoretical value S0 × exp(μ × T). The output is a plot of 200 price paths saved as `day1_paths.png`. Parameters used: starting price S0 = 100, annual drift μ = 0.08, volatility σ = 0.20. Requires numpy and matplotlib.

# References
- Black & Scholes (1973). The Pricing of Options and Corporate Liabilities.
- Boyle (1977). Options: A Monte Carlo Approach.


Phase 2:
## European Option Pricing
Prices a European call and put via risk-neutral Monte Carlo, benchmarked against
the Black-Scholes closed form.

Phase 3:
## Greeks via Finite Difference
Estimates Delta, Gamma, Vega, Theta, Rho using central finite differences.
Each Greek bumps one input by a small h, measures price change, divides by 2h.
Same Z array reused across all bumps (Common Random Numbers) to reduce noise.
Validated against Black-Scholes analytical Greeks- all errors < 0.1.


Phase 4:
## Variance redcution
Compares three MC pricing methods at the same N:
1) Plain MC: baseline
2) Antithetic Variates: pair each Z with -Z, 2x variance reduction
3) Control Variates: use known E[S_T] to correct noise, 6.9x variance reduction
All three converge at 1/sqrt(N) but control variates starts much lower.

