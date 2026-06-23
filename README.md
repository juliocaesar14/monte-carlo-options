# Monte Carlo Option Pricing Engine

This project builds a Monte Carlo option pricing engine from scratch over 8 phases, starting from simulating stock price paths and ending with a real market calibration comparing two stochastic volatility models against live SPY options data.


Phase 1, GBM Simulation.

Stock price paths are simulated using Geometric Brownian Motion. At each time step the price evolves as S times exp of the quantity mu minus half sigma squared times dt, plus sigma times sqrt dt times Z, where Z is a standard normal draw. 1000 paths are simulated over 252 trading days. The simulated mean terminal price is checked against the known theoretical value S0 times exp of mu times T, and they match. Parameters used are S0 of 100, drift of 0.08, volatility of 0.20.



Phase 2, European Option Pricing.

A European call and put are priced under the risk neutral measure using Monte Carlo. Terminal stock prices are simulated using the same GBM formula but with the risk free rate replacing the drift. Payoffs are discounted back and averaged. Results are benchmarked against Black Scholes closed form prices. At N of 100,000 paths, the Monte Carlo call price matches Black Scholes to within 0.3 percent and the put to within 0.2 percent.



Phase 3, Greeks via Finite Difference.

Delta, Gamma, Vega, Theta, and Rho are estimated using central finite differences. Each Greek bumps one input by a small h, measures the price change, and divides by 2h. The same set of random draws Z is reused across all bumps, which is called Common Random Numbers, and reduces noise significantly. All five Greeks match their Black Scholes analytical values with errors below 0.001.



Phase 4, Variance Reduction.

Three Monte Carlo pricing methods are compared at the same number of paths. Plain Monte Carlo is the baseline. Antithetic Variates pairs each random draw Z with its mirror negative Z, cutting variance by roughly 2x. Control Variates uses the known expected value of the terminal stock price to correct the estimator, achieving a variance reduction of around 7x over plain Monte Carlo. All three converge at the rate of one over square root N on a log log plot, but Control Variates starts from a much lower standard error baseline.



Phase 5, Exotic Options.

Two path dependent options are priced using full simulated paths of 252 steps. The Asian call uses the average stock price along the path as the effective spot, rather than the terminal price. Its price of 5.79 is lower than the vanilla European call of 10.48 because averaging reduces the effective volatility seen by the option, and option value increases with volatility. An up and out barrier call knocks out if the path ever touches the upper barrier of 120, and a down and in barrier put only activates if the path ever touches the lower barrier of 80.



Phase 6, Implied Volatility via Bisection.

Implied volatility is recovered from option prices using bisection search over the volatility input to Black Scholes. A synthetic smile is constructed as a quadratic function of moneyness across a grid of 17 strikes and 7 expiries, priced with Black Scholes, then recovered from those prices. The round trip maximum error across the full grid is below 0.000001, confirming the bisection solver is correct. The synthetic smile is symmetric by construction. Phase 7 checks whether real SPY market data shows an asymmetric skew instead.



Phase 7, Real Market Calibration on SPY.

Live SPY options chain data is pulled using yfinance for three expiries of roughly 1 month, 3 months, and 6 months. Options are filtered to remove stale quotes, wide bid ask spreads, and contracts far outside the moneyness range of 0.75 to 1.25. For each surviving contract, implied volatility is recovered from the mid price using the bisection solver from Phase 6 and overlaid against the implied volatility reported by yfinance directly. The recovered curves show a clear put skew, meaning out of the money puts carry higher implied volatility than at the money options. This skew steepens for short expiries and flattens at 6 months. The put skew exists because downside puts carry a crash risk premium that is not present in the symmetric Black Scholes world.

A synthetic version of the same calibration is also included in day7 synthetic.py, which uses a hand specified smile function to run the same round trip test without depending on live market data.



Phase 8, Performance Benchmarking.

A Python loop pricer and a NumPy vectorised pricer are benchmarked at path counts of 1000, 10000, and 100000. The vectorised pricer is around 30x faster than the loop at 100,000 paths and scales linearly in both cases as confirmed by the slope on the log log chart. A multiprocessing estimate is extrapolated from the vectorised timings assuming linear scaling across available CPU cores.



Phase 9, Model Comparison, GBM versus Heston versus Market.

The final phase prices a strip of 15 put strikes from moneyness 0.80 to 1.20 at a 3 month expiry under two models, then compares the implied volatility smiles against live SPY market data.

Under GBM with constant volatility, the implied vol is flat across all strikes by construction. Under the Heston stochastic volatility model, the variance process is mean reverting and correlated with the stock price via a negative rho parameter. This negative correlation means that when the stock price falls, variance rises, which makes out of the money puts more expensive. The Heston model reproduces a downward sloping put skew that GBM cannot produce.

The live SPY market data confirms the skew is real. GBM misprices out of the money puts because it cannot account for crash risk. Heston matches the qualitative shape of the market skew. The key result is that the difference between GBM and Heston prices is largest for deep out of the money puts, precisely where tail risk is most relevant.



Results Summary.

Monte Carlo vs Black Scholes call price error at 100k paths, below 0.3 percent. Greeks error vs analytical, all below 0.001. Variance reduction ratio, Control Variates achieves around 7x over plain Monte Carlo. Bisection round trip implied vol error, below 0.000001. Vectorised pricer speedup over Python loop at 100k paths, around 30x.



Requirements.

numpy, scipy, matplotlib, yfinance.

Install with pip install numpy scipy matplotlib yfinance.



References.

Black and Scholes, 1973, The Pricing of Options and Corporate Liabilities.
Boyle, 1977, Options, A Monte Carlo Approach.
Heston, 1993, A Closed Form Solution for Options with Stochastic Volatility.

