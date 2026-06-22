import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

stock_price = 100.0
interest_rate = 0.05
base_volatility = 0.20
skew_strength = 0.50

strike_list = np.linspace(80, 120, 17)
expiry_list = np.array([1/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])


#usig same scholes formula
def black_scholes_price(stock_price, strike, interest_rate, volatility, time_to_expiry, option_type="call"):
    d1 = (np.log(stock_price / strike) + (interest_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    if option_type == "call":
        return stock_price * norm.cdf(d1) - strike * np.exp(-interest_rate * time_to_expiry) * norm.cdf(d2)
    else:
        return strike * np.exp(-interest_rate * time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)


def find_implied_vol(market_price, stock_price, strike, interest_rate, time_to_expiry, option_type="call", tolerance=1e-6, max_steps=200):
    low_guess = 0.001
    high_guess = 5.0

    if option_type == "call":
        floor_value = max(stock_price - strike * np.exp(-interest_rate * time_to_expiry), 0.0)
    else:
        floor_value = max(strike * np.exp(-interest_rate * time_to_expiry) - stock_price, 0.0)
    if market_price < floor_value - tolerance:
        return np.nan

    for step in range(max_steps):
        mid_guess = (low_guess + high_guess) / 2
        guessed_price = black_scholes_price(stock_price, strike, interest_rate, mid_guess, time_to_expiry, option_type)
        if abs(guessed_price - market_price) < tolerance:
            return mid_guess
        if guessed_price > market_price:
            high_guess = mid_guess
        else:
            low_guess = mid_guess
    return mid_guess



#fake market testing
true_vol_grid = np.zeros((len(expiry_list), len(strike_list)))
price_grid = np.zeros((len(expiry_list), len(strike_list)))
recovered_vol_grid = np.zeros((len(expiry_list), len(strike_list)))

for row, time_to_expiry in enumerate(expiry_list):
    for col, strike in enumerate(strike_list):
        moneyness = strike / stock_price
        true_volatility = base_volatility + skew_strength * (moneyness - 1) ** 2
        true_vol_grid[row, col] = true_volatility

        price = black_scholes_price(stock_price, strike, interest_rate, true_volatility, time_to_expiry, "call")
        price_grid[row, col] = price

        recovered_vol_grid[row, col] = find_implied_vol(price, stock_price, strike, interest_rate, time_to_expiry, "call")


#vheck error ehich should be close to 0
error_grid = np.abs(recovered_vol_grid - true_vol_grid)
print("Round trip check, baking in a smile and recovering it with bisection")
print(f"Largest error: {error_grid.max():.6f}")
print(f"Average error: {error_grid.mean():.6f}")
if error_grid.max() < 1e-4:
    print("Round trip worked.")
else:
    print("Something is off, check the bisection function.")


chosen_row = 4
chosen_expiry = expiry_list[chosen_row]
moneyness_values = strike_list / stock_price

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(moneyness_values, true_vol_grid[chosen_row], "o", linestyle="solid", color="steelblue", label="true smile")
ax1.plot(moneyness_values, recovered_vol_grid[chosen_row], "x", linestyle="dashed", color="orange", label="recovered smile")
ax1.set_xlabel("Moneyness, strike over stock price")
ax1.set_ylabel("Implied Volatility")
ax1.set_title(f"Volatility Smile, expiry of {chosen_expiry:.2f} years")
ax1.legend()
fig1.tight_layout()
fig1.savefig("vol_smile.png", dpi=150)
print("Saved vol_smile.png")

strike_grid, expiry_grid = np.meshgrid(strike_list, expiry_list)

fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.add_subplot(projection="3d")
surface = ax2.plot_surface(strike_grid, expiry_grid, recovered_vol_grid, cmap="viridis", edgecolor="none", alpha=0.95)
ax2.set_xlabel("Strike")
ax2.set_ylabel("Expiry, in years")
ax2.set_zlabel("Implied Volatility")
ax2.set_title("Implied Volatility Surface")
fig2.colorbar(surface, ax=ax2, shrink=0.6, label="Implied Volatility")
fig2.tight_layout()
fig2.savefig("vol_surface.png", dpi=150)
print("Saved vol_surface.png")



