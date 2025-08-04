
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import norm  # Importing norm for the Black-Scholes calculations

# Black-Scholes formula implementation
def black_scholes(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'Put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type should be 'Call' or 'Put'")
    
    return price

# Function to predict option prices
def predict_option_prices(input_csv):
    # Load the input data
    df = pd.read_csv(StringIO(input_csv))

    # Normalize MarketFearIndex and RiskfreeRate
    df['Volatility'] = df['MarketFearIndex'] / 100  # MarketFearIndex becomes volatility
    df['RiskfreeRateNorm'] = df['RiskfreeRate'] / 100  # Normalized risk-free rate

    # Calculate Black-Scholes price for each row
    df['BlackScholesPrice'] = df.apply(
        lambda row: black_scholes(row['OptionType'], row['Spot'], row['Strike'], 
                                   row['TimeToExpiry'], row['RiskfreeRateNorm'], row['Volatility']),
        axis=1
    )

    # Define the features for the model
    X = df[['BlackScholesPrice', 'BuySellRatio', 'Strike', 'Spot', 'TimeToExpiry', 'RiskfreeRateNorm', 'Volatility']]

    # Weights from the previously trained model
    weights = np.array([0.959559, 2.146508, -0.000396, -0.053197, -2.855389, -8.137899, -14.942021])
    intercept = 7.915819

    # Predict the option prices using the regression equation
    df['PredictedOptionPrice'] = np.dot(X, weights) + intercept

    # Ensure all predicted prices are non-negative
    df['PredictedOptionPrice'] = df['PredictedOptionPrice'].clip(lower=0)

    # Prepare the output DataFrame
    output_df = df[['Id', 'PredictedOptionPrice']].rename(columns={'PredictedOptionPrice': 'OptionPrice'})

    return output_df

# Main code to read input and output predictions
if __name__ == "__main__":
    inputdata = input()
    inputdata = inputdata.replace("\\n","\n")
    option_prices = predict_option_prices(inputdata)
    
    # Output the results
    print("Id,OptionPrice")
    for _, row in option_prices.iterrows():
        print(f"{row['Id']},{row['OptionPrice']:.6f}")
