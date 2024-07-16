import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Configuration
datafile = 'hPFMPCD_visco_data.csv'

# Load the data
df = pd.read_csv(datafile, delim_whitespace=True)
print(df)

# Initialize dictionaries to store results
fit_results = {}

# Plotting setup
plt.figure(figsize=(10, 6))

# Get unique densities and sort dataframe by density
densities = df['Density'].unique()
df = df.sort_values(by='Density')

# Plot raw data
for density in densities:
    subset = df[df['Density'] == density]
    plt.scatter(subset['Temperature'], subset['CollisionRate'], label=f'Density {density}')

# Adding labels and title
plt.xlabel('Temperature')
plt.ylabel('Collision Rate')
plt.title('CollisionRate vs. Temperature')
plt.legend()
plt.show()

# Calculating and printing slopes and fit metrics
for density in densities:
    subset = df[df['Density'] == density]
    x_data = subset['Temperature'].values.reshape(-1, 1)
    y_data = subset['CollisionRate'].values

    # Fit linear regression model
    model = LinearRegression().fit(x_data, y_data)
    a_fit = model.coef_[0]
    b_fit = model.intercept_

    # Calculate fitted values
    y_pred = model.predict(x_data)

    # Evaluate the fit
    mse = mean_squared_error(y_data, y_pred)
    mae = mean_absolute_error(y_data, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_data, y_pred)

    # Store the results
    fit_results[density] = {
        'a': a_fit,
        'b': b_fit,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

    #Plot
    plt.scatter(x_data, y_data, label=f'Density {density}', marker='o')
    plt.plot(x_data, y_pred, linestyle='dotted')

# Adding labels and title for the linear fit plot
plt.xlabel('Temperature')
plt.ylabel('Collision rate')
plt.title('Collision rate vs. Temperature with Linear Fit')
plt.legend()
plt.show()

# Print the fit results including slopes
for density, results in fit_results.items():
    print(f"Density {density}:")
    print(f"  Fitted parameters: a = {results['a']:.4f}, b = {results['b']:.4f}")
    print(f"  Slope: {results['slope']:.4f}")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"  MAE: {results['mae']:.6f}")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  R-squared (R2): {results['r2']:.6f}")