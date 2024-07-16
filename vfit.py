import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


# Read data functions
def read_data_Lautebshlaeger(data):
    df = pd.read_csv(data, delim_whitespace=True)
    print(df)
    Temp = df['Temp']
    thermal_cond = df['ThCond']
    thermal_cond_err = df['ThCond_err']
    eta = df['eta_LJ']
    eta_err = df['eta_err']
    return Temp, eta, eta_err, thermal_cond, thermal_cond_err


def read_data(data):
    df = pd.read_csv(data, delim_whitespace=True)
    print(df)
    Temp = df['T']
    Col_rate = df['ColisionRate']
    eta_hPF = df['eta_HPF']
    eta_LJ = df['eta_LJ']

    dens = float(filename.split('_')[2].replace('.txt', ''))

    return Temp, Col_rate, eta_hPF, eta_LJ, dens


#Exponential function fitting
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


#Arrhenius fitting
def arrhenius_func(x, a, b):
    return - (a * x) + b


# MSE loss function for exponential fit
def mse_loss_exp(params, x, y):
    a, b, c = params
    y_pred = exp_func(x, a, b, c)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    return mse


# MSE loss function for Arrhenius fit
def mse_loss_arrhenius(params, x, y):
    a, b = params
    y_pred = arrhenius_func(x, a, b)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    return mse


# Function to perform fitting and plot results
def perform_fitting_and_plot(x_data, y_data, dens, fit_type='exp'):
    if fit_type == 'exp':
        initial_guess = [6.0, -0.3, 0.0]
        mse_loss = mse_loss_exp
        fit_func = exp_func
        y_label = 'Collision rate'
        x_label = 'Temperature'
        fit_label = 'Exponential Fit'
    elif fit_type == 'arrhenius':
        initial_guess = [-2.5, 1.7]
        mse_loss = mse_loss_arrhenius
        fit_func = arrhenius_func
        y_label = 'ln($\eta$)'
        x_label = '1/T'
        fit_label = 'Arrhenius Fit'
        y_data = np.log(y_data)
        x_data = 1 / np.array(x_data)

    max_iterations = 1000
    deviation_threshold = 1e-6

    prev_mse = float('inf')
    tolerance = deviation_threshold

    for iteration in range(max_iterations):
        result = minimize(mse_loss, initial_guess, args=(x_data, y_data))
        current_mse = mse_loss(result.x, x_data, y_data)
        tolerance = abs(prev_mse - current_mse)

        print(f"Iteration {iteration + 1}: MSE = {current_mse:.6f}, Tolerance = {tolerance:.6f}")

        if tolerance < deviation_threshold:
            break

        prev_mse = current_mse
        initial_guess = result.x

    fit_params = result.x
    y_pred = fit_func(x_data, *fit_params)
    mse = mean_squared_error(y_data, y_pred)
    mae = mean_absolute_error(y_data, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_data, y_pred)

    textstr = '\n'.join((
        f'MSE: {mse:.4f}',
        f'MAE: {mae:.4f}',
        f'RMSE: {rmse:.4f}',
        f'R$^{2}$: {r2:.4f}',
        f'ρ : {dens:.4f}'
    ))

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R-squared (R2): {r2:.6f}")
    print(f"Fitted parameters: {fit_params}")

    # Plotting function
    def plot_results():
        plt.scatter(x_data, y_data, label='Data')
        plt.plot(x_data, y_pred, color='red', label=fit_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='lower right')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                       verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.show()

    plot_results()

    return fit_params, mse, mae, rmse, r2


# Main script
filename = 'data_visco_0.8.txt'

Temp, Col_rate, eta_hPF, eta_LJ, dens = read_data(filename)

# Choose which fit to perform: 'exp' for exponential, 'arrhenius' for Arrhenius
fit_type = 'exp'  # Change this to 'arrhenius' to perform Arrhenius fit

if fit_type == 'exp':
    x_data = Temp
    y_data = Col_rate
else:
    x_data = Temp
    y_data = eta_LJ

fit_params, mse, mae, rmse, r2 = perform_fitting_and_plot(x_data, y_data, dens, fit_type)