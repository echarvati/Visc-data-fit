import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function


# Internal packages
from nn_structure_SymReg import Data, NeuralNetwork

plt.rcParams.update({
    'font.size': 30,
    'font.family': 'Arial',
    'figure.figsize': (10, 10)
})



# Woodcock Viscosity Function
def Woodcock_equation(rho, T):
    """
    Calculate the viscosity (Y) of Lennard-Jones fluids based on density (rho) and temperature (T).
    """
    CAH = 3.025 # Empirical constant for soft-sphere scaling: 3.025 (Woodcock parameter)
    pi = np.pi

    # Zero-density viscosity term
    eta_0 = (5 / (16 * np.sqrt(pi))) * T

    # Second viscosity virial coefficient B*(T*)
    B_T = -2 + (T / 8) + (1 / T**4)

    # Soft-sphere scaling term
    soft_sphere_term = CAH * rho * (T**(-1 / 3)) * (rho**4)

    # Final viscosity
    Y = eta_0 * (1 + B_T * rho) + soft_sphere_term
    return Y


def prepare_data_with_Woodcock(X, Cp):
    """
    Prepare the dataset using the Woodcock equation with Cp as an additional feature.
    """
    rho = X[:, 1]  # Density
    T = X[:, 0]  # Temperature

    # Compute Woodcock viscosity predictions
    Y_Woodcock = np.array([Woodcock_equation(r, t) for r, t in zip(rho, T)])

    # Combine Woodcock predictions with Cp
    X_new = np.column_stack((Y_Woodcock, Cp))  # New dataset with Cp and Y_Woodcock
    return X_new, Y_Woodcock


def plot_error_distribution(Y_true, Y_pred):
    errors = Y_true - Y_pred
    plt.figure(dpi=300)
    plt.hist(errors, bins=20, edgecolor='k', color='skyblue')
    plt.title("Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("error_distribution.png", bbox_inches="tight")
    plt.show()



def visualize_symbolic_regression(Y_true_train, Y_pred_train, Y_true_test, Y_pred_test):
    plt.figure(dpi=300)
    plt.scatter(Y_true_train, Y_pred_train, alpha=0.6, label=r'$\eta_{train}$', color="blue", s=150)
    plt.scatter(Y_true_test, Y_pred_test, alpha=0.6, label=r'$\eta_{test}$', color="orange", s=150)
    plt.plot([min(Y_true_train), max(Y_true_train)], [min(Y_true_train), max(Y_true_train)],
             color="black", linestyle="--", label=r'$\eta_{HPF-MPCD}$', linewidth=7)
    plt.xlabel(r'$\eta_{HPF-MPCD}$')
    plt.ylabel(r'$\eta_{predicted}$')
    plt.legend(fontsize=30)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("predicted_vs_actual.png", bbox_inches="tight")
    plt.show()

    residuals_train = Y_true_train - Y_pred_train
    residuals_test = Y_true_test - Y_pred_test
    plt.figure(dpi=300)
    plt.scatter(Y_pred_train, residuals_train, alpha=0.6, label=r'$dev._{train}$', color="blue", s=150)
    plt.scatter(Y_pred_test, residuals_test, alpha=0.6, label=r'$dev._{test}$', color="orange", s=150)
    plt.axhline(0, color="black", linestyle="--", label="Zero Dev.",linewidth=7)
    plt.xlabel(r'$\eta_{predicted}$')
    plt.ylabel("dev.")
    plt.legend(fontsize=30)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Symbolic Regression with Cp
def run_SymbolicRegression_with_Cp(X_train, Y_train, X_test, Y_test):
    print("\nRunning Symbolic Regression with Woodcock Baseline and Cp...")

      # Split data into train and test sets
    Cp_train = X_train[:, 2]
    Cp_test = X_test[:, 2]

    # Apply Woodcock Baseline adjustments
    X_train_new, Y_Woodcock_train = prepare_data_with_Woodcock(X_train, Cp_train)
    X_test_new, Y_Woodcock_test = prepare_data_with_Woodcock(X_test, Cp_test)


    # Symbolic Regression with updated function set
    sym_reg = SymbolicRegressor(
        population_size=8000,  # Reduced for efficiency
        generations=30,  # Reduced for quicker experimentation
        stopping_criteria=0.001,
        function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt'),
        parsimony_coefficient=0.01,
        verbose=1,
        random_state=42
    )

    # Fit Symbolic Regression model
    sym_reg.fit(X_train_new, Y_train)

    # Predict and evaluate
    Y_train_pred = sym_reg.predict(X_train_new)
    Y_test_pred = sym_reg.predict(X_test_new)

    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    mae_test = mean_absolute_error(Y_test, Y_test_pred)

    print(f"Symbolic Regression Training R²: {r2_train:.4f}")
    print(f"Symbolic Regression Test R²: {r2_test:.4f}")
    print(f"Symbolic Regression Test RMSE: {rmse_test:.4f}")
    print(f"Symbolic Regression Test MAE: {mae_test:.4f}")
    print("Derived Symbolic Equation:")
    print(sym_reg._program)

    # Visualizations
    visualize_symbolic_regression(Y_train, Y_train_pred, Y_test, Y_test_pred)
    plot_error_distribution(Y_test, Y_test_pred)

    return sym_reg

def run_SymbolicRegression_with_forced_Cp(X_train, Y_train, X_test, Y_test):
    """
    Refines the viscosity model eta = f(T, rho, Cp) by combining the Woodcock baseline with forced Cp inclusion.
    """

    print("\nRunning Symbolic Regression with Forced Cp Inclusion...")

    # Step 1: Extract features
    T_train, rho_train, Cp_train = X_train[:, 0], X_train[:, 1], X_train[:, 2]
    T_test, rho_test, Cp_test = X_test[:, 0], X_test[:, 1], X_test[:, 2]

    # Step 2: Compute the Woodcock baseline
    Y_Woodcock_train = np.array([Woodcock_equation(rho, T) for rho, T in zip(rho_train, T_train)])
    Y_Woodcock_test = np.array([Woodcock_equation(rho, T) for rho, T in zip(rho_test, T_test)])

    # Step 3: Force Cp inclusion by combining features
    # Introduce Cp as an explicit interaction term
    X_train_for_regression = np.column_stack((Y_Woodcock_train, Cp_train, Y_Woodcock_train * Cp_train))
    X_test_for_regression = np.column_stack((Y_Woodcock_test, Cp_test, Y_Woodcock_test * Cp_test))

    # Step 4: Initialize symbolic regression model
    sym_reg = SymbolicRegressor(
        population_size=8000,
        generations=30,
        stopping_criteria=0.001,
        function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt'),
        parsimony_coefficient=0.001,  # Penalize complexity
        verbose=1,
        random_state=42
    )

    # Step 5: Fit symbolic regression to the full viscosity
    sym_reg.fit(X_train_for_regression, Y_train)

    # Step 6: Predict viscosity for train and test sets
    Y_train_pred = sym_reg.predict(X_train_for_regression)
    Y_test_pred = sym_reg.predict(X_test_for_regression)

    # Step 7: Evaluate performance
    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    mae_test = mean_absolute_error(Y_test, Y_test_pred)

    rmse_train = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    mae_train = mean_absolute_error(Y_train, Y_train_pred)

    print("\nEvaluation Metrics:")
    print(f"Training R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")
    print(f"Test RMSE: {rmse_test:.4f}")
    print(f"Test MAE: {mae_test:.4f}")
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Train MAE: {mae_train:.4f}")

    # Step 8: Output the derived equation
    print("\nDerived Symbolic Equation for Viscosity (Y = f(Y_Woodcock, Cp)):")
    print(sym_reg._program)

    # Step 9: Visualize the results
    visualize_symbolic_regression(Y_train, Y_train_pred, Y_test, Y_test_pred)
    plot_error_distribution(Y_test, Y_test_pred)

    return sym_reg


def run_SymbolicRegression(X_train, Y_train, X_test, Y_test):
    print("\nRunning Symbolic Regression...")
    sym_reg = SymbolicRegressor(
        population_size=5000, generations=30, stopping_criteria=0.01,
        function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt'),
        parsimony_coefficient=0.01, random_state=42
    )

    # Train Symbolic Regressor
    sym_reg.fit(X_train, Y_train)

    # Evaluate on training set
    Y_train_pred = sym_reg.predict(X_train)
    Y_test_pred = sym_reg.predict(X_test)

    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)

    print(f"Symbolic Regression Training R²: {r2_train:.4f}")
    print(f"Symbolic Regression Test R²: {r2_test:.4f}")
    print("Derived Symbolic Equation:")
    print(sym_reg._program)  # Display the derived equation

    return sym_reg


def exp_function(x):
    with np.errstate(over='ignore'):  # Handle overflow errors gracefully
        return np.where(x < 700, np.exp(x), 1e308)  # Limit to avoid overflow

# Wrap the function for gplearn
exp = make_function(function=exp_function, name="exp", arity=1)

def run_RefinedSymbolicRegression(X_train, Y_train, X_test, Y_test):
    print("\nRunning Symbolic Regression with Woodcock Baseline and Cp...")

    # Prepare the data
    Cp_train = X_train[:, 2]  # Extract Cp from training data
    Cp_test = X_test[:, 2]  # Extract Cp from test data

    X_train_new, Y_Woodcock_train = prepare_data_with_Woodcock(X_train, Cp_train)
    X_test_new, Y_Woodcock_test = prepare_data_with_Woodcock(X_test, Cp_test)

    # Symbolic Regression
    sym_reg = SymbolicRegressor(
        population_size=8000,
        generations=50,
        stopping_criteria=0.001,
        function_set=('add', 'sub', 'mul', 'div', 'log', 'sqrt'),
        parsimony_coefficient=0.01,
        verbose=1,
        random_state=42
    )

    # Fit Symbolic Regression model
    sym_reg.fit(X_train_new, Y_train)

    # Predict and evaluate
    Y_train_pred = sym_reg.predict(X_train_new)
    Y_test_pred = sym_reg.predict(X_test_new)

    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)

    print(f"Symbolic Regression Training R²: {r2_train:.4f}")
    print(f"Symbolic Regression Test R²: {r2_test:.4f}")
    print("Derived Symbolic Equation:")
    print(sym_reg._program)

    # Call the visualization function
    visualize_symbolic_regression(Y_train, Y_train_pred, Y_test, Y_test_pred)

    return sym_reg



def plot_learning(num_epochs, loss_values, batch_size, learning_rate):
    step = np.linspace(0, num_epochs, len(loss_values))
    fig1, ax = plt.subplots(figsize=(8, 5))
    plt.plot(step, np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig1.savefig('LearningCurve_%i_%i_%s.jpg' % (num_epochs, batch_size, str(learning_rate)), bbox_inches='tight')

def plt_regression(y_true, y_pred, stage='stage', title='title'):
    fig1, ax = plt.subplots(figsize=(8, 5))
    plt.title("Regression results at the %s set" % stage)
    plt.plot(y_true, y_true, color='black')
    plt.scatter(y_true, y_pred.detach().numpy(), color='orange')
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.show()
    fig1.savefig(title, bbox_inches='tight')
    # plt.close()


def plot_learning_curve(loss_values):
    plt.figure()
    plt.plot(loss_values)
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

# Main script
workdir = os.getcwd()
datadir = os.path.join(workdir, 'DataSets_visc')
if not os.path.exists(datadir):
    raise FileNotFoundError(f"Data directory {datadir} not found.")
os.chdir(datadir)

# dataset = ['Borgelt_1990_47.csv','Galliero_2005_80.csv', 'Gosling_1973_3.csv', 'Heyes_1983_3.csv','Heyes_1988_146.csv',
#            'Heyes_1990_26.csv', 'Hoheisel_1990_3.csv', 'Meier_2004_344.csv', 'Michaels_1985_36.csv', 'Moutain_2006_14.csv',
#            'Rowley_1997_117.csv', 'Schoen_1985_12.csv', 'Vasquez_2005_105.csv']

dataset = 'hPFMPCD_visco_sine.csv'
#dataset = 'hPFMPCD_visco_130angle.csv'
#dataset = 'hPFMPCD_visco_data.csv'


dfs = []
if isinstance(dataset, str):
    dataset = [dataset]  # Convert single filename to list

for data in dataset:
    try:
        df = pd.read_csv(data)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {data}")

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = shuffle(combined_df).reset_index(drop=True)

Temp = combined_df['Temperature']
Dens = combined_df['Density']
Cp = combined_df['Cp']
Visc = combined_df['ShearViscosity']
Visc_norm = (Visc - Visc.min()) / (Visc.max() - Visc.min())

X = np.column_stack((Temp, Dens, Cp))
Y = Visc_norm.to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=10)
train_data = Data(X_train, Y_train)
test_data = Data(X_test, Y_test)

train_dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, drop_last=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, drop_last=True)

# Run
run_SymbolicRegression_with_Cp(X_train, Y_train, X_test, Y_test)
