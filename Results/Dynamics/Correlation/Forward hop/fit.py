import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Define the linear function to fit
def linear(x, a, b):
    return a * x + b

# Function to fit the curve and compute parameters
def fit_and_compute_parameters(filename):
    # Read the file into a DataFrame assuming it has two columns: x and y
    df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['x', 'y'])
    
    # Extract x and y values
    x = df['x']
    y = df['y']
    
    # Fit the curve to the linear function
    popt_linear, pcov_linear = curve_fit(linear, x, y)
    
    # Extract the fitting parameters for y = ax + b
    a_linear, b_linear = popt_linear
    
    # Compute the standard deviations of the parameters
    std_dev_a_linear, std_dev_b_linear = np.sqrt(np.diag(pcov_linear))
    
    # Fit the curve to the function y1 = a1x
    popt_y1, pcov_y1 = curve_fit(lambda x, a: a * x, x, y)
    
    # Extract the fitting parameter for y1 = a1x
    a_y1 = popt_y1[0]
    
    # Compute the standard deviation of the parameter a1
    std_dev_a_y1 = np.sqrt(pcov_y1[0][0])
    
    return a_linear, std_dev_a_linear, b_linear, std_dev_b_linear, a_y1, std_dev_a_y1

if __name__ == "__main__":
    # Check if filename is provided as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]  # Get filename from command-line argument
    a, std_dev_a, b, std_dev_b, a1, std_dev_a1 = fit_and_compute_parameters(filename)
    print("For y = ax + b:")
    print("a =", a, "+/-", std_dev_a)
    print("b =", b, "+/-", std_dev_b)
    print()
    print("For y1 = a1x:")
    print("a1 =", a1, "+/-", std_dev_a1)

