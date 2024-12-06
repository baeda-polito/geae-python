import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', apply_relu_output=False):
        """
        Parameters:
        - input_size (int): The size of the input layer.
        - hidden_sizes (list of int): A list containing the sizes of each hidden layer.
        - output_size (int): The size of the output layer.
        - activation (str): The activation function for hidden layers ('relu' or 'tanh').
        - apply_relu_output (bool): Whether to apply ReLU to the output layer.
        """
        super(MLP, self).__init__()

        # Store activation function type
        if activation not in ['relu', 'tanh']:
            raise ValueError("Activation function must be either 'relu' or 'tanh'")
        self.activation = activation
        self.apply_relu_output = apply_relu_output

        # Create the list of layers
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            previous_size = hidden_size

        # Output layer
        layers.append(nn.Linear(previous_size, output_size))

        # Register the layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Apply hidden layers with the chosen activation function
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == 'relu':
                x = torch.relu(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)

        # Apply the output layer
        x = self.layers[-1](x)

        # Optionally apply ReLU to the output layer
        if self.apply_relu_output:
            x = torch.relu(x)

        return x

    def predict(self, df):
        """
        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the features.

        Returns:
        - np.ndarray: The prediction as a NumPy array.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            x = torch.tensor(df.values, dtype=torch.float32)
            output = self.forward(x)
        return np.array(output.numpy(), dtype=float)

def create_forecasting_datasets(data, lag_variables, forward_variables, static_variables=None, lag_steps=3,
                                forward_steps=1):
    """
    Builds a dataset with lagged input features, forward-stepped outputs, and static features for time series forecasting.

    Parameters:
        data (pd.DataFrame): The original dataset containing time series data.
        lag_variables (list): List of column names used as lagged features.
        forward_variables (list): List of column names used as output features.
        static_variables (list, optional): List of column names to be included as-is without lagging or forwarding.
        lag_steps (int): Number of lagged steps for input features.
        forward_steps (int): Number of forward steps for outputs.

    Returns:
        pd.DataFrame: A DataFrame with lagged inputs, forward-stepped outputs, and static features.
    """

    # Initialize lists to store lagged, forward-stepped, and static columns
    lagged_data = []
    forward_data = []
    static_data = []

    # Generate lagged features for each input feature
    for feature in lag_variables:
        for lag in range(0, lag_steps):
            lagged_data.append(data[feature].shift(lag).rename(f'{feature}_lag_{lag}'))

    # Generate forward-stepped features for each output feature
    for feature in forward_variables:
        for step in range(1, forward_steps + 1):
            forward_data.append(data[feature].shift(-step).rename(f'{feature}_step_{step}'))

    # Add static features without any lagging or forwarding
    if static_variables:
        static_data = [data[feature] for feature in static_variables]

    # Concatenate the lagged, forward, and static data
    lagged_df = pd.concat(lagged_data, axis=1)
    forward_df = pd.concat(forward_data, axis=1)
    static_df = pd.concat(static_data, axis=1) if static_data else pd.DataFrame()

    # Combine all data
    result = pd.concat([lagged_df, static_df, forward_df], axis=1).dropna()

    return result