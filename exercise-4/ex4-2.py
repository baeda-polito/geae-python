import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


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

df = pd.read_csv('data/data.csv', decimal=',', sep=';')

df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Hour'].astype(str), format='%d/%m/%Y %H')
df['Year'] = df['datetime'].dt.year

df['P_norm'] = df['P'] / df['P'].max()

train_df = df[df['Year'] == 2015].copy()
test_df = df[df['Year'] == 2016].copy()


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

train_df_prepared = create_forecasting_datasets(train_df,
                                                lag_variables=['P_norm'],
                                                forward_variables=['P_norm'],
                                                lag_steps=4,
                                                forward_steps=1)

test_df_prepared = create_forecasting_datasets(test_df,
                                                lag_variables=['P_norm'],
                                                forward_variables=['P_norm'],
                                                lag_steps=4,
                                                forward_steps=1)

batch_size = 128
hidden_sizes = [16, 8]
activation = 'relu'
apply_relu_output = True
learning_rate = 0.001
epochs = 100

train_features = torch.FloatTensor(train_df_prepared.values[:, 0:4])
train_targets = torch.FloatTensor(train_df_prepared.values[:, 4:5])
dataset = TensorDataset(train_features, train_targets)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = MLP(input_size=4, hidden_sizes=hidden_sizes, output_size=1,
            activation=activation, apply_relu_output=apply_relu_output)

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# train the model
epochs = epochs

for epoch in range(epochs):
    epoch_loss = 0.0
    model.train()  # Set the model to training mode

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        optimizer.zero_grad()
        output = model(inputs)

        # Calculate loss
        loss = criterion(output, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the batch loss
        epoch_loss += loss.item()

    # Calculate and print the average epoch loss
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}')


# assing the prediction to training data and testing data
train_df = train_df[train_df.index.isin(train_df_prepared.index)]
test_df = test_df[test_df.index.isin(test_df_prepared.index)]

train_df['prediction'] = model.predict(train_df_prepared.iloc[:,0:4])
test_df['prediction'] = model.predict(test_df_prepared.iloc[:,0:4])

# Calculate global metrics for the entire training set
global_metrics = {
    'Metric': ['Global MSE', 'Global MAPE', 'Global R2'],
    'Value': [
        float(mean_squared_error(train_df['P_norm'], train_df['prediction'])),
        float(mean_absolute_percentage_error(train_df['P_norm'], train_df['prediction'])),
        float(r2_score(train_df['P_norm'], train_df['prediction']))
    ]
}

global_metrics_df = pd.DataFrame(global_metrics)

global_metrics_test = {
    'Metric': ['Global Test MSE', 'Global Test MAPE', 'Global Test R2'],
    'Value': [
        float(mean_squared_error(test_df['P_norm'], test_df['prediction'])),
        float(mean_absolute_percentage_error(test_df['P_norm'], test_df['prediction'])),
        float(r2_score(test_df['P_norm'], test_df['prediction']))
    ]
}

global_metrics_test_df = pd.DataFrame(global_metrics_test)

print("\nGlobal Training Metrics:")
print(global_metrics_df)

print("\nGlobal Test Metrics:")
print(global_metrics_test_df)

# denormalize the prediction
train_df['prediction'] = train_df['prediction'] * df['P'].max()
test_df['prediction'] = test_df['prediction'] * df['P'].max()

# Filter data for each year
years = df['Year'].unique()
fig, axs = plt.subplots(len(years), 1, figsize=(15, 6 * len(years)))

# Plot data for each year in separate subplots
for i, year in enumerate(years):
    ax = axs[i] if len(years) > 1 else axs  # Handle the case when there's only one subplot
    df_year = df[df['Year'] == year]
    train_year = train_df[train_df['Year'] == year]
    test_year = test_df[test_df['Year'] == year]

    ax.plot(df_year['datetime'], df_year['P'], label='Actual', color='blue')
    ax.plot(train_year['datetime'], train_year['prediction'], label='Train Prediction', linestyle='--', color='green')
    ax.plot(test_year['datetime'], test_year['prediction'], label='Test Prediction', linestyle='--', color='red')

    ax.set_xlabel('Datetime')
    ax.set_ylabel('Load (kW)')
    ax.set_title(f'Year: {year}')
    ax.legend()

plt.tight_layout()
plt.show()

# plot residuals distribution
train_residuals = train_df['P'] - train_df['prediction']
test_residuals = test_df['P'] - test_df['prediction']

fig, axs = plt.subplots(1, 2, figsize=(15, 6))

axs[0].hist(train_residuals, bins=50, color='blue', alpha=0.7, label='Train Residuals')
axs[0].set_title('Train Residuals Distribution')
axs[0].set_xlabel('Residuals')
axs[0].set_ylabel('Frequency')
axs[0].legend()

axs[1].hist(test_residuals, bins=50, color='red', alpha=0.7, label='Test Residuals')
axs[1].set_title('Test Residuals Distribution')
axs[1].set_xlabel('Residuals')
axs[1].set_ylabel('Frequency')
axs[1].legend()

plt.tight_layout()
plt.show()
