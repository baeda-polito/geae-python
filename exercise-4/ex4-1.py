import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

model = MLP(input_size=1, hidden_size=16, output_size=1)

criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=0.01)

inputs = torch.linspace(-10, 10, 1000).view(-1, 1)
targets = 0.5 * inputs + torch.sin(inputs)

epochs = 100  # Number of epochs for training

batch_size = 18

# Create a dataset from inputs and targets
dataset = TensorDataset(inputs, targets)

# Create DataLoader to handle batching
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for batch_inputs, batch_targets in dataloader:
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

predictions = model(inputs)

# plot predictions vs targets
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(inputs, targets, label='Targets')
plt.scatter(inputs, predictions.detach().numpy(), label='Predictions')
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.title('Predictions vs Targets')
plt.legend()
plt.show()