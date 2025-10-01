"""Ones step ahead forecasting"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error


class LSTMModel(nn.Module):
    """LSTM for Forecasting one step ahead"""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """Forward func"""
        out, (hn, _) = self.lstm(x)

        # We only need the output from the last time step
        out = self.fc(hn[-1, :, :])
        return out


def predictive_score_metrics(real_data, generated_data, device):
    """multivariate, one-step-ahead forecasting"""

    # Network parameters
    input_dim = real_data.shape[2]
    num_layers = 2
    hidden_dim = 4
    iterations = 5000
    batch_size = 128
    learning_rate = 1e-4

    # Instantiate the model, loss, and optimizer
    model = LSTMModel(input_dim, hidden_dim, num_layers)
    # --- Training using Synthetic dataset ---
    # Prepare data for training
    train_x_data = torch.tensor(generated_data[:, :-1, :], dtype=torch.float32)
    train_y_data = torch.tensor(generated_data[:, -1, :], dtype=torch.float32)

    train_dataset = TensorDataset(train_x_data, train_y_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_x_data = torch.tensor(real_data[:, :-1, :], dtype=torch.float32)
    test_y_data = torch.tensor(real_data[:, -1, :], dtype=torch.float32)

    test_dataset = TensorDataset(test_x_data, test_y_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Starting training...")
    for epoch in range(iterations):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{iterations}, Loss: {loss.item():.4f}")

    # --- Test the trained model on the original data ---
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            predictions = model(x_batch)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    predictions_tensor = np.concatenate(all_predictions, axis=0)
    targets_tensor = np.concatenate(all_targets, axis=0)

    mse = mean_squared_error(targets_tensor, predictions_tensor)

    return mse
