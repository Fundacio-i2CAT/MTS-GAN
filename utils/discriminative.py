import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LSTMModelDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMModelDiscriminator, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        out, (hn, cn) = self.lstm(x)

        # We only need the output from the last time step
        out = self.fc(hn[-1, :, :])
        return out


def discriminative_score_metrics(real_data, generated_data, device):
    """Binary classification"""

    # Network parameters
    input_dim = real_data.shape[2]
    num_layers = 2
    hidden_dim = 4
    iterations = 5000
    batch_size = 128
    learning_rate = 1e-4

    # Instantiate the model, loss, and optimizer
    model = LSTMModelDiscriminator(input_dim, hidden_dim, num_layers)

    generated_x = torch.tensor(generated_data, dtype=torch.float32)
    generated_y = torch.zeros(len(generated_data))

    real_x = torch.tensor(real_data, dtype=torch.float32)
    real_y = torch.ones(len(real_data))

    combined_x = torch.cat((generated_x, real_x), dim=0).numpy()
    combined_y = torch.cat((generated_y, real_y), dim=0).numpy()

    train_x, test_x, train_y, test_y = train_test_split(
        combined_x,
        combined_y,
        test_size=0.2,
        shuffle=True
    )
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Starting training...")
    for epoch in range(iterations):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{iterations}, Loss: {loss.item():.4f}")

    # --- Test the trained model on the original data ---
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            y_prob_pred = torch.sigmoid(pred)
            predictions.append(y_prob_pred.cpu().numpy())
            labels.append(y_batch.cpu().numpy())

    predictions_tensor = np.concatenate(predictions, axis=0)
    targets_tensor = np.concatenate(labels, axis=0)

    acc = accuracy_score(targets_tensor, (predictions_tensor > 0.5))
    discriminative_score = np.abs(0.5 - acc)

    return acc, discriminative_score


if __name__ == '__main__':

    real_data = np.random.rand(10, 300, 2)
    generated_data = np.random.rand(10, 300, 2)

    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run the metrics calculation
    disc_score = discriminative_score_metrics(real_data, generated_data, device)
    print(f"acc, disc_score: {disc_score}")
